from functools import partial
from typing import Tuple, Union

import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import GroupedResidualVQ

from osu_vqvae.modules.causal_convolution import CausalConv1d, CausalConvTranspose1d
from osu_vqvae.modules.discriminator import Discriminator
from osu_vqvae.modules.residual import ResidualBlock
from osu_vqvae.modules.transformer import LocalTransformerBlock


def log(t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(t.dtype).eps
    return torch.log(t + eps)


def gradient_penalty(
    sig: torch.Tensor,
    output: torch.Tensor,
    weight: int = 10,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=sig,
        grad_outputs=torch.ones_like(output, device=sig.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return weight * ((gradients.norm(2, dim=-1) - 1) ** 2).mean()


def bce_discriminator_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    return -log(torch.sigmoid(fake)).mean()


def hinge_discriminator_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


class EncoderBlock(nn.Module):
    def __init__(
        self: "EncoderBlock",
        dim_in: int,
        dim_out: int,
        stride: int,
        dilations: Tuple[int] = (1, 3, 9),
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ResidualBlock(
                dim_in=dim_in,
                dim_out=dim_in,
                dilation=dilations[0],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_in=dim_in,
                dim_out=dim_in,
                dilation=dilations[1],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_in=dim_in,
                dim_out=dim_in,
                dilation=dilations[2],
                squeeze_excite=squeeze_excite,
            ),
            CausalConv1d(dim_in, dim_out, stride * 2, stride=stride),
        )

    def forward(self: "EncoderBlock", x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderAttn(nn.Module):
    def __init__(
        self: "EncoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (2, 4, 8, 16),
        strides: Tuple[int] = (2, 4, 4, 8),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        attn_dynamic_pos_bias: bool = False,
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__()
        dims_h = tuple((dim_h * m for m in dim_h_mult))
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Down
        down_blocks = []
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            stride = strides[ind]
            down_blocks.append(
                EncoderBlock(
                    dim_in=layer_dim_in,
                    dim_out=layer_dim_out,
                    stride=stride,
                    squeeze_excite=squeeze_excite,
                ),
            )

        self.downs = nn.Sequential(
            CausalConv1d(dim_in, dim_h, 7),
            *down_blocks,
            CausalConv1d(dims_h[-1], dim_emb, 3),
        )
        self.attn = LocalTransformerBlock(
            dim=dim_emb,
            depth=attn_depth,
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
            dynamic_pos_bias=attn_dynamic_pos_bias,
        )

    def forward(self: "EncoderAttn", x: torch.Tensor) -> torch.Tensor:
        x = self.downs(x)

        # Attention
        # rearrange to (batch, length, channels)
        x = rearrange(x, "b c l -> b l c")
        x = self.attn(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self: "DecoderBlock",
        dim_in: int,
        dim_out: int,
        stride: int,
        dilations: Tuple[int] = (1, 3, 9),
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__()
        # even_stride = stride % 2 == 0
        # padding = (stride + (0 if even_stride else 1)) // 2
        # output_padding = 0 if even_stride else 1

        self.layers = nn.Sequential(
            CausalConvTranspose1d(dim_in, dim_out, 2 * stride, stride=stride),
            ResidualBlock(
                dim_in=dim_out,
                dim_out=dim_out,
                dilation=dilations[0],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_in=dim_out,
                dim_out=dim_out,
                dilation=dilations[1],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_in=dim_out,
                dim_out=dim_out,
                dilation=dilations[2],
                squeeze_excite=squeeze_excite,
            ),
        )

    def forward(self: "DecoderBlock", x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderAttn(nn.Module):
    def __init__(
        self: "DecoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (2, 4, 8, 16),
        strides: Tuple[int] = (2, 4, 4, 8),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        attn_dynamic_pos_bias: bool = False,
        squeeze_excite: bool = False,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()
        self.use_tanh = use_tanh

        dims_h = tuple((dim_h * m for m in dim_h_mult))
        dims_h = (dim_h, *dims_h)
        strides = tuple(reversed(strides))
        in_out = reversed(tuple(zip(dims_h[:-1], dims_h[1:])))
        in_out = tuple(in_out)
        num_layers = len(in_out)

        self.attn = LocalTransformerBlock(
            dim=dim_emb,
            depth=attn_depth,
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
            dynamic_pos_bias=attn_dynamic_pos_bias,
        )

        # Up
        up_blocks = []
        for ind in range(num_layers):
            layer_dim_out, layer_dim_in = in_out[ind]
            stride = strides[ind]
            up_blocks.append(
                DecoderBlock(
                    dim_in=layer_dim_in,
                    dim_out=layer_dim_out,
                    stride=stride,
                    squeeze_excite=squeeze_excite,
                ),
            )

        self.ups = nn.Sequential(
            CausalConv1d(dim_emb, dims_h[-1], 7),
            *up_blocks,
            CausalConv1d(dim_h, dim_in, 7),
        )

    def forward(self: "DecoderAttn", x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)

        # Up
        # rearrange to (batch, channels, length)
        x = rearrange(x, "b l c -> b c l")
        x = self.ups(x)

        return torch.tanh(x) if self.use_tanh else x


class VQVAE(nn.Module):
    def __init__(
        self: "VQVAE",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        n_emb: int,
        dim_h_mult: Tuple[int] = (2, 4, 8, 16),
        strides: Tuple[int] = (2, 4, 4, 8),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        attn_dynamic_pos_bias: bool = False,
        num_codebooks: int = 8,
        vq_groups: int = 1,
        vq_decay: float = 0.95,
        vq_commitment_weight: float = 1.0,
        vq_quantize_dropout_cutoff_index: int = 1,
        vq_stochastic_sample_codes: bool = False,
        discriminator_layers: int = 4,
        use_tanh: bool = False,
        use_hinge_loss: bool = False,
        recon_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        feature_loss_weight: float = 100.0,
    ) -> None:
        super().__init__()
        self.recon_loss_weight = recon_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.encoder = EncoderAttn(
            dim_in=dim_in,
            dim_h=dim_h,
            dim_emb=dim_emb,
            dim_h_mult=dim_h_mult,
            strides=strides,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
            attn_dynamic_pos_bias=attn_dynamic_pos_bias,
        )

        self.decoder = DecoderAttn(
            dim_in=dim_in,
            dim_h=dim_h,
            dim_emb=dim_emb,
            dim_h_mult=dim_h_mult,
            strides=strides,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
            attn_dynamic_pos_bias=attn_dynamic_pos_bias,
            use_tanh=use_tanh,
        )
        self.vq = GroupedResidualVQ(
            dim=dim_emb,
            num_quantizers=num_codebooks,
            codebook_size=n_emb,
            groups=vq_groups,
            decay=vq_decay,
            commitment_weight=vq_commitment_weight,
            quantize_dropout_multiple_of=1,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=vq_quantize_dropout_cutoff_index,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            stochastic_sample_codes=vq_stochastic_sample_codes,
        )

        # Discriminator
        layer_mults = [2**i for i in range(discriminator_layers)]
        layer_dims = [dim_h * mult for mult in layer_mults]
        self.discriminator = Discriminator(dim_in=dim_in, h_dims=layer_dims)

        self.gen_loss = hinge_generator_loss if use_hinge_loss else bce_generator_loss
        self.disc_loss = (
            hinge_discriminator_loss if use_hinge_loss else bce_discriminator_loss
        )

    def encode(self: "VQVAE", x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.vq(x)

    def decode(self: "VQVAE", x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def decode_from_ids(self: "VQVAE", quantized_ids: torch.Tensor) -> torch.Tensor:
        codes = self.vq.get_codes_from_indices(quantized_ids)
        x = reduce(codes, "g q b l c -> b l (g c)", "sum")
        return self.decode(x)

    def forward(
        self: "VQVAE",
        sig: torch.Tensor,
        return_loss: float = False,
        return_disc_loss: float = False,
        return_recons: float = True,
        add_gradient_penalty: float = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        quantized, _, commit_loss = self.encode(sig)
        recon_sig = self.decode(quantized)

        if not (return_loss or return_disc_loss):
            return recon_sig

        # Discriminator
        if return_disc_loss:
            recon_sig.detach_()
            sig.requires_grad_()

            real_disc_logits, fake_disc_logits = map(
                self.discriminator,
                (sig, recon_sig),
            )
            loss = self.disc_loss(real_disc_logits, fake_disc_logits)

            if add_gradient_penalty:
                loss += gradient_penalty(sig, real_disc_logits)

            if return_recons:
                return loss, recon_sig
            else:
                return loss

        # Recon loss
        recon_loss = F.mse_loss(sig, recon_sig)

        # Generator
        disc_intermediates = []
        (real_disc_logits, real_disc_intermediates), (
            fake_disc_logits,
            fake_disc_intermediates,
        ) = map(
            partial(self.discriminator, return_intermediates=True),
            (sig, recon_sig),
        )
        disc_intermediates.append((real_disc_intermediates, fake_disc_intermediates))
        gan_loss = self.gen_loss(fake_disc_logits)

        # Features
        feature_losses = []
        for real_disc_intermediates, fake_disc_intermediates in disc_intermediates:
            losses = [
                F.l1_loss(real_disc_intermediate, fake_disc_intermediate)
                for real_disc_intermediate, fake_disc_intermediate in zip(
                    real_disc_intermediates,
                    fake_disc_intermediates,
                )
            ]
            feature_losses.extend(losses)
        feature_loss = torch.stack(feature_losses).mean()

        loss = (
            recon_loss * self.recon_loss_weight
            + gan_loss * self.gan_loss_weight
            + feature_loss * self.feature_loss_weight
            + commit_loss.sum()
        )
        if return_recons:
            return loss, recon_sig
        else:
            return loss
