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
from osu_vqvae.modules.utils import gradient_penalty, log


def bce_discriminator_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_discriminator_fake_loss(fake: torch.Tensor) -> torch.Tensor:
    return -log(1 - torch.sigmoid(fake)).mean()


def bce_discriminator_real_loss(real: torch.Tensor) -> torch.Tensor:
    return -log(torch.sigmoid(real)).mean()


def bce_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    return -log(torch.sigmoid(fake)).mean()


def hinge_discriminator_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_discriminator_fake_loss(fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1 + fake).mean()


def hinge_discriminator_real_loss(real: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real).mean()


def hinge_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


class EncoderBlock(nn.Sequential):
    def __init__(
        self: "EncoderBlock",
        dim_in: int,
        dim_out: int,
        stride: int,
        dilations: Tuple[int] = (1, 3, 9),
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__(
            ResidualBlock(dim_in, dim_in, dilations[0], squeeze_excite=squeeze_excite),
            ResidualBlock(dim_in, dim_in, dilations[1], squeeze_excite=squeeze_excite),
            ResidualBlock(dim_in, dim_in, dilations[2], squeeze_excite=squeeze_excite),
            CausalConv1d(dim_in, dim_out, 2 * stride, stride=stride),
        )


class EncoderAttn(nn.Module):
    def __init__(
        self: "EncoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        strides: Tuple[int] = (2, 2, 2, 2),
        res_dilations: Tuple[int] = (1, 3, 9),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 256,
        attn_dynamic_pos_bias: bool = False,
        attn_alibi_pos_bias: bool = False,
    ) -> None:
        super().__init__()

        assert not (
            attn_dynamic_pos_bias and attn_alibi_pos_bias
        ), "Cannot have both dynamic and alibi positional bias"

        dims_h = tuple((dim_h * m for m in dim_h_mult))
        dims_h = (dim_h, *dims_h)
        in_out = tuple(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Down
        encoder_blocks = []
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            stride = strides[ind]
            encoder_blocks.append(
                EncoderBlock(
                    layer_dim_in,
                    layer_dim_out,
                    stride,
                    dilations=res_dilations,
                ),
            )

        self.downs = nn.Sequential(
            CausalConv1d(dim_in, dim_h, 7),
            *encoder_blocks,
            CausalConv1d(dims_h[-1], dim_emb, 3),
        )
        self.attn = LocalTransformerBlock(
            dim=dim_emb,
            depth=attn_depth,
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
            dynamic_pos_bias=attn_dynamic_pos_bias,
            alibi_pos_bias=attn_alibi_pos_bias,
        )

    def forward(self: "EncoderAttn", x: torch.Tensor) -> torch.Tensor:
        # Downs
        x = self.downs(x)

        # Rearrange to (batch, length, channels)
        x = rearrange(x, "b c l -> b l c")

        # Attn
        x = self.attn(x)

        return x


class DecoderBlock(nn.Sequential):
    def __init__(
        self: "DecoderBlock",
        dim_in: int,
        dim_out: int,
        stride: int,
        dilations: Tuple[int] = (1, 3, 9),
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__(
            CausalConvTranspose1d(dim_in, dim_out, 2 * stride, stride=stride),
            ResidualBlock(
                dim_out,
                dim_out,
                dilations[0],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_out,
                dim_out,
                dilations[1],
                squeeze_excite=squeeze_excite,
            ),
            ResidualBlock(
                dim_out,
                dim_out,
                dilations[2],
                squeeze_excite=squeeze_excite,
            ),
        )


class DecoderAttn(nn.Module):
    def __init__(
        self: "DecoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        strides: Tuple[int] = (2, 2, 2, 2),
        res_dilations: Tuple[int] = (1, 3, 9),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 256,
        attn_dynamic_pos_bias: bool = False,
        attn_alibi_pos_bias: bool = False,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()

        assert not (
            attn_dynamic_pos_bias and attn_alibi_pos_bias
        ), "Cannot have both dynamic and alibi positional bias"

        self.use_tanh = use_tanh

        dims_h = tuple((dim_h * m for m in dim_h_mult))
        dims_h = (dim_h, *dims_h)
        strides = tuple(reversed(strides))
        in_out = reversed(tuple(zip(dims_h[:-1], dims_h[1:])))
        in_out = tuple(in_out)
        num_layers = len(in_out)

        # Up
        decoder_blocks = []
        for ind in range(num_layers):
            layer_dim_out, layer_dim_in = in_out[ind]
            stride = strides[ind]
            decoder_blocks.append(
                DecoderBlock(
                    layer_dim_in,
                    layer_dim_out,
                    stride,
                    dilations=res_dilations,
                ),
            )
        self.attn = LocalTransformerBlock(
            dim=dim_emb,
            depth=attn_depth,
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
            dynamic_pos_bias=attn_dynamic_pos_bias,
            alibi_pos_bias=attn_alibi_pos_bias,
        )
        self.ups = nn.Sequential(
            CausalConv1d(dim_emb, dims_h[-1], 7),
            *decoder_blocks,
            CausalConv1d(dims_h[0], dim_in, 7),
        )

    def forward(self: "DecoderAttn", x: torch.Tensor) -> torch.Tensor:
        # Attn
        x = self.attn(x)

        # Rearrange to (batch, channels, length)
        x = rearrange(x, "b l c -> b c l")

        # Ups
        x = self.ups(x)

        return torch.tanh(x) if self.use_tanh else x


class VQVAE(nn.Module):
    def __init__(
        self: "VQVAE",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        n_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        strides: Tuple[int] = (2, 2, 2, 2),
        res_dilations: Tuple[int] = (1, 3, 9),
        attn_depth: int = 2,
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 256,
        attn_dynamic_pos_bias: bool = False,
        attn_alibi_pos_bias: bool = False,
        vq_num_codebooks: int = 8,
        vq_groups: int = 1,
        vq_decay: float = 0.95,
        vq_commitment_weight: float = 0.0,
        vq_quantize_dropout_cutoff_index: int = 1,
        vq_stochastic_sample_codes: bool = False,
        discriminator_layers: int = 4,
        use_tanh: bool = False,
        use_hinge_loss: bool = False,
        split_disc_loss: bool = False,
        recon_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        feature_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.split_disc_loss = split_disc_loss
        self.recon_loss_weight = recon_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.encoder = EncoderAttn(
            dim_in=dim_in,
            dim_h=dim_h,
            dim_emb=dim_emb,
            dim_h_mult=dim_h_mult,
            strides=strides,
            res_dilations=res_dilations,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
            attn_dynamic_pos_bias=attn_dynamic_pos_bias,
            attn_alibi_pos_bias=attn_alibi_pos_bias,
        )
        self.vq = GroupedResidualVQ(
            dim=dim_emb,
            codebook_size=n_emb,
            num_quantizers=vq_num_codebooks,
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
        self.decoder = DecoderAttn(
            dim_in=dim_in,
            dim_h=dim_h,
            dim_emb=dim_emb,
            dim_h_mult=dim_h_mult,
            strides=strides,
            res_dilations=res_dilations,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
            attn_dynamic_pos_bias=attn_dynamic_pos_bias,
            attn_alibi_pos_bias=attn_alibi_pos_bias,
            use_tanh=use_tanh,
        )

        # Discriminator
        layer_mults = [2**i for i in range(discriminator_layers)]
        layer_dims = [dim_h * mult for mult in layer_mults]
        self.discriminator = Discriminator(dim_in=dim_in, h_dims=layer_dims)

        self.gen_loss = hinge_generator_loss if use_hinge_loss else bce_generator_loss
        if not self.split_disc_loss:
            self.disc_loss = (
                hinge_discriminator_loss if use_hinge_loss else bce_discriminator_loss
            )
        else:
            self.disc_fake_loss = (
                hinge_discriminator_fake_loss
                if use_hinge_loss
                else bce_discriminator_fake_loss
            )
            self.disc_real_loss = (
                hinge_discriminator_real_loss
                if use_hinge_loss
                else bce_discriminator_real_loss
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

    def forward(  # noqa: C901
        self: "VQVAE",
        x: torch.Tensor,
        seperate_loss: bool = False,
        return_loss: float = False,
        return_disc_loss: float = False,
        return_recons: float = True,
        add_gradient_penalty: float = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        quantized, _, commit_loss = self.encode(x)
        recon_x = self.decode(quantized)

        if not (return_loss or return_disc_loss):
            return recon_x

        # Discriminator
        if return_disc_loss:
            recon_x.detach_()
            x.requires_grad_()

            real_disc_logits, fake_disc_logits = map(
                self.discriminator,
                (x, recon_x),
            )

            if not self.split_disc_loss:
                loss = self.disc_loss(fake_disc_logits, real_disc_logits)

                if add_gradient_penalty:
                    loss += gradient_penalty(x, real_disc_logits)

                if return_recons:
                    return loss, recon_x
                else:
                    return loss
            else:
                fake_loss = self.disc_fake_loss(fake_disc_logits)
                real_loss = self.disc_real_loss(real_disc_logits)

                combined_loss = fake_loss + real_loss

                if add_gradient_penalty:
                    combined_loss += gradient_penalty(x, real_disc_logits)

                if return_recons:
                    return fake_loss, real_loss, combined_loss, recon_x
                else:
                    return fake_loss, real_loss, combined_loss

        # Recon loss
        recon_loss = F.l1_loss(x, recon_x)

        # Generator
        disc_intermediates = []
        (real_disc_logits, real_disc_intermediates), (
            fake_disc_logits,
            fake_disc_intermediates,
        ) = map(
            partial(self.discriminator, return_intermediates=True),
            (x, recon_x),
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

        if not seperate_loss:
            loss = (
                recon_loss * self.recon_loss_weight
                + gan_loss * self.gan_loss_weight
                + feature_loss * self.feature_loss_weight
                + commit_loss.sum()
            )
        else:
            loss = (
                recon_loss * self.recon_loss_weight,
                gan_loss * self.gan_loss_weight,
                feature_loss * self.feature_loss_weight,
                commit_loss.sum(),
            )
        if return_recons:
            return loss, recon_x
        else:
            return loss
