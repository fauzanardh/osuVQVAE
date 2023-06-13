from functools import partial
from typing import Tuple, Union

import torch
from einops import rearrange, reduce
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import GroupedResidualVQ

from osu_vqvae.modules.causal_convolution import CausalConv1d
from osu_vqvae.modules.discriminator import MultiScaleDiscriminator
from osu_vqvae.modules.scaler import DownsampleLinear, UpsampleLinear
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


class EncoderAttn(nn.Module):
    def __init__(
        self: "EncoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        attn_depth: Tuple[int] = (2, 2, 2, 4),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.init_conv = CausalConv1d(dim_in, dim_h, 7)

        dims_h = [dim_h * mult for mult in dim_h_mult]
        in_out = list(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Down
        self.downs = nn.ModuleList([])
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            self.downs.append(
                nn.ModuleList(
                    [
                        DownsampleLinear(layer_dim_in, layer_dim_out),
                        LocalTransformerBlock(
                            dim=layer_dim_out,
                            depth=attn_depth[ind],
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            window_size=attn_window_size,
                        ),
                    ],
                ),
            )

        # Middle
        dim_mid = dims_h[-1]
        self.mid_attn = LocalTransformerBlock(
            dim=dim_mid,
            depth=attn_depth[0],
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
        )

        # Project to embedding space
        self.proj = CausalConv1d(dim_mid, dim_emb, 3)

    def forward(self: "EncoderAttn", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = rearrange(x, "b c l -> b l c")

        # Down
        for downsample, attn_block in self.downs:
            x = downsample(x)
            x = attn_block(x)

        # Middle
        x = self.mid_attn(x)

        # Project to embedding space
        # annoyingly, conv1d expects (batch, channels, length)
        # while the rest of the model expects (batch, length, channels)
        x = rearrange(x, "b l c -> b c l")
        x = self.proj(x)
        x = rearrange(x, "b c l -> b l c")

        return x


class DecoderAttn(nn.Module):
    def __init__(
        self: "DecoderAttn",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        attn_depth: Tuple[int] = (2, 2, 2, 4),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        use_tanh: bool = False,
    ) -> None:
        super().__init__()
        self.use_tanh = use_tanh

        dim_h_mult = tuple(reversed(dim_h_mult))
        attn_depth = tuple(reversed(attn_depth))
        dims_h = [dim_h * mult for mult in dim_h_mult]
        in_out = list(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)
        dim_mid = dims_h[0]

        # Project from embedding space
        self.proj = CausalConv1d(dim_emb, dim_mid, 7)

        # Middle
        self.mid_attn = LocalTransformerBlock(
            dim=dim_mid,
            depth=attn_depth[0],
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=attn_window_size,
        )

        # Up
        self.ups = nn.ModuleList([])
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            self.ups.append(
                nn.ModuleList(
                    [
                        UpsampleLinear(layer_dim_in, layer_dim_out),
                        LocalTransformerBlock(
                            dim=layer_dim_out,
                            depth=attn_depth[ind],
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            window_size=attn_window_size,
                        ),
                    ],
                ),
            )

        # End
        self.end_conv = CausalConv1d(dim_h, dim_in, 7)

    def forward(self: "DecoderAttn", x: torch.Tensor) -> torch.Tensor:
        # Project from embedding space
        # annoyingly, conv1d expects (batch, channels, length)
        # while the rest of the model expects (batch, length, channels)
        x = rearrange(x, "b l c -> b c l")
        x = self.proj(x)
        x = rearrange(x, "b c l -> b l c")

        # Middle
        x = self.mid_attn(x)

        # Up
        for upsample, attn_block in self.ups:
            x = upsample(x)
            x = attn_block(x)

        # End
        x = rearrange(x, "b l c -> b c l")
        x = self.end_conv(x)

        return torch.tanh(x) if self.use_tanh else x


class VQVAE(nn.Module):
    def __init__(
        self: "VQVAE",
        dim_in: int,
        dim_h: int,
        dim_emb: int,
        n_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        attn_depth: Tuple[int] = (2, 2, 2, 4),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        num_codebooks: int = 8,
        vq_groups: int = 1,
        vq_decay: float = 0.95,
        vq_commitment_weight: float = 1.0,
        vq_quantize_dropout_cutoff_index: int = 1,
        vq_stochastic_sample_codes: bool = False,
        discriminator_scales: Tuple[float] = (1.0, 0.5, 0.25),
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
            dim_in,
            dim_h,
            dim_emb,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
        )
        self.encoder_norm = nn.LayerNorm(dim_emb)

        self.decoder = DecoderAttn(
            dim_in,
            dim_h,
            dim_emb,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
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
        self.discriminators = nn.ModuleList(
            [
                MultiScaleDiscriminator(dim_in=dim_in)
                for _ in range(len(discriminator_scales))
            ],
        )
        disc_relative_factors = [
            int(s1 / s2)
            for s1, s2 in zip(discriminator_scales[:-1], discriminator_scales[1:])
        ]
        self.discriminator_downsamples = nn.ModuleList(
            [nn.Identity()]
            + [
                nn.AvgPool1d(2 * factor, stride=factor, padding=factor)
                for factor in disc_relative_factors
            ],
        )

        self.gen_loss = hinge_generator_loss if use_hinge_loss else bce_generator_loss
        self.disc_loss = (
            hinge_discriminator_loss if use_hinge_loss else bce_discriminator_loss
        )

    def encode(self: "VQVAE", x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.encoder_norm(x)
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
        orig_sig = sig.clone()

        sig, _, commit_loss = self.encode(sig)
        recon_sig = self.decode(sig)

        if not (return_loss or return_disc_loss):
            return recon_sig

        # Discriminator
        if return_disc_loss:
            real, fake = orig_sig, recon_sig.detach()
            disc_loss = 0.0
            for discriminator, downsample in zip(
                self.discriminators,
                self.discriminator_downsamples,
            ):
                real, fake = map(downsample, (real, fake))
                real_disc_logits, fake_disc_logits = map(
                    discriminator,
                    (real.requires_grad_(), fake),
                )
                _disc_loss = self.disc_loss(real_disc_logits, fake_disc_logits)

                if add_gradient_penalty:
                    _disc_loss += gradient_penalty(real, _disc_loss)

                disc_loss += _disc_loss

            if return_recons:
                return disc_loss, recon_sig
            else:
                return disc_loss

        recon_loss = F.mse_loss(orig_sig, recon_sig)

        # Generator
        real, fake = orig_sig, recon_sig
        gen_loss = 0.0
        disc_intermediates = []
        for discriminator, downsample in zip(
            self.discriminators,
            self.discriminator_downsamples,
        ):
            real, fake = map(downsample, (real, fake))
            (real_disc_logits, real_disc_intermediates), (
                fake_disc_logits,
                fake_disc_intermediates,
            ) = map(partial(discriminator, return_intermediates=True), (real, fake))
            disc_intermediates.append(
                (real_disc_intermediates, fake_disc_intermediates),
            )

            gen_loss += self.gen_loss(fake_disc_logits)

        feature_loss = 0.0
        for real_disc_intermediates, fake_disc_intermediates in disc_intermediates:
            for real_disc_intermediate, fake_disc_intermediate in zip(
                real_disc_intermediates,
                fake_disc_intermediates,
            ):
                feature_loss += F.l1_loss(
                    real_disc_intermediate,
                    fake_disc_intermediate,
                )

        loss = (
            recon_loss * self.recon_loss_weight
            + gen_loss * self.gan_loss_weight
            + feature_loss * self.feature_loss_weight
            + commit_loss.sum()
        )
        if return_recons:
            return loss, recon_sig
        else:
            return loss
