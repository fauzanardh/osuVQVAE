from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualVQ

from osu_vqvae.modules.discriminator import Discriminator
from osu_vqvae.modules.scaler import DownsampleLinear, UpsampleLinear
from osu_vqvae.modules.transformer import LocalTransformerBlock


def log(t: torch.Tensor, eps: int = 1e-10) -> torch.Tensor:
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
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        attn_depth: Tuple[int] = (2, 2, 2, 4),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Conv1d(dim_in, dim_h, 7, padding=3)

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

    def forward(self: "EncoderAttn", x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = rearrange(x, "b c l -> b l c")

        # Down
        for downsample, attn_block in self.downs:
            x = downsample(x)
            x = attn_block(x)

        # Middle
        x = self.mid_attn(x)

        return x


class DecoderAttn(nn.Module):
    def __init__(
        self: "DecoderAttn",
        dim_in: int,
        dim_h: int,
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

        # Middle
        dim_mid = dims_h[0]
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
        self.end_conv = nn.Conv1d(dim_h, dim_in, 1)

    def forward(self: "DecoderAttn", x: torch.Tensor) -> torch.Tensor:
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
        n_emb: int,
        dim_h_mult: Tuple[int] = (1, 2, 4, 8),
        attn_depth: Tuple[int] = (2, 2, 2, 4),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_window_size: int = 1024,
        num_codebooks: int = 8,
        vq_decay: float = 0.9,
        rvq_quantize_dropout: bool = True,
        use_discriminator: bool = False,
        discriminator_layers: int = 4,
        use_tanh: bool = False,
        use_hinge_loss: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = EncoderAttn(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
        )
        self.encoder_norm = nn.LayerNorm(dim_h * dim_h_mult[-1])

        self.decoder = DecoderAttn(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_window_size=attn_window_size,
            use_tanh=use_tanh,
        )
        self.vq = ResidualVQ(
            dim=dim_h * dim_h_mult[-1],
            num_quantizers=num_codebooks,
            codebook_size=n_emb,
            decay=vq_decay,
            quantize_dropout=num_codebooks > 1 and rvq_quantize_dropout,
            commitment_weight=0.0,
            kmeans_init=True,
            kmeans_iters=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.1,
            shared_codebook=True,
        )

        # Discriminator
        if use_discriminator:
            layer_mults = [2**x for x in range(discriminator_layers)]
            layer_dims = [dim_h * mult for mult in layer_mults]
            self.discriminator = Discriminator(
                dim_in,
                layer_dims,
            )

            self.gen_loss = (
                hinge_generator_loss if use_hinge_loss else bce_generator_loss
            )
            self.disc_loss = (
                hinge_discriminator_loss if use_hinge_loss else bce_discriminator_loss
            )

    def encode(self: "VQVAE", x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return self.vq(x)

    def decode(self: "VQVAE", x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def decode_from_ids(self: "VQVAE", ids: torch.Tensor) -> torch.Tensor:
        codes = self.codebook[ids]
        fmap = self.vq.project_out(codes)
        # fmap = rearrange(fmap, "b l c -> b c l")
        return self.decode(fmap)

    def forward(
        self: "VQVAE",
        sig: torch.Tensor,
        return_loss: float = False,
        return_disc_loss: float = False,
        return_recons: float = True,
        add_gradient_penalty: float = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # limit sig to [-1, 1] range
        sig = torch.clamp(sig, -1, 1)

        fmap, _, _ = self.encode(sig)
        fmap = self.decode(fmap)

        if not (return_loss or return_disc_loss):
            return fmap

        # Discriminator
        if return_disc_loss and hasattr(self, "discriminator"):
            fmap.detach_()
            sig.requires_grad_()

            fmap_disc_logits, sig_disc_logits = map(self.discriminator, (fmap, sig))
            loss = self.disc_loss(fmap_disc_logits, sig_disc_logits)

            if add_gradient_penalty:
                loss += gradient_penalty(sig, sig_disc_logits)

            if return_recons:
                return loss, fmap
            else:
                return loss

        recon_loss = F.mse_loss(fmap, sig)

        # Generator
        gen_loss = 0
        if hasattr(self, "discriminator"):
            gen_loss = self.gen_loss(self.discriminator(fmap))

        loss = recon_loss + gen_loss
        if return_recons:
            return loss, fmap
        else:
            return loss
