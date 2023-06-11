import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from vector_quantize_pytorch import ResidualVQ

from osu_vqvae.modules.scaler import UpsampleLinear, DownsampleLinear
from osu_vqvae.modules.discriminator import Discriminator
from osu_vqvae.modules.transformer import LocalTransformerBlock


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gradient_penalty(sig, output, weight=10):
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


def bce_discriminator_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def bce_generator_loss(fake):
    return -log(torch.sigmoid(fake)).mean()


def hinge_discriminator_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_generator_loss(fake):
    return -fake.mean()


class EncoderAttn(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        attn_depth=(2, 2, 2, 4),
        attn_heads=8,
        attn_dim_head=64,
    ):
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
                            window_size=1024,
                        ),
                    ]
                )
            )

        # Middle
        dim_mid = dims_h[-1]
        self.mid_attn = LocalTransformerBlock(
            dim=dim_mid,
            depth=attn_depth[0],
            heads=attn_heads,
            dim_head=attn_dim_head,
            window_size=1024,
        )

    def forward(self, x):
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
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        attn_depth=(2, 2, 2, 4),
        attn_heads=8,
        attn_dim_head=64,
        use_tanh=False,
    ):
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
            window_size=1024,
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
                            window_size=1024,
                        ),
                    ]
                )
            )

        # End
        self.end_conv = nn.Conv1d(dim_h, dim_in, 1)

    def forward(self, x):
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
        self,
        dim_in,
        dim_h,
        n_emb,
        dim_emb,
        dim_h_mult=(1, 2, 4, 8),
        attn_depth=(2, 2, 2, 4),
        attn_heads=8,
        attn_dim_head=64,
        num_codebooks=8,
        vq_decay=0.9,
        rvq_quantize_dropout=True,
        use_discriminator=False,
        discriminator_layers=4,
        use_tanh=False,
        use_hinge_loss=False,
    ):
        super().__init__()

        self.encoder = EncoderAttn(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )
        self.encoder_norm = nn.LayerNorm(dim_h * dim_h_mult[-1])

        self.decoder = DecoderAttn(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            use_tanh=use_tanh,
        )
        self.vq = ResidualVQ(
            dim=dim_h * dim_h_mult[-1],
            num_quantizers=num_codebooks,
            codebook_dim=dim_emb,
            codebook_size=n_emb,
            decay=vq_decay,
            quantize_dropout=num_codebooks > 1 and rvq_quantize_dropout,
            commitment_weight=0.0,
            kmeans_init=True,
            kmeans_iters=10,
        )

        # Discriminator
        if use_discriminator:
            layer_mults = list(map(lambda x: 2**x, range(discriminator_layers)))
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

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, x):
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return self.vq(x)

    def decode(self, x):
        return self.decoder(x)

    def decode_from_ids(self, ids):
        codes = self.codebook[ids]
        fmap = self.vq.project_out(codes)
        # fmap = rearrange(fmap, "b l c -> b c l")
        return self.decode(fmap)

    def forward(
        self,
        sig,
        return_loss=False,
        return_disc_loss=False,
        return_recons=True,
        add_gradient_penalty=False,
    ):
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
