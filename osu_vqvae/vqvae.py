import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from vector_quantize_pytorch import VectorQuantize

from osu_vqvae.modules.residual import ResnetBlock
from osu_vqvae.modules.scaler import Downsample, Upsample
from osu_vqvae.modules.discriminator import Discriminator
from osu_vqvae.modules.transformer import TransformerBlock


def log(t, eps=1e-4):
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


class Encoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        attn_depth=1,
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
                        nn.ModuleList(
                            [
                                ResnetBlock(layer_dim_in, layer_dim_in)
                                for _ in range(res_block_depth)
                            ]
                        ),
                        TransformerBlock(
                            layer_dim_in, attn_depth, attn_heads, attn_dim_head
                        ),
                        Downsample(layer_dim_in, layer_dim_out),
                    ]
                )
            )

        # Middle
        dim_mid = dims_h[-1]
        self.mid_block1 = ResnetBlock(dim_mid, dim_mid)
        self.mid_attn = TransformerBlock(dim_mid, 1, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(dim_mid, dim_mid)

    def forward(self, x):
        x = self.init_conv(x)

        # Down
        for resnet_blocks, attn_block, downsample in self.downs:
            for resnet_block in resnet_blocks:
                x = resnet_block(x)
            x = attn_block(x)
            x = downsample(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        attn_depth=1,
        attn_heads=8,
        attn_dim_head=64,
    ):
        super().__init__()

        dim_h_mult = tuple(reversed(dim_h_mult))
        dims_h = [dim_h * mult for mult in dim_h_mult]
        in_out = list(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Middle
        dim_mid = dims_h[0]
        self.mid_block1 = ResnetBlock(dim_mid, dim_mid)
        self.mid_attn = TransformerBlock(dim_mid, 1, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(dim_mid, dim_mid)

        # Up
        self.ups = nn.ModuleList([])
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            self.ups.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ResnetBlock(layer_dim_in, layer_dim_in)
                                for _ in range(res_block_depth)
                            ]
                        ),
                        TransformerBlock(
                            layer_dim_in, attn_depth, attn_heads, attn_dim_head
                        ),
                        Upsample(layer_dim_in, layer_dim_out),
                    ]
                )
            )

        # End
        self.end_conv = nn.Conv1d(dim_h, dim_in, 7, padding=3)

    def forward(self, x):
        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Up
        for resnet_blocks, attn_block, upsample in self.ups:
            for resnet_block in resnet_blocks:
                x = resnet_block(x)
            x = attn_block(x)
            x = upsample(x)

        return self.end_conv(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        n_emb,
        dim_emb,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        attn_depth=1,
        attn_heads=8,
        attn_dim_head=64,
        commitment_weight=0.25,
        discriminator_layers=4,
    ):
        super().__init__()
        self.encoder = Encoder(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            res_block_depth=res_block_depth,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )
        self.decoder = Decoder(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            res_block_depth=res_block_depth,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )
        self.vq = VectorQuantize(
            dim=dim_h * dim_h_mult[-1],
            codebook_dim=dim_emb,
            codebook_size=n_emb,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            use_cosine_sim=True,
            channel_last=False,
        )

        # Discriminator
        layer_mults = list(map(lambda x: 2**x, range(discriminator_layers)))
        layer_dims = [dim_h * mult for mult in layer_mults]
        self.discriminator = Discriminator(
            dim_in,
            layer_dims,
        )

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, x):
        x = self.encoder(x)
        return self.vq(x)

    def decode(self, x):
        return self.decoder(x)

    def decode_from_ids(self, ids):
        codes = self.codebook[ids]
        fmap = self.vq.project_out(codes)
        fmap = rearrange(fmap, "b l c -> b c l")
        return self.decode(fmap)

    def forward(
        self,
        sig,
        return_loss=False,
        return_disc_loss=False,
        return_recons=True,
        add_gradient_penalty=False,
    ):
        fmap, _, commit_loss = self.encode(sig)
        fmap = self.decode(fmap)

        if not (return_loss or return_disc_loss):
            return fmap

        if return_disc_loss:
            fmap.detach_()
            sig.requires_grad_()

            fmap_disc_logits, sig_disc_logits = map(self.discriminator, (fmap, sig))
            loss = bce_discriminator_loss(fmap_disc_logits, sig_disc_logits)

            if add_gradient_penalty:
                loss += gradient_penalty(sig, sig_disc_logits)

            if return_recons:
                return loss, fmap
            else:
                return loss

        recon_loss = F.mse_loss(fmap, sig)
        gen_loss = bce_generator_loss(self.discriminator(fmap))

        loss = recon_loss + gen_loss + commit_loss
        if return_recons:
            return loss, fmap
        else:
            return loss
