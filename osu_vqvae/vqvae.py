import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from vector_quantize_pytorch import VectorQuantize

from osu_vqvae.modules.residual import ResnetBlock, GLUResnetBlock
from osu_vqvae.modules.scaler import Downsample, Upsample
from osu_vqvae.modules.discriminator import Discriminator
from osu_vqvae.modules.transformer import TransformerBlock


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
    return -(log(torch.sigmoid(real)) + log(1 - torch.sigmoid(fake))).mean()


def bce_generator_loss(fake):
    return -log(torch.sigmoid(fake)).mean()


def hinge_discriminator_loss(fake, real):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def hinge_generator_loss(fake):
    return fake.mean()


class EncoderAttn(nn.Module):
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
                        Downsample(layer_dim_in, layer_dim_out),
                        nn.ModuleList(
                            [ResnetBlock(layer_dim_out) for _ in range(res_block_depth)]
                        ),
                        TransformerBlock(
                            layer_dim_out, attn_depth, attn_heads, attn_dim_head
                        ),
                    ]
                )
            )

        # Middle
        dim_mid = dims_h[-1]
        self.mid_block1 = ResnetBlock(dim_mid)
        self.mid_attn = TransformerBlock(dim_mid, 1, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(dim_mid)

    def forward(self, x):
        x = self.init_conv(x)

        # Down
        for downsample, resnet_blocks, attn_block in self.downs:
            x = downsample(x)
            for resnet_block in resnet_blocks:
                x = resnet_block(x)
            x = attn_block(x)

        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        **kwargs,
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
                        Downsample(layer_dim_in, layer_dim_out),
                        nn.ModuleList(
                            [ResnetBlock(layer_dim_out) for _ in range(res_block_depth)]
                        ),
                    ]
                )
            )

    def forward(self, x):
        x = self.init_conv(x)

        # Down
        for downsample, resnet_blocks in self.downs:
            x = downsample(x)
            for resnet_block in resnet_blocks:
                x = resnet_block(x)

        return x


class DecoderAttn(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        attn_depth=1,
        attn_heads=8,
        attn_dim_head=64,
        use_tanh=False,
    ):
        super().__init__()
        self.use_tanh = use_tanh

        dim_h_mult = tuple(reversed(dim_h_mult))
        dims_h = [dim_h * mult for mult in dim_h_mult]
        in_out = list(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Middle
        dim_mid = dims_h[0]
        self.mid_block1 = ResnetBlock(dim_mid)
        self.mid_attn = TransformerBlock(dim_mid, 1, attn_heads, attn_dim_head)
        self.mid_block2 = ResnetBlock(dim_mid)

        # Up
        self.ups = nn.ModuleList([])
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            self.ups.append(
                nn.ModuleList(
                    [
                        Upsample(layer_dim_in, layer_dim_out),
                        nn.ModuleList(
                            [
                                GLUResnetBlock(layer_dim_out)
                                for _ in range(res_block_depth)
                            ]
                        ),
                        TransformerBlock(
                            layer_dim_out, attn_depth, attn_heads, attn_dim_head
                        ),
                    ]
                )
            )

        # End
        self.end_conv = nn.Conv1d(dim_h, dim_in, 1)

    def forward(self, x):
        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Up
        for upsample, resnet_blocks, attn_block in self.ups:
            x = upsample(x)
            for resnet_block in resnet_blocks:
                x = resnet_block(x)
            x = attn_block(x)

        return torch.tanh(self.end_conv(x)) if self.use_tanh else self.end_conv(x)


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        use_tanh=False,
        **kwargs,
    ):
        super().__init__()
        self.use_tanh = use_tanh

        dim_h_mult = tuple(reversed(dim_h_mult))
        dims_h = [dim_h * mult for mult in dim_h_mult]
        in_out = list(zip(dims_h[:-1], dims_h[1:]))
        num_layers = len(in_out)

        # Up
        self.ups = nn.ModuleList([])
        for ind in range(num_layers):
            layer_dim_in, layer_dim_out = in_out[ind]
            self.ups.append(
                nn.ModuleList(
                    [
                        Upsample(layer_dim_in, layer_dim_out),
                        nn.ModuleList(
                            [
                                GLUResnetBlock(layer_dim_out)
                                for _ in range(res_block_depth)
                            ]
                        ),
                    ]
                )
            )

        # End
        self.end_conv = nn.Conv1d(dim_h, dim_in, 1)

    def forward(self, x):
        # Up
        for upsample, resnet_blocks in self.ups:
            x = upsample(x)
            for resnet_block in resnet_blocks:
                x = resnet_block(x)

        return torch.tanh(self.end_conv(x)) if self.use_tanh else self.end_conv(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        n_emb,
        dim_emb,
        dim_h_mult=(1, 2, 4, 8),
        res_block_depth=3,
        enc_use_attn=False,
        dec_use_attn=False,
        attn_depth=1,
        attn_heads=8,
        attn_dim_head=64,
        commitment_weight=0.25,
        use_discriminator=False,
        discriminator_layers=4,
        use_tanh=False,
        use_hinge_loss=False,
    ):
        super().__init__()

        encoder_class = EncoderAttn if enc_use_attn else Encoder
        self.encoder = encoder_class(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            res_block_depth=res_block_depth,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            use_tanh=use_tanh,
        )
        decoder_class = DecoderAttn if dec_use_attn else Decoder
        self.decoder = decoder_class(
            dim_in,
            dim_h,
            dim_h_mult=dim_h_mult,
            res_block_depth=res_block_depth,
            attn_depth=attn_depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            use_tanh=use_tanh,
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

        loss = recon_loss + gen_loss + commit_loss
        if return_recons:
            return loss, fmap
        else:
            return loss
