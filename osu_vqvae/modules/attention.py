import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))  # type: ignore
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout,
            )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Used for conditioning, after the current model is proven to work
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        heads=8,
        dim_head=64,
        norm_context=False,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim),
        )

    def forward(self, x, context):
        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))  # type: ignore
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout,
            )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
