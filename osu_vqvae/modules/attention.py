import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self: "RMSNorm", dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self: "RMSNorm", x: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(x, dim=-1, eps=1e-3)
        return normed * self.scale * self.gamma


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self: "Attention", x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))  # type: ignore
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)
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
        self: "CrossAttention",
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        norm_context: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout

        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.norm_context = (
            nn.LayerNorm(context_dim, eps=1e-3) if norm_context else nn.Identity()
        )
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim, eps=1e-3),
        )

    def forward(
        self: "CrossAttention",
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))  # type: ignore
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)
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
