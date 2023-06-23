from functools import partial

import torch
from einops import rearrange
from local_attention import LocalAttention
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


class ConvLocalAttention(nn.Module):
    def __init__(
        self: "ConvLocalAttention",
        dim: int,
        window_size: int,
        heads: int = 8,
        dim_head: int = 64,
        causal: bool = False,
        prenorm: bool = False,
        qk_rmsnorm: bool = False,
        qk_scale: int = 8,
        use_xpos: bool = False,
        xpos_scale_base: int = None,
        exact_window: bool = True,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(1, dim) if prenorm else nn.Identity()

        inner_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)

        self.qk_rmsnorm = qk_rmsnorm
        if self.qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            scale=(qk_scale if self.qk_rmsnorm else None),
            exact_windowsize=exact_window,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        self.to_out = nn.Conv1d(inner_dim, dim, 1, bias=False)

    def forward(
        self: "ConvLocalAttention",
        x: torch.Tensor,
        mask: torch.Tensor = None,
        attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-2)
        q, k, v = (
            rearrange(t, "b (h d) n -> b h n d", h=self.heads) for t in (q, k, v)
        )

        if self.qk_rmsnorm:
            q, k = map(partial(F.normalize, dim=-1), (q, k))
            q *= self.q_scale
            k *= self.k_scale

        out = self.attn_fn(q, k, v, mask=mask, attn_bias=attn_bias)
        out = rearrange(out, "b h n d -> b (h d) n")
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
