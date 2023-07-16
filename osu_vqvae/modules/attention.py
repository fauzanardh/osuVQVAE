from collections import namedtuple

import torch
from einops import rearrange, repeat
from packaging import version
from torch import einsum, nn
from torch.nn import functional as F

from osu_vqvae.modules.utils import print_once

# constants
_config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: int, freqs: torch.Tensor, scale: float = 1) -> torch.Tensor:
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)


class Attend(nn.Module):
    def __init__(
        self: "Attend",
        dropout: float = 0.0,
        causal: bool = False,
        flash: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "flash attention requires torch 2.0 or greater"

        # configs
        self.cpu_config = _config(True, True, True)

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda",
            )
            self.cuda_config = _config(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda",
            )
            self.cuda_config = _config(False, True, True)

    def flash_attn(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        k = rearrange(k, "b ... -> b 1 ...").expand_as(q)
        v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        causal = self.causal

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)
            if causal:
                causal_mask = torch.ones(
                    (q_len, k_len),
                    device=q.device,
                    dtype=torch.bool,
                ).triu(k_len - q_len + 1)
                mask = mask & ~causal_mask
                causal = False

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )
        return out

    def forward(
        self: "Attend",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5

        if self.flash:
            assert attn_bias is None, "attention bias not supported for flash attention"
            return self.flash_attn(q, k, v, mask=mask)

        # similarity
        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention bias
        if attn_bias is not None:
            sim = sim + attn_bias

        # key padding mask
        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=sim.device, dtype=torch.bool).triu(
                j - i + 1,
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(
        self: "Attention",
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dim_context: int = None,
        norm_context: bool = False,
        causal: bool = False,
        num_null_kv: int = 0,
        dropout: float = 0.1,
        flash: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = dim_context if dim_context is not None else dim

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = (
            nn.Parameter(torch.randn(2, num_null_kv, dim_head))
            if num_null_kv > 0
            else None
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)

        self.attend = Attend(dropout=dropout, causal=causal, flash=flash)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self: "Attention",
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
        rel_pos: nn.Module = None,
        rotary_pos_emb: torch.Tensor = None,
        # attn_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        if context is not None:
            context = self.context_norm(context)

        kv_input = context if context is not None else x

        # prenorm
        x = self.norm(x)

        # project to q, k, v
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # split for multi-headed attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # rotary positional embedding
        if rotary_pos_emb is not None and context is None:
            freqs, xpos_scale = rotary_pos_emb
            seq_len = freqs.shape[-1]

            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0)
                if xpos_scale is not None
                else (1.0, 1.0)
            )
            (ql, qr), (kl, kr), (vl, vr) = (
                (t[..., :seq_len], t[..., seq_len:]) for t in (q, k, v)
            )

            ql, kl, vl = (
                apply_rotary_pos_emb(arg[0], freqs, arg[1])
                for arg in ((ql, q_xpos_scale), (kl, k_xpos_scale), (vl, k_xpos_scale))
            )
            q, k, v = (torch.cat(t, dim=-1) for t in ((ql, qr), (kl, kr), (vl, vr)))

        # null key / values
        if self.num_null_kv > 0:
            null_k, null_v = repeat(
                self.null_kv,
                "kv n d -> kv b n d",
                b=x.shape[0],
            ).unbind(
                dim=0,
            )
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        i, j = (t.shape[-2] for t in (q, k))

        # handle mask and null key / value
        if mask is not None:
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)

        # prepare relative positional bias, if needed
        attn_bias = None
        if rel_pos is not None:
            attn_bias = rel_pos(i, j)

        # attention
        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask)

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
