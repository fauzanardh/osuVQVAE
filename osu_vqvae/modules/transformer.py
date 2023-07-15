import math
from typing import List

import torch
from einops import rearrange
from local_attention import LocalMHA
from torch import nn
from torch.nn import functional as F

from osu_vqvae.modules.attention import Attention
from osu_vqvae.modules.utils import pad_at_dim


def shift_token(t: torch.Tensor) -> torch.Tensor:
    t, t_shift = t.chunk(2, dim=-1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value=0.0)
    return torch.cat((t, t_shift), dim=-1)


class GEGLU(nn.Module):
    def forward(self: "GEGLU", x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class AlibiPositionalBias(nn.Module):
    def __init__(
        self: "AlibiPositionalBias",
        heads: int = 4,
        total_heads: int = 8,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @property
    def device(self: "AlibiPositionalBias") -> torch.device:
        return next(self.buffers()).device

    def get_bias(
        self: "AlibiPositionalBias",
        i: int,
        j: int,
        device: torch.device,
    ) -> torch.Tensor:
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"),
        )
        return bias

    def forward(self: "AlibiPositionalBias", i: int, j: int) -> torch.Tensor:
        h, device = self.heads, self.device

        if (
            self.bias is not None
            and self.bias.shape[-1] >= j
            and self.bias.shape[-2] >= i
        ):
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

    @staticmethod
    def _get_slopes(heads: int) -> List[float]:
        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(
        self: "LearnedAlibiPositionalBias",
        heads: int = 4,
        total_heads: int = 8,
    ) -> None:
        super().__init__(heads=heads, total_heads=total_heads)
        log_slopes = torch.log(self.slopes)
        self.learned_log_slopes = nn.Parameter(log_slopes)

    def forward(self: "LearnedAlibiPositionalBias", i: int, j: int) -> torch.Tensor:
        h, device = self.heads, self.device

        def get_slopes(param: torch.Tensor) -> torch.Tensor:
            return pad_at_dim(param.exp(), (0, h - param.shape[0]), dim=-2)

        if (
            self.bias is not None
            and self.bias.shape[-1] >= j
            and self.bias.shape[-2] >= i
        ):
            return self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer("bias", bias, persistent=False)

        slopes = get_slopes(self.learned_log_slopes)
        bias = bias * slopes
        return bias


class RelativePositionBias(nn.Module):
    def __init__(
        self: "RelativePositionBias",
        scale: int,
        causal: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        causal: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1),
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self: "RelativePositionBias") -> torch.device:
        return next(self.parameters()).device

    def forward(self: "RelativePositionBias", i: int, j: int) -> torch.Tensor:
        device = self.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(
        self: "RotaryEmbedding",
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward(
        self: "RotaryEmbedding",
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        t = t / self.interpolation_factor

        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.0

        power = (
            torch.arange(seq_len, device=device) - (seq_len // 2)
        ) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


class FeedForward(nn.Sequential):
    def __init__(
        self: "FeedForward",
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        hidden_dim = int(dim * mult * 2 / 3)
        super().__init__(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2, bias=False),
            GEGLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 64,
        attn_flash: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        rotary_pos_emb: bool = False,
        rotary_emb_dim: int = None,
        rotary_xpos: bool = False,
        rotary_interpolation_factor: float = 4.0,
        relative_pos_bias: bool = False,
        alibi_pos_bias: bool = False,
    ) -> None:
        super().__init__()
        rotary_emb_dim = max(
            rotary_emb_dim if rotary_emb_dim is not None else (dim_head // 2),
            32,
        )
        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim,
                use_xpos=rotary_xpos,
                interpolation_factor=rotary_interpolation_factor,
            )
        else:
            self.rotary_pos_emb = None

        if relative_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head**-0.5,
                causal=True,
                heads=heads,
            )
        elif alibi_pos_bias:
            self.rel_pos = AlibiPositionalBias(
                heads=heads,
            )
        else:
            self.rel_pos = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                            flash=attn_flash,
                            causal=True,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ],
                ),
            )

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        rotary_pos_emb = None
        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb(x.shape[1], device=x.device)
        for attn, ff in self.layers:
            x = attn(x, rel_pos=self.rel_pos, rotary_pos_emb=rotary_pos_emb) + x
            x = ff(x) + x
        return x


class LocalTransformerBlock(nn.Module):
    def __init__(
        self: "LocalTransformerBlock",
        dim: int,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 64,
        window_size: int = 512,
        ff_dropout: float = 0.0,
        relative_pos_bias: bool = False,
        alibi_pos_bias: bool = False,
    ) -> None:
        super().__init__()
        self.ws = window_size
        if relative_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head**-0.5,
                causal=True,
                heads=heads,
            )
        elif alibi_pos_bias:
            self.rel_pos = AlibiPositionalBias(
                heads=heads,
            )
        else:
            self.rel_pos = None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LocalMHA(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            qk_rmsnorm=True,
                            window_size=window_size,
                            use_rotary_pos_emb=self.rel_pos is None,
                            use_xpos=True,
                            prenorm=True,
                            causal=True,
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ],
                ),
            )

    def forward(self: "LocalTransformerBlock", x: torch.Tensor) -> torch.Tensor:
        attn_bias = None
        if self.rel_pos is not None:
            attn_bias = self.rel_pos(self.ws, self.ws * 2)

        for attn, ff in self.layers:
            x = attn(shift_token(x), attn_bias=attn_bias) + x
            x = ff(shift_token(x)) + x
        return x
