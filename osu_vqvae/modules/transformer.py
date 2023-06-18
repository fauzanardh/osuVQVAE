import math
from typing import List

import torch
from einops import rearrange
from local_attention import DynamicPositionBias, LocalMHA
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


class FeedForward(nn.Module):
    def __init__(self: "FeedForward", dim: int, mult: int = 4) -> None:
        super().__init__()
        hidden_dim = int(dim * mult * 2 / 3)
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2),
            GEGLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self: "FeedForward", x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self: "TransformerBlock",
        dim: int,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim),
                    ],
                ),
            )

    def forward(self: "TransformerBlock", x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
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
        dynamic_pos_bias: bool = False,
        alibi_pos_bias: bool = False,
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.rel_pos = None

        if dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(
                dim=dim // 2,
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
                            use_rotary_pos_emb=not dynamic_pos_bias,
                            use_xpos=True,
                            prenorm=True,
                            causal=True,
                        ),
                        FeedForward(dim),
                    ],
                ),
            )

    def forward(self: "LocalTransformerBlock", x: torch.Tensor) -> torch.Tensor:
        attn_bias = None
        if self.rel_pos is not None:
            attn_bias = self.rel_pos(self.ws, self.ws * 2)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x
