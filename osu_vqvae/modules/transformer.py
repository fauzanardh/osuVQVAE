import torch
from local_attention import DynamicPositionBias, LocalMHA
from torch import nn
from torch.nn import functional as F

from osu_vqvae.modules.attention import Attention


def shift_token(t: torch.Tensor) -> torch.Tensor:
    t, t_shift = t.chunk(2, dim=-1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value=0.0)
    return torch.cat((t, t_shift), dim=-1)


class GEGLU(nn.Module):
    def forward(self: "GEGLU", x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


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
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.dynamic_pos_bias = None

        if dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(
                dim=dim // 2,
                heads=heads,
            )

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
        if self.dynamic_pos_bias is not None:
            attn_bias = self.dynamic_pos_bias(self.ws, self.ws * 2)

        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x
