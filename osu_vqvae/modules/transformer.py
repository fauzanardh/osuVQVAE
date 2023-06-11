import torch
from torch import nn
from torch.nn import functional as F

from local_attention import LocalMHA
from osu_vqvae.modules.attention import Attention


def shift_token(t):
    t, t_shift = t.chunk(2, dim=-1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value=0.0)
    return torch.cat((t, t_shift), dim=-1)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4):
    hidden_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim * 2),
        GEGLU(),
        nn.Linear(hidden_dim, dim),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(shift_token(x)) + x
            x = ff(shift_token(x)) + x
        return x


class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        heads=8,
        dim_head=64,
        window_size=512,
    ):
        super().__init__()
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
                            use_rotary_pos_emb=True,
                            use_xpos=True,
                            causal=True,
                        ),
                        FeedForward(dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(shift_token(x)) + x
            x = ff(shift_token(x)) + x
        return x
