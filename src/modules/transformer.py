from torch import nn
from einops import rearrange

from modules.attention import Attention


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        heads=8,
        dim_head=64,
        ff_mult=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        x = rearrange(x, "b c l -> b l c")
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return rearrange(x, "b l c -> b c l")
