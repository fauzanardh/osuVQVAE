from torch import nn


class Block(nn.Sequential):
    def __init__(
        self,
        dim,
        dim_out,
        norm=True,
    ):
        super().__init__(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(dim, dim_out, 7, padding=3),
        )


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm=True,
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, norm)
        self.block2 = Block(dim_out, dim_out, norm)
        self.res_conv = nn.Conv1d(dim, dim_out, 7, padding=3) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)
