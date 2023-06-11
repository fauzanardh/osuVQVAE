import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpsampleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, factor=2):
        super().__init__()

        self.factor = factor

        linear = nn.Linear(dim_in, dim_out * factor)
        self.init_(linear)
        self.up = nn.Sequential(
            linear, nn.SiLU(), Rearrange("b l (p c) -> b (l p) c", p=factor)
        )

    def init_(self, linear):
        o, i = linear.weight.shape

        linear_weight = torch.empty(o // self.factor, i)
        nn.init.kaiming_uniform_(linear_weight)

        linear_weight = repeat(linear_weight, "o ... -> (o r) ...", r=self.factor)

        linear.weight.data.copy_(linear_weight)
        nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        x = self.up(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1), nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class DownsampleLinear(nn.Module):
    def __init__(self, dim_in, dim_out, factor=2):
        super().__init__()
        self.down = nn.Sequential(
            Rearrange("b (l p) c -> b l (p c)", p=factor),
            nn.Linear(dim_in * factor, dim_out),
        )

    def forward(self, x):
        x = self.down(x)
        return x
