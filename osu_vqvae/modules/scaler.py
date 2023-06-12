import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn


class UpsampleConv(nn.Module):
    def __init__(self: "UpsampleConv", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self: "UpsampleConv", x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class UpsampleLinear(nn.Module):
    def __init__(
        self: "UpsampleLinear",
        dim_in: int,
        dim_out: int,
        factor: int = 2,
    ) -> None:
        super().__init__()

        self.factor = factor

        linear = nn.Linear(dim_in, dim_out * factor)
        self.init_(linear)
        self.up = nn.Sequential(
            linear,
            nn.SiLU(),
            Rearrange("b l (p c) -> b (l p) c", p=factor),
        )

    def init_(self: "UpsampleLinear", linear: torch.nn.Module) -> None:
        o, i = linear.weight.shape

        linear_weight = torch.empty(o // self.factor, i)
        nn.init.kaiming_uniform_(linear_weight)

        linear_weight = repeat(linear_weight, "o ... -> (o r) ...", r=self.factor)

        linear.weight.data.copy_(linear_weight)
        nn.init.zeros_(linear.bias.data)

    def forward(self: "UpsampleLinear", x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class DownsampleConv(nn.Module):
    def __init__(self: "DownsampleConv", dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )

    def forward(self: "DownsampleConv", x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class DownsampleLinear(nn.Module):
    def __init__(
        self: "DownsampleLinear",
        dim_in: int,
        dim_out: int,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            Rearrange("b (l p) c -> b l (p c)", p=factor),
            nn.Linear(dim_in * factor, dim_out),
        )

    def forward(self: "DownsampleLinear", x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x
