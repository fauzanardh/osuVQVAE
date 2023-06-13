from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self: "CausalConv1d",
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def forward(self: "CausalConv1d", x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self: "CausalConvTranspose1d",
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        **kwargs: Dict,
    ) -> None:
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            **kwargs,
        )

    def forward(self: "CausalConvTranspose1d", x: torch.Tensor) -> torch.Tensor:
        n = x.size(-1)

        out = self.conv(x)
        out = out[..., : n * self.upsample_factor]

        return out
