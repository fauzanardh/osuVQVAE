from typing import Dict

import torch
from einops import rearrange
from torch import nn

from osu_vqvae.modules.causal_convolution import CausalConv1d


class SqueezeExcite(nn.Module):
    def __init__(
        self: "SqueezeExcite",
        dim: int,
        reduction_factor: int = 4,
        dim_minimum: int = 8,
    ) -> None:
        super().__init__()
        hidden_dim = max(dim // reduction_factor, dim_minimum)
        self.layers = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self: "SqueezeExcite", x: torch.Tensor) -> torch.Tensor:
        seq, device = x.shape[-2], x.device

        cum_sum = x.cumsum(dim=-2)
        denom = torch.arange(1, seq + 1, device=device).float()
        cum_mean = cum_sum / rearrange(denom, "n -> n 1")

        return x * self.layers(cum_mean)


class Residual(nn.Module):
    def __init__(self: "Residual", fn: callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self: "Residual", x: torch.Tensor, **kwargs: Dict) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class ResidualBlock(nn.Module):
    def __init__(
        self: "ResidualBlock",
        dim_in: int,
        dim_out: int,
        dilation: int,
        kernel_size: int = 7,
        squeeze_excite: bool = False,
    ) -> None:
        super().__init__()
        self.layers = Residual(
            nn.Sequential(
                CausalConv1d(dim_in, dim_out, kernel_size, dilation=dilation),
                nn.ELU(),
                nn.BatchNorm1d(dim_out),
                CausalConv1d(dim_out, dim_out, 1),
                nn.ELU(),
                nn.BatchNorm1d(dim_out),
                SqueezeExcite(dim_out) if squeeze_excite else nn.Identity(),
            ),
        )

    def forward(self: "ResidualBlock", x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
