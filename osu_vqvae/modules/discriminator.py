from typing import Tuple

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self: "Discriminator", dim_in: int, h_dims: Tuple[int]) -> None:
        super().__init__()
        dim_pairs = list(zip(h_dims[:-1], h_dims[1:]))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim_in, h_dims[0], 15, padding=7),
                    nn.LeakyReLU(0.1),
                ),
            ],
        )

        for _in_dim, _out_dim in dim_pairs:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(_in_dim, _out_dim, 41, stride=4, padding=20),
                    nn.LeakyReLU(0.1),
                ),
            )

        dim = h_dims[-1]
        self.to_logits = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim, 1, 3, padding=1),
        )

    def forward(
        self: "Discriminator", x: torch.Tensor, return_intermediates: bool = False
    ) -> torch.Tensor:
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        out = self.to_logits(x)
        if not return_intermediates:
            return out
        else:
            return out, intermediates


class MultiScaleDiscriminator(nn.Module):
    def __init__(
        self: "MultiScaleDiscriminator",
        dim_in: int,
        dim_h: int = 16,
        groups: Tuple[int] = (4, 16, 64, 256),
        max_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.init_conv = nn.Conv1d(dim_in, dim_h, 15, padding=7)
        self.conv_layers = nn.ModuleList([])

        cur_dim = dim_h
        for group in groups:
            out_dim = min(cur_dim * 4, max_dim)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(cur_dim, out_dim, 41, stride=4, padding=20, groups=group),
                    nn.LeakyReLU(0.1),
                ),
            )
            cur_dim = out_dim

        self.to_logits = nn.Sequential(
            nn.Conv1d(cur_dim, cur_dim, 5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(cur_dim, 1, 3, padding=1),
        )

    def forward(
        self: "MultiScaleDiscriminator",
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        x = self.init_conv(x)

        intermediates = []
        for layer in self.conv_layers:
            x = layer(x)
            intermediates.append(x)

        out = self.to_logits(x)

        if not return_intermediates:
            return out

        return out, intermediates
