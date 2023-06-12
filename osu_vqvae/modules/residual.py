import torch
from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self: "ResnetBlock", dim: int, groups: int=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 7, padding=3),
            nn.GroupNorm(groups, dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim, dim, 7, padding=3),
            nn.GroupNorm(groups, dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self: "ResnetBlock", x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class GLUResnetBlock(nn.Module):
    def __init__(self: "GLUResnetBlock", dim: int, groups: int=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 7, padding=3),
            nn.GLU(dim=1),
            nn.GroupNorm(groups, dim),
            nn.Conv1d(dim, dim * 2, 7, padding=3),
            nn.GLU(dim=1),
            nn.GroupNorm(groups, dim),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self: "GLUResnetBlock", x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x
