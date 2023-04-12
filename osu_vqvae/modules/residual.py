from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, dim, groups=32):
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

    def forward(self, x):
        return self.net(x) + x


class GLUResnetBlock(nn.Module):
    def __init__(self, dim, groups=32):
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

    def forward(self, x):
        return self.net(x) + x
