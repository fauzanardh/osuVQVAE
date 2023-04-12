from torch import nn


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.up = nn.Sequential(nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1), nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.up(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.down = nn.Sequential(nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1), nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.down(x)
        return x
