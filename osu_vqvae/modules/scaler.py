from torch import nn


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
