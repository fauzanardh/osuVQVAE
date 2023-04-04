from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_dim, h_dims):
        super().__init__()
        dim_pairs = list(zip(h_dims[:-1], h_dims[1:]))
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_dim, h_dims[0], 7, padding=3),
                    nn.SiLU(),
                ),
            ]
        )

        for _in_dim, _out_dim in dim_pairs:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(_in_dim, _out_dim, 4, stride=2, padding=1),
                    nn.SiLU(),
                )
            )

        dim = h_dims[-1]
        self.to_logits = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, 1, 4),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.to_logits(x)
