import torch.nn as nn


def _block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            _block(input_dim, embed_dim, dropout),
            *[_block(embed_dim, embed_dim, dropout) for _ in range(num_layers - 1)],
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
