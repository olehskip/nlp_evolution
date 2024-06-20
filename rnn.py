import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        embedding_model: nn.Module,
        embedding_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        linear_sizes: list[int],
        attention_heads: int = None,
    ):
        super(RNNModel, self).__init__()

        self.emb = embedding_model
        self.rnn = nn.GRU(
            embedding_dim,
            rnn_hidden_size,
            rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(
                    [rnn_hidden_size] + linear_sizes, (linear_sizes + [1])
                )
            ]
        )
        self.attention = (
            nn.MultiheadAttention(embedding_dim, attention_heads, batch_first=True)
            if attention_heads
            else None
        )

    def forward(self, x):
        y = self.emb(x)
        y = F.tanh(y)
        if self.attention:
            y, _ = self.attention(y, y, y, need_weights=False)
        y, _ = self.rnn(y)
        y = y[:, -1, :]
        for linear in self.linears[:-1]:
            y = F.gelu(linear(y))
        y = self.linears[-1](y)
        y = F.sigmoid(y)
        return y
