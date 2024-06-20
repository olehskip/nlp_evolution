import torch
import torch.nn as nn
import torch.nn.functional as F


# (Yoon Kim, 2014), (Ye Zhang, et al, 2015)
class CNNModel(nn.Module):
    def __init__(
        self,
        embedding_model: nn.Module,
        embedding_dim: int,
        region_sizes: list[int],
        feature_maps: int,
        dropout: float,
    ):
        super(CNNModel, self).__init__()

        self.emb = embedding_model
        self.filters = nn.ModuleList(
            [
                nn.Conv2d(1, feature_maps, (region_size, embedding_dim))
                for region_size in region_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_maps * len(region_sizes), 1)

    def forward(self, x):
        # x has shape (B, L)
        y = x.unsqueeze(1)
        # y has shape (B, 1, L)
        y = self.emb(y)
        y = F.tanh(y)
        # y has shape (B, 1, L, embedding_dim)

        ys = [F.relu(filter(y)) for filter in self.filters]
        # each y has shape (B, feature_maps, filter output, 1)

        ys = [y.max(dim=2).values for y in ys]
        # each y has shape (B, feature_maps, 1)

        y = torch.cat(ys, dim=-1)
        # y has shape (B, feature_maps, region_sizes)

        y = y.flatten(1)
        # y has shape (B, feature_maps * region_sizes)

        y = self.dropout(y)
        y = self.linear(y)
        y = F.sigmoid(y)
        return y
