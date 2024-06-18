import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import tqdm


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


class RNNTrainer:
    def __init__(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        embedding_model: nn.Module,
        embedding_train: bool,
        embedding_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        linear_sizes: list[int],
        device: str,
        lr: float = 0.005,
        attention_heads: int = None,
    ):
        self.train_data_loader, self.val_data_loader = (
            train_data_loader,
            val_data_loader,
        )

        self.device = device
        self.model = torch.compile(
            RNNModel(
                embedding_model,
                embedding_dim,
                rnn_hidden_size,
                rnn_num_layers,
                rnn_dropout,
                linear_sizes,
                attention_heads,
            ).to(device)
        )
        if not embedding_train:
            for param in embedding_model.parameters():
                param.requires_grad = False
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-3
        )
        self.criterion = nn.BCELoss()

    @staticmethod
    def acc(pred, label):
        return torch.sum(torch.round(pred).flatten() == label.flatten()).item()

    @torch.no_grad()
    def validate(self, use_tqdm: bool):
        self.model.eval()
        true_predictions = 0
        for inputs, labels in tqdm.tqdm(
            self.val_data_loader, desc="Validate RNN", disable=not use_tqdm
        ):
            xs = inputs.to(self.device)
            ys = labels.to(self.device)
            out = self.model.forward(xs)
            true_predictions += self.acc(out, ys)

        return true_predictions / len(self.val_data_loader.dataset) * 100

    def train_one_epoch(self, use_tqdm: bool):
        self.model.train()
        loss_epoch = []

        true_predictions = 0
        for inputs, labels in tqdm.tqdm(
            self.train_data_loader, desc="Training RNN", disable=not use_tqdm
        ):
            xs = inputs.to(self.device)
            ys = labels.to(self.device)
            out = self.model.forward(xs)

            loss = self.criterion(out.squeeze(1), ys)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            true_predictions += self.acc(out, ys)

            loss_epoch.append(loss.item())
        loss_epoch_mean = np.mean(loss_epoch)
        return loss_epoch_mean, true_predictions / len(
            self.train_data_loader.dataset
        ) * 100

    def train(
        self, max_epochs: int, target_validation_score: float, use_tqdm: bool = True
    ):
        lossi = []
        for epoch in range(max_epochs):
            loss_epoch_mean, train_score = self.train_one_epoch(use_tqdm)
            lossi.append(loss_epoch_mean)
            validate_score = self.validate(use_tqdm)
            print(
                f"epoch = {epoch:02d}; loss = {loss_epoch_mean:.8f}; train = {train_score:.4f}; validate = {validate_score:.4f}"
            )

            if loss_epoch_mean >= target_validation_score:
                print("Target validate score {target_validation_score.4f} reached")
        plt.plot(lossi)
        print("Finished")
