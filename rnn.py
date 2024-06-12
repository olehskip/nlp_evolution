import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class RNNModel(nn.Module):
    def __init__(
        self,
        embedding_model: nn.Module,
        embedding_dim,
        rnn_hidden_size,
        rnn_num_layers,
        rnn_dropout,
        linear_sizes,
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

    def forward(self, x):
        y = self.emb(x)
        y = F.tanh(y)
        y, _ = self.rnn(y)
        y = y[:, -1, :]
        for linear in self.linears[:-1]:
            y = F.relu(linear(y))
        y = self.linears[-1](y)
        y = F.sigmoid(y)
        return y


class RNNTrainer:
    def __init__(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        embedding_model: int,
        embedding_train: bool,
        embedding_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        linear_sizes: list[int],
        device: str,
    ):
        self.train_data_loader, self.val_data_loader = (
            train_data_loader,
            val_data_loader,
        )

        self.device = device
        self.model = RNNModel(
            embedding_model,
            embedding_dim,
            rnn_hidden_size,
            rnn_num_layers,
            rnn_dropout,
            linear_sizes,
        ).to(device)
        if not embedding_train:
            for param in embedding_model.parameters():
                param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: torch.clamp(grad, -16, 16))

    @torch.no_grad
    def validate(self):
        def acc(pred, label):
            return torch.sum(torch.round(pred).flatten() == label.flatten()).item()

        self.model.eval()
        true_predictions = 0
        for inputs, labels in self.val_data_loader:
            xs = inputs.to(self.device)
            ys = labels.to(self.device)
            out = self.model.forward(xs)
            true_predictions += acc(out, ys)

        return true_predictions / len(self.val_data_loader.dataset) * 100

    def train(self, max_epochs, target_validation_score):
        lossi = []
        for epoch in range(max_epochs):
            self.model.train()
            loss_epoch = []

            for inputs, labels in self.train_data_loader:
                xs = inputs.to(self.device)
                ys = labels.to(self.device)
                out = self.model.forward(xs)

                loss = self.criterion(out.squeeze(1), ys)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())

            loss_epoch_mean = np.mean(loss_epoch)
            lossi.append(loss_epoch_mean)
            validate_score = self.validate()
            print(
                f"epoch = {epoch:02d}; loss = {loss_epoch_mean:.10f}; validate = {validate_score:.4f}"
            )

            if loss_epoch_mean >= target_validation_score:
                print("Target validate score {target_validation_score.4f} reached")
        plt.plot(lossi)
        print("Finished")
