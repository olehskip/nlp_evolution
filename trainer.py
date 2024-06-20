import torch
import torch.nn as nn
import numpy as np

import tqdm
import os


class Trainer:
    def __init__(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        embedding_train: bool,
        device: str,
        lr: float = 0.005,
    ):
        self.train_data_loader, self.val_data_loader = (
            train_data_loader,
            val_data_loader,
        )

        self.device = device
        self.model = (torch.compile(model) if os.getenv("compile", True) else model).to(
            device
        )
        if not embedding_train:
            for param in model.emb.parameters():
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
            self.val_data_loader, desc="Validating", disable=not use_tqdm
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
            self.train_data_loader, desc="Training", disable=not use_tqdm
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
        print("Finished")
