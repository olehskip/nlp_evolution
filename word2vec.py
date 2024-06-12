import torch
import torch.nn as nn
import numpy as np

import tqdm


def build_cbow_dataset(
    tokens_list: list[str],
    context_size: int,
    batch_size: int,
    device: str = "cpu",
) -> torch.utils.data.DataLoader:
    assert context_size > 0 and context_size % 2 == 0
    context_size_half = context_size // 2
    xs, ys = [], []
    for tokens in tqdm.tqdm(tokens_list, desc="Building CBOW dataset"):
        for i, token in enumerate(tokens):
            start = i - context_size_half
            end = i + context_size_half
            if start >= 0 and end < len(tokens):
                x = (
                    tokens[start : start + context_size_half]
                    + tokens[i + 1 : i + 1 + context_size_half]
                )

                xs.append(x)
                ys.append(token)

    tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor(xs, dtype=torch.int, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )
    data_loader = torch.utils.data.DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader


class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):
        super(CBOWModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        y = self.emb(x)
        y = y.sum(dim=1)
        y = self.linear(y)
        return y


class CBOWTrainer:
    def __init__(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
        device: str = "cpu",
    ):
        self.train_data_loader, self.val_data_loader = (
            train_data_loader,
            val_data_loader,
        )
        self.device = device
        self.model = CBOWModel(vocab_size, embedding_dim, context_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self.model.train()
            loss_epoch = []

            for inputs, labels in tqdm.tqdm(
                self.train_data_loader, desc=f"Training CBOW epoch #{epoch}"
            ):
                xs = inputs.to(self.device)
                ys = labels.to(self.device)
                out = self.model.forward(xs)
                loss = self.criterion(out.squeeze(1), ys)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_epoch.append(loss.item())

            loss_epoch_mean = np.mean(loss_epoch)
            print(f"epoch = {epoch:02d}; loss = {loss_epoch_mean:.10f}")

        print("Finished")
