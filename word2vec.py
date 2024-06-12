import torch
import torch.nn as nn
import numpy as np

import tqdm
import itertools


def get_skips_and_contexts(tokens_list: list[str], context_size: int):
    assert context_size > 0 and context_size % 2 == 0
    context_size_half = context_size // 2
    contexts, skips = [], []
    for tokens in tqdm.tqdm(tokens_list, desc="Building word2vec dataset"):
        for i, token in enumerate(tokens):
            start = i - context_size_half
            end = i + context_size_half
            if start >= 0 and end < len(tokens):
                context = (
                    tokens[start : start + context_size_half]
                    + tokens[i + 1 : i + 1 + context_size_half]
                )

                contexts.append(context)
                skips.append(token)

    return skips, contexts


def build_cbow_dataset(
    tokens_list: list[str],
    context_size: int,
    batch_size: int,
    device: str = "cpu",
) -> torch.utils.data.DataLoader:
    ys, xs = get_skips_and_contexts(tokens_list, context_size)
    tensor_dataset = torch.utils.data.TensorDataset(
        torch.tensor(xs, dtype=torch.int, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )
    data_loader = torch.utils.data.DataLoader(
        tensor_dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader


def build_skip_gram_dataset(
    tokens_list: list[str],
    context_size: int,
    batch_size: int,
    device: str = "cpu",
) -> torch.utils.data.DataLoader:
    skips, contexts = get_skips_and_contexts(tokens_list, context_size)
    xs = [[skip] for skip in skips for _ in range(context_size)]
    ys = list(itertools.chain(*contexts))
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


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        y = self.emb(x)
        y = self.linear(y)
        return y


class Word2VecTrainer:
    def __init__(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
        model_name: str,
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
        self.valid_model_names = ["CBOW", "skip-gram"]
        self.model_name = model_name
        if model_name not in self.valid_model_names:
            raise ValueError(
                f"model_name must be one of {self.valid_model_names}, but got model_name='{model_name}'"
            )
        if model_name == "CBOW":
            self.model = CBOWModel(vocab_size, embedding_dim, context_size).to(device)
        elif model_name == "skip-gram":
            self.model = SkipGramModel(vocab_size, embedding_dim).to(device)
        self.model_name = model_name
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self.model.train()
            loss_epoch = []

            for inputs, labels in tqdm.tqdm(
                self.train_data_loader, desc=f"Training {self.model_name} epoch #{epoch}"
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
