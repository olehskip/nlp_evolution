import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn

import matplotlib.pyplot as plt

import preprocess
import word2vec


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


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128

    print(f"Device = {DEVICE}")

    clean_dataset = preprocess.get_prepared_dataset()
    train_dataset, val_dataset = sklearn.model_selection.train_test_split(
        clean_dataset, test_size=0.2
    )
    tokenizer = preprocess.FreqTokenizer(
        train_dataset["preprocessed_text"].values.tolist(), 0.5
    )
    preprocess.add_tokens(train_dataset, tokenizer)
    preprocess.add_tokens(val_dataset, tokenizer)

    train_data_loader = preprocess.create_data_loader(train_dataset, BATCH_SIZE)
    val_data_loader = preprocess.create_data_loader(val_dataset, BATCH_SIZE)

    print("Dataset is ready")
    vocab_size = tokenizer.vocab_size

    SKIP_GRAM_CONTEXT_SIZE = 4
    train_skip_gram_data_loader = word2vec.build_skip_gram_dataset(
        train_dataset["tokens"].values.tolist(),
        SKIP_GRAM_CONTEXT_SIZE,
        BATCH_SIZE,
    )
    print("Train skip-gram dataset is ready")
    val_skip_gram_data_loader = word2vec.build_skip_gram_dataset(
        val_dataset["tokens"].values.tolist(), SKIP_GRAM_CONTEXT_SIZE, BATCH_SIZE
    )
    print("Validate skip-gram dataset is ready")
    EMBEDDING_DIM = 64
    skip_gram_trainer = word2vec.SkipGramTrainer(
        train_skip_gram_data_loader,
        val_skip_gram_data_loader,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=SKIP_GRAM_CONTEXT_SIZE,
        device=DEVICE,
    )

    skip_gram_trainer.train(16)
    # exit()

    rnn_trainer = RNNTrainer(
        train_data_loader,
        val_data_loader,
        # embedding_model=nn.Embedding(vocab_size, EMBEDDING_DIM),
        embedding_model=skip_gram_trainer.model.emb,
        embedding_train=False,
        embedding_dim=EMBEDDING_DIM,
        rnn_hidden_size=256,
        rnn_num_layers=3,
        rnn_dropout=0.5,
        linear_sizes=[512, 512],
        device=DEVICE,
    )
    print(f"Initial validation_score = {rnn_trainer.validate()}")
    rnn_trainer.train(max_epochs=64, target_validation_score=90)
    print(f"Final validation_score = {rnn_trainer.validate()}")
