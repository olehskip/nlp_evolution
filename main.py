import os
import sklearn
import torch
from torch import nn

import preprocess
import word2vec
import rnn
import cnn
from trainer import Trainer


def get_embedding_model(
    embedding_dim: int, vocab_size: int, device: str
) -> (nn.Module, bool):
    embedding_model_name = os.getenv("embedding_model", None)

    if embedding_model_name is None:
        return nn.Embedding(vocab_size, embedding_dim), True
    elif embedding_model_name == "CBOW":
        CONTEXT_SIZE = 4
        train_word2vec_data_loader = word2vec.build_cbow_dataset(
            train_dataset["tokens"].values.tolist(),
            CONTEXT_SIZE,
            BATCH_SIZE,
        )
        print("Train CBOW dataset is ready")
        # val_word2vec_data_loader = word2vec.build_cbow_dataset(
        #     val_dataset["tokens"].values.tolist(), CONTEXT_SIZE, BATCH_SIZE
        # )
        print("Validate CBOW dataset is ready")
    elif embedding_model_name == "skip-gram":
        CONTEXT_SIZE = 4
        train_word2vec_data_loader = word2vec.build_skip_gram_dataset(
            train_dataset["tokens"].values.tolist(),
            CONTEXT_SIZE,
            BATCH_SIZE,
        )
        print("Train skip-gram dataset is ready")
        # val_word2vec_data_loader = word2vec.build_skip_gram_dataset(
        #     val_dataset["tokens"].values.tolist(), CONTEXT_SIZE, BATCH_SIZE
        # )
        print("Validate skip-gram dataset is ready")
    else:
        print(f"There is no dataset for {embedding_model_name} word2vec model")

    val_word2vec_data_loader = None  # not used so far

    # TODO: refactor it
    word2vec_trainer = word2vec.Word2VecTrainer(
        train_word2vec_data_loader,
        val_word2vec_data_loader,
        model_name=embedding_model_name,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_size=CONTEXT_SIZE,
        device=device,
    )

    word2vec_trainer.train(10)
    return word2vec_trainer.model.emb, False


def get_model():
    model_name = os.getenv("model_name", "rnn")
    if model_name == "rnn":
        return rnn.RNNModel(
            embedding_model=embedding_model,
            embedding_dim=EMBEDDING_DIM,
            rnn_hidden_size=64,
            rnn_num_layers=2,
            rnn_dropout=0.8,
            linear_sizes=[],
            attention_heads=None,
        )
    elif model_name == "cnn":
        return cnn.CNNModel(
            embedding_model=embedding_model,
            embedding_dim=EMBEDDING_DIM,
            region_sizes=[2, 3, 4, 5],
            feature_maps=100,
            dropout=0.5,
        )
    else:
        print(f"There is no a model named {model_name}")
        exit()


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 512

    print(f"Device = {DEVICE}")

    clean_dataset = preprocess.get_prepared_dataset()
    train_dataset, val_dataset = sklearn.model_selection.train_test_split(
        clean_dataset, test_size=0.2
    )
    tokenizer = preprocess.FreqTokenizer(
        train_dataset["preprocessed_text"].values.tolist()
    )
    preprocess.add_tokens(train_dataset, tokenizer)
    preprocess.add_tokens(val_dataset, tokenizer)

    train_data_loader = preprocess.create_data_loader(train_dataset, BATCH_SIZE, DEVICE)
    val_data_loader = preprocess.create_data_loader(val_dataset, BATCH_SIZE, DEVICE)

    print("Dataset is ready")
    vocab_size = tokenizer.vocab_size

    EMBEDDING_DIM = 16
    embedding_model, embedding_train = get_embedding_model(
        EMBEDDING_DIM, vocab_size, DEVICE
    )

    model = get_model()

    rnn_trainer = Trainer(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        model=model,
        embedding_train=embedding_train,
        device=DEVICE,
    )

    print(
        f"Model has {sum(p.numel() for p in rnn_trainer.model.parameters() if p.requires_grad)} parameters"
    )
    print(f"Initial validation_score = {rnn_trainer.validate(use_tqdm=False)}")
    rnn_trainer.train(max_epochs=64, target_validation_score=90)
    print(f"Final validation_score = {rnn_trainer.validate(use_tqdm=False)}")
