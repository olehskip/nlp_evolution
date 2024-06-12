import os
import sklearn
import torch

import preprocess
import word2vec
import rnn


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

    word2vec_model_name = os.getenv("word2vec_model", "skip-gram")

    if word2vec_model_name == "CBOW":
        CBOW_CONTEXT_SIZE = 4
        train_word2vec_data_loader = word2vec.build_cbow_dataset(
            train_dataset["tokens"].values.tolist(),
            CBOW_CONTEXT_SIZE,
            BATCH_SIZE,
        )
        print("Train CBOW dataset is ready")
        # val_word2vec_data_loader = word2vec.build_cbow_dataset(
        #     val_dataset["tokens"].values.tolist(), CBOW_CONTEXT_SIZE, BATCH_SIZE
        # )
        print("Validate CBOW dataset is ready")
    elif word2vec_model_name == "skip-gram":
        CBOW_CONTEXT_SIZE = 4
        train_word2vec_data_loader = word2vec.build_skip_gram_dataset(
            train_dataset["tokens"].values.tolist(),
            CBOW_CONTEXT_SIZE,
            BATCH_SIZE,
        )
        print("Train skip-gram dataset is ready")
        # val_word2vec_data_loader = word2vec.build_skip_gram_dataset(
        #     val_dataset["tokens"].values.tolist(), CBOW_CONTEXT_SIZE, BATCH_SIZE
        # )
        print("Validate skip-gram dataset is ready")
    else:
        print(f"There is no dataset for {word2vec_model_name} word2vec model")

    val_word2vec_data_loader = None  # not used so far

    EMBEDDING_DIM = 64
    word2vec_trainer = word2vec.Word2VecTrainer(
        train_word2vec_data_loader,
        val_word2vec_data_loader,
        model_name=word2vec_model_name,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CBOW_CONTEXT_SIZE,
        device=DEVICE,
    )

    word2vec_trainer.train(10)

    rnn_trainer = rnn.RNNTrainer(
        train_data_loader,
        val_data_loader,
        # embedding_model=nn.Embedding(vocab_size, EMBEDDING_DIM),
        embedding_model=word2vec_trainer.model.emb,
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
