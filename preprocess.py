from datasets import load_dataset
import nltk
import torch

from collections import Counter
import re
import string


def get_freqs(dataset, size_factor):
    cnt = Counter(
        word for text in dataset["preprocessed_text"] for word in text.split()
    )
    return cnt.most_common(round(len(cnt) * size_factor))


def tokenize_dataset(
    dataset, top_words_map, max_length
) -> (list[list[int]], list[int]):
    xs = []
    ys = []

    for text, label in zip(dataset["preprocessed_text"], dataset["label"]):
        xs.append(
            torch.tensor(
                [top_words_map[word] for word in text.split() if word in top_words_map][
                    :max_length
                ],
                dtype=torch.int,
            )
        )
        ys.append(torch.tensor(label, dtype=torch.float))

    return xs, ys


def left_pad(seqs):
    max_len = max(len(seq) for seq in seqs)
    padded = torch.zeros(
        (len(seqs), max_len), device=seqs[0].device, dtype=seqs[0].dtype
    )
    for i, seq in enumerate(seqs):
        if len(seq) < max_len:
            padded[i, max_len - len(seq) :] = seq
        else:
            padded[i] = seq[:max_len]

    return padded


def review_preprocessing(text, stopwords):
    text = text.lower()
    text = re.sub("<.*?>", "", text)
    text = "".join([c for c in text if c not in string.punctuation])
    text = [word for word in text.split() if word not in stopwords]
    text = " ".join(text)
    return text


def get_loaders(
    validate_dataset_size, batch_size, seq_max_length, tokenizer_size_factor
):
    nltk.download("stopwords", quiet=True)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    imdb_dataset = load_dataset("stanfordnlp/imdb")

    train_dataset = imdb_dataset["train"]
    train_dataset_df = train_dataset.to_pandas()

    validate_dataset, test_dataset = imdb_dataset["test"].train_test_split(0.4).values()
    validate_dataset_df = validate_dataset.to_pandas()
    test_dataset_df = test_dataset.to_pandas()

    for df in [train_dataset_df, validate_dataset_df, test_dataset_df]:
        df["preprocessed_text"] = df["text"].apply(
            review_preprocessing, stopwords=stopwords
        )

    freqs = get_freqs(train_dataset_df, tokenizer_size_factor)
    vocab_size = len(freqs) + 1
    top_words_map = {word: index + 1 for index, (word, _) in enumerate(freqs)}
    x_train, y_train = tokenize_dataset(train_dataset_df, top_words_map, seq_max_length)
    x_val, y_val = tokenize_dataset(validate_dataset_df, top_words_map, seq_max_length)

    x_train_pad = left_pad(x_train)
    y_train_tensor = torch.stack(y_train)
    x_val_pad = left_pad(x_val[:validate_dataset_size])
    y_val_tensor = torch.stack(y_val[:validate_dataset_size])

    train_tensor_data = torch.utils.data.TensorDataset(x_train_pad, y_train_tensor)
    val_tensor_data = torch.utils.data.TensorDataset(x_val_pad, y_val_tensor)

    train_data_loader = torch.utils.data.DataLoader(
        train_tensor_data, batch_size=batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_tensor_data, batch_size=batch_size, shuffle=True
    )

    return vocab_size, train_data_loader, val_data_loader
