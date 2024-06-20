from datasets import load_dataset
import nltk
import torch
import pandas

from collections import Counter
import re
import string
import abc


class Tokenizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tokenize(self, text: str) -> list[list[int]]:
        pass


class FreqTokenizer(Tokenizer):
    def __init__(self, texts: list[str], freqs_size_factor: float = 1):
        cnt = Counter(word for text in texts for word in text.split())
        freqs = cnt.most_common(round(len(cnt) * freqs_size_factor))
        self.vocab_size = len(freqs) + 1
        self.top_words_map = {word: index + 1 for index, (word, _) in enumerate(freqs)}

    def tokenize(self, text: str) -> list[list[int]]:
        return [
            self.top_words_map[word]
            for word in text.split()
            if word in self.top_words_map
        ]


def create_data_loader(
    dataset: pandas.DataFrame,
    batch_size: int,
    device: str = "cpu",
):
    def seqs_to_tensor(seqs):
        max_len = max(len(seq) for seq in seqs)
        padded = torch.zeros((len(seqs), max_len), dtype=torch.int, device=device)
        for i, seq in enumerate(seqs):
            if len(seq) < max_len:
                padded[i, max_len - len(seq) :] = torch.tensor(seq).to(padded)
            else:
                padded[i] = torch.tensor(seq[:max_len]).to(padded)

        return padded

    tensor_dataset = torch.utils.data.TensorDataset(
        seqs_to_tensor(dataset["tokens"].values.tolist()),
        torch.tensor([y for y in dataset["label"]], dtype=torch.float, device=device),
    )
    data_loader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cpu"),
    )

    return data_loader


def get_prepared_dataset() -> pandas.DataFrame:
    imdb_dataset = load_dataset("stanfordnlp/imdb")

    df = pandas.concat(
        [imdb_dataset["train"].to_pandas(), imdb_dataset["test"].to_pandas()]
    )

    nltk.download("stopwords", quiet=True)
    stopwords = set(nltk.corpus.stopwords.words("english"))

    def review_preprocessing(text, stopwords):
        text = text.lower()
        text = re.sub("<.*?>", "", text)
        text = "".join([c for c in text if c not in string.punctuation])
        text = [word for word in text.split() if word not in stopwords]
        text = " ".join(text)
        return text

    df["preprocessed_text"] = df["text"].apply(
        review_preprocessing, stopwords=stopwords
    )

    return df


def add_tokens(dataset: pandas.DataFrame, tokenizer: Tokenizer):
    dataset["tokens"] = dataset["preprocessed_text"].apply(
        lambda x: tokenizer.tokenize(x)
    )
