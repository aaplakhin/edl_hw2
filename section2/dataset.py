from typing import Optional, List, Tuple

import torch

from torch import nn
from torch.utils.data.dataset import Dataset
from torchtext.data.utils import get_tokenizer

MAX_LENGTH = 640


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length

        tokenizer = get_tokenizer("basic_english")

        data = []

        with open(data_path, "r") as file:
            for line in file.read().splitlines():
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = tokens[:self.max_length]
                    tokens += ['<unk>'] * (self.max_length - len(tokens))
                    data.append(tokens)

        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length

        tokenizer = get_tokenizer("basic_english")

        data = []

        with open(data_path, "r") as file:
            for line in file.read().splitlines():
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = tokens[:self.max_length]
                    data.append(tokens)

        self.data = data

    def __getitem__(self, idx: int):
        return self.data[idx]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


class MyCollator(object):
    def __init__(self, text_pipeline):
        self.text_pipeline = text_pipeline

    def __call__(self, batch):
        text_list = []
        for text in batch:
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)

        text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
        return text_list
