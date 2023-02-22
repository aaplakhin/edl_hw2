from typing import Optional, List, Tuple

import torch
import random

from torch import nn
from torch.utils.data.dataset import Dataset
from torchtext.data.utils import get_tokenizer


MAX_LENGTH = 640


class BrainDataset(Dataset):
    def __init__(self, data_path_1: str, data_path_2: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length

        tokenizer = get_tokenizer("basic_english")

        data = []

        with open(data_path_1, "r") as file:
            for line in file.read().splitlines():
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    tokens += ['<pad>'] * (self.max_length - len(tokens) + 1)
                    data.append(tokens)

        with open(data_path_2, "r") as file:
            for line in file.read().splitlines():
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    tokens += ['<pad>'] * (self.max_length - len(tokens) + 1)
                    data.append(tokens)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class BigBrainDataset(Dataset):
    def __init__(self, data_path_1: str, data_path_2: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length

        tokenizer = get_tokenizer("basic_english")

        data = []

        with open(data_path_1, "r") as file:
            for i, line in enumerate(file.read().splitlines()):
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    data.append(tokens)

        with open(data_path_2, "r") as file:
            for i, line in enumerate(file.read().splitlines()):
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    data.append(tokens)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path_1: str, data_path_2: str, max_length: int = MAX_LENGTH, n_bins=1):
        self.max_length = max_length

        tokenizer = get_tokenizer("basic_english")

        data = []
        lengths = []

        with open(data_path_1, "r") as file:
            for i, line in enumerate(file.read().splitlines()):
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    data.append(tokens)
                    lengths.append(len(tokens))

        with open(data_path_2, "r") as file:
            for i, line in enumerate(file.read().splitlines()):
                line = line.strip()
                if line and line[0] != "=":
                    tokens = tokenizer(line)
                    tokens = ['<sos>'] + tokens[:self.max_length - 1] + ['<eos>']
                    data.append(tokens)
                    lengths.append(len(tokens))

        self.data = data
        self.lengths = lengths
        self.n_bins = n_bins

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class UltraDuperBigBrainSampler(torch.utils.data.Sampler):

    def __init__(self, lengths: List[int], max_length: int = MAX_LENGTH, n_bins: int = 1,
                 shuffle: bool = True, batch_size: int = 64, drop_last:bool = True):

        self.n_bins = n_bins
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_length = max_length
        self.lengths = lengths

        self.sorted_length_with_idx = sorted(enumerate(self.lengths), key=lambda x: x[1])

        self.bin_size = len(self.lengths) // n_bins
        self.length_idx_bins = [self.sorted_length_with_idx[i: i + self.bin_size]
                                for i in range(0, len(self.lengths), self.bin_size)]

        self.bins = [[] for i in range(self.n_bins)]
        for bin_num in range(self.n_bins):
            for idx, _ in self.length_idx_bins[bin_num]:
                self.bins[bin_num].append(idx)

        self.__iter__() #if we want to know the number of batches

    def __iter__(self):
        if self.shuffle:
            for bin_num in range(self.n_bins):
                random.shuffle(self.bins[bin_num])

        batches = []

        for bin_num in range(self.n_bins):
            cur_batches = [self.bins[bin_num][i: i + self.batch_size]
                           for i in range(0, len(self.bins[bin_num]), self.batch_size)]

            if len(cur_batches) > 1 and self.drop_last:
                if len(cur_batches[-1]) != self.batch_size:
                    cur_batches = cur_batches[:-1]

            batches += cur_batches

        self.length = len(batches)

        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.length


class MyCollator(object):
    def __init__(self, text_pipeline, pad_num):
        self.text_pipeline = text_pipeline
        self.pad_num = pad_num

    def __call__(self, batch):
        text_list = []
        for text in batch:
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)

        text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=self.pad_num)
        return text_list[:, :-1], text_list[:, 1:]
