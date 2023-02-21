from enum import Enum
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

from dataset import BrainDataset, BigBrainDataset, UltraDuperBigBrainDataset, MyCollator


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model() -> torch.nn.Module:
    pass


def yield_tokens(data_iter):
    for tokens in data_iter:
        yield tokens


def run_epoch(data_mode: DataMode) -> None:
    if data_mode.name == 'BRAIN':
        dataset = BrainDataset("wikitext-103/wiki.train.tokens")
    elif data_mode.name == 'BIG_BRAIN':
        dataset = BigBrainDataset("wikitext-103/wiki.train.tokens")
    else:
        pass
    print(data_mode.name)

    vocab = build_vocab_from_iterator(yield_tokens(iter(dataset)), specials=["<unk>", "<sos>", ",<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    vocab.set_default_index(vocab["<sos>"])
    vocab.set_default_index(vocab["<eos>"])

    text_pipeline = lambda x: vocab(x)
    print(data_mode.name)
    collate_fn = MyCollator(text_pipeline)

    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

    return dataloader
