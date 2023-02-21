from enum import Enum

import torch

from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

from dataset import BrainDataset, BigBrainDataset, UltraDuperBigBrainDataset, MyCollator
from transformer import PositionalEncoding


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


class GPT_2(nn.Module):
    def __init__(self, vocab_size=226783):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 1024)
        self.positional = PositionalEncoding(1024)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=1024, nhead=8), 1)
        self.linear = nn.Linear(1024, vocab_size)

    def forward(self, src, trg):
        embs_src = self.embedding(src)
        embs_src = self.positional(embs_src)
        embs_trg = self.embedding(trg)
        embs_trg = self.positional(embs_trg)

        att = self.decoder(embs_src, embs_trg)
        return self.linear(att)


def yield_tokens(data_iter):
    for tokens in data_iter:
        yield tokens


def run_epoch(data_mode: DataMode):
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

    collate_fn = MyCollator(text_pipeline)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT_2(len(vocab)).to(device)
    model.eval()

    for src, trg in dataloader:
        src.to(device)
        trg.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model(src, trg)
        end.record()

        torch.cuda.synchronize()

    return {"type": "Brain", "time": start.elapsed_time(end)}
