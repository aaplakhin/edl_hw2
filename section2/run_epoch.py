from enum import Enum

import torch

from tqdm import tqdm
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

    def forward(self, tgt, src, src_mask, tgt_mask):
        embs_tgt = self.embedding(tgt)
        embs_tgt = self.positional(embs_tgt)

        embs_src = self.embedding(src)
        memory = self.positional(embs_src)

        att = self.decoder(embs_tgt, memory, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return att


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

    vocab = build_vocab_from_iterator(yield_tokens(iter(dataset)), specials=["<unk>", "<sos>", "<eos>", '<pad>'])
    vocab.set_default_index(vocab["<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    vocab.set_default_index(vocab["<sos>"])
    vocab.set_default_index(vocab["<eos>"])

    BATCH_SIZE = 16

    text_pipeline = lambda x: vocab(x)

    collate_fn = MyCollator(text_pipeline, vocab["<pad>"])

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT_2(len(vocab)).to(device)
    model.eval()

    brain_times = []

    for i, (src, tgt) in enumerate(tqdm(dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        src_pad_mask = (src == vocab["<pad>"]).view(src.shape[1], BATCH_SIZE)
        tgt_pad_mask = (src == vocab["<pad>"]).view(tgt.shape[1], BATCH_SIZE)

        start.record()
        model(tgt, src, tgt_pad_mask, src_pad_mask)
        end.record()

        torch.cuda.synchronize()

        brain_times.append(start.elapsed_time(end))

    return {"type": "Brain", "time": start.elapsed_time(end)}
