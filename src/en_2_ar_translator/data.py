"""Parallel text on disk -> padded tensor batches.

Folder layout (one sentence per line, line i of each file is a translation pair)::

    data/
      train.en   train.ar
      val.en     val.ar
      test.en    test.ar
      tokenizer.json

The language codes are just file suffixes: ``fr``/``de``/``zh`` work the same.

Batches are *token-bucketed*: pairs of similar length are grouped so that each
batch holds about ``tokens_per_batch`` target tokens with little padding. Short
sentences give big batches, long sentences give small ones.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import Tokenizer


def split_paths(data_dir: str, split: str, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    return os.path.join(data_dir, f"{split}.{src_lang}"), os.path.join(data_dir, f"{split}.{tgt_lang}")


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\n", " ") + "\n")


def read_parallel(data_dir: str, split: str, src_lang: str, tgt_lang: str) -> list[tuple[str, str]]:
    src_path, tgt_path = split_paths(data_dir, split, src_lang, tgt_lang)
    for p in (src_path, tgt_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing {p} (see docs/data.md for the expected layout)")
    src, tgt = read_lines(src_path), read_lines(tgt_path)
    if len(src) != len(tgt):
        raise ValueError(f"{src_path} has {len(src)} lines but {tgt_path} has {len(tgt)}")
    return list(zip(src, tgt))


@dataclass
class Batch:
    src: torch.Tensor  # (B, S) source ids, padded
    tgt_in: torch.Tensor  # (B, T) <s> y1 ... yn, padded
    tgt_out: torch.Tensor  # (B, T) y1 ... yn </s>, padded

    def to(self, device) -> Batch:
        return Batch(self.src.to(device), self.tgt_in.to(device), self.tgt_out.to(device))

    @property
    def n_tokens(self) -> int:
        return int((self.tgt_out != 0).sum())  # <pad> is always id 0 (see tokenizer.SPECIAL_TOKENS)


class ParallelDataset:
    """Tokenised sentence pairs. Pairs that are empty or longer than ``max_len`` tokens are dropped."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int):
        self.tokenizer = tokenizer
        src_ids = tokenizer.encode_batch([s for s, _ in pairs])
        tgt_ids = tokenizer.encode_batch([t for _, t in pairs])
        # +1 for </s> (source) and +1 for <s>/</s> (target): the model sees len+1 positions
        self.examples = [
            (s, t) for s, t in zip(src_ids, tgt_ids) if s and t and len(s) + 1 <= max_len and len(t) + 1 <= max_len
        ]
        self.n_dropped = len(pairs) - len(self.examples)

    @classmethod
    def from_dir(cls, data_dir, split, src_lang, tgt_lang, tokenizer, max_len) -> ParallelDataset:
        return cls(read_parallel(data_dir, split, src_lang, tgt_lang), tokenizer, max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def collate(self, examples: list[tuple[list[int], list[int]]]) -> Batch:
        tok = self.tokenizer
        src = [torch.tensor(s + [tok.eos_id]) for s, _ in examples]
        tgt_in = [torch.tensor([tok.bos_id] + t) for _, t in examples]
        tgt_out = [torch.tensor(t + [tok.eos_id]) for _, t in examples]
        pad = lambda xs: pad_sequence(xs, batch_first=True, padding_value=tok.pad_id)  # noqa: E731
        return Batch(pad(src), pad(tgt_in), pad(tgt_out))

    def batches(self, tokens_per_batch: int, shuffle: bool = False, seed: int = 0) -> list[Batch]:
        """Group pairs of similar length into batches of ~``tokens_per_batch`` target tokens."""
        order = sorted(range(len(self.examples)), key=lambda i: (len(self.examples[i][1]), len(self.examples[i][0])))
        groups, current, width = [], [], 0  # width = longest padded row in `current`
        for i in order:
            s, t = self.examples[i]
            w = max(len(s), len(t)) + 1
            if current and max(width, w) * (len(current) + 1) > tokens_per_batch:
                groups.append(current)
                current, width = [], 0
            current.append(i)
            width = max(width, w)
        if current:
            groups.append(current)
        if shuffle:
            random.Random(seed).shuffle(groups)
        return [self.collate([self.examples[i] for i in g]) for g in groups]

    def epochs(self, tokens_per_batch: int, seed: int = 0) -> Iterator[Batch]:
        """Endless stream of shuffled batches, reshuffled at every epoch."""
        epoch = 0
        while True:
            yield from self.batches(tokens_per_batch, shuffle=True, seed=seed + epoch)
            epoch += 1


# ---- a tiny synthetic language pair, used by the tests and for smoke runs -------------------

TOY_LEXICON = {
    "one": "واحد", "two": "اثنان", "three": "ثلاثة", "four": "أربعة", "five": "خمسة",
    "six": "ستة", "seven": "سبعة", "eight": "ثمانية", "nine": "تسعة", "ten": "عشرة",
    "red": "أحمر", "blue": "أزرق", "green": "أخضر", "cat": "قطة", "dog": "كلب", "bird": "طائر",
}  # fmt: skip


def toy_pair(rng: random.Random, min_words: int = 2, max_words: int = 6) -> tuple[str, str]:
    """A random English word sequence and its word-by-word Arabic translation."""
    words = [rng.choice(list(TOY_LEXICON)) for _ in range(rng.randint(min_words, max_words))]
    return " ".join(words), " ".join(TOY_LEXICON[w] for w in words)


def write_toy_dataset(data_dir: str, n_train: int = 2000, n_val: int = 100, n_test: int = 100, seed: int = 0):
    """Write train/val/test .en/.ar files plus a trained tokenizer.json to ``data_dir``."""
    os.makedirs(data_dir, exist_ok=True)
    rng = random.Random(seed)
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        pairs = [toy_pair(rng) for _ in range(n)]
        src_path, tgt_path = split_paths(data_dir, split, "en", "ar")
        write_lines(src_path, [s for s, _ in pairs])
        write_lines(tgt_path, [t for _, t in pairs])
    train = read_parallel(data_dir, "train", "en", "ar")
    tok = Tokenizer.train((x for pair in train for x in pair), vocab_size=200, min_frequency=1)
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    return tok
