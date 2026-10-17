"""Every hyperparameter in one place.

Three dataclasses, one per concern:

- ``ModelConfig``: the shape of the Transformer (saved inside every checkpoint).
- ``DataConfig``: where the parallel text lives and how it is batched.
- ``TrainConfig``: optimiser, schedule, logging and checkpointing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelConfig:
    vocab_size: int = 16000  # shared source + target BPE vocabulary
    n_embd: int = 512  # model width
    n_heads: int = 8
    n_layers: int = 6  # encoder layers = decoder layers
    ffn_mult: int = 4  # feed-forward hidden size = ffn_mult * n_embd
    dropout: float = 0.1
    max_len: int = 256  # longest sequence (in tokens) the positional table covers
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_heads != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if min(self.vocab_size, self.n_layers, self.max_len, self.ffn_mult) <= 0:
            raise ValueError("vocab_size, n_layers, max_len and ffn_mult must be positive")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_heads

    @classmethod
    def tiny(cls, vocab_size: int) -> ModelConfig:
        """A ~1-3M-parameter model (depending on the vocabulary) that trains on a laptop CPU (tests, smoke runs)."""
        return cls(vocab_size=vocab_size, n_embd=128, n_heads=4, n_layers=2, dropout=0.1, max_len=128)


@dataclass
class DataConfig:
    data_dir: str = "data"
    src_lang: str = "en"
    tgt_lang: str = "ar"
    tokenizer: str = ""  # defaults to <data_dir>/tokenizer.json
    tokens_per_batch: int = 4000  # target tokens per batch (batches are bucketed by length)
    max_len: int = 128  # pairs longer than this (in tokens) are skipped while loading

    def tokenizer_path(self) -> str:
        return self.tokenizer or f"{self.data_dir.rstrip('/')}/tokenizer.json"


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/run"
    max_steps: int = 100_000  # optimiser steps
    accum_steps: int = 1  # batches accumulated per optimiser step
    warmup_steps: int = 4000
    lr_scale: float = 2.0  # peak lr = lr_scale * n_embd^-0.5 * warmup^-0.5
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    label_smoothing: float = 0.1
    grad_clip: float = 1.0  # 0 disables clipping
    log_every: int = 50
    eval_every: int = 1000
    keep_last: int = 5  # step checkpoints kept for averaging (0 keeps none)
    device: str = "auto"  # auto | cpu | cuda | mps
    seed: int = 0
    overfit: int = 0  # >0: train and validate on the first N pairs only (sanity check)
    sample_sentences: list[str] = field(default_factory=list)  # translated at every eval


def to_dict(cfg) -> dict:
    return asdict(cfg)


def from_dict(cls, d: dict):
    """Build a dataclass from a dict, ignoring unknown keys (forward-compatible checkpoints)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})
