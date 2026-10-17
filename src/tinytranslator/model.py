"""The Transformer encoder-decoder ("Attention Is All You Need", pre-norm variant).

Shapes used in comments: B = batch, S = source length, T = target length,
C = n_embd, H = n_heads, D = head_size (C = H * D), V = vocab_size.

    source ids (B, S) ──► Encoder ──► memory (B, S, C)
                                           │ cross-attention
    target ids (B, T) ──► Decoder ◄────────┘ ──► logits (B, T, V)

Each block is pre-norm: ``x = x + Dropout(Sublayer(LayerNorm(x)))``, which
trains stably without a long warm-up. Source embedding, target embedding and
the output projection share one weight matrix (one shared BPE vocabulary).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


def sinusoidal_positions(max_len: int, n_embd: int) -> torch.Tensor:
    """Fixed sin/cos position table, (max_len, n_embd)."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    freq = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd))
    table = torch.zeros(max_len, n_embd)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq[: n_embd // 2])
    return table


class MultiHeadAttention(nn.Module):
    """Multi-head attention. Self-attention when ``kv is x``, cross-attention otherwise."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_size = cfg.n_heads, cfg.head_size
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, Tq, C) queries, kv: (B, Tk, C), mask: bool (B, 1, Tq|1, Tk), True = may attend."""
        B, Tq, C = x.shape
        Tk = kv.size(1)
        q = self.q_proj(x).view(B, Tq, self.n_heads, self.head_size).transpose(1, 2)  # (B, H, Tq, D)
        k, v = self.kv_proj(kv).split(C, dim=-1)
        k = k.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)  # (B, H, Tk, D)
        v = v.view(B, Tk, self.n_heads, self.head_size).transpose(1, 2)
        # softmax(q k^T / sqrt(D)) v, masked positions get -inf before the softmax
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )  # (B, H, Tq, D)
        return self.out_proj(out.transpose(1, 2).reshape(B, Tq, C))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.n_embd
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, cfg.n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), MultiHeadAttention(cfg)
        self.ln2, self.ffn = nn.LayerNorm(cfg.n_embd), FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, src_mask))
        return x + self.drop(self.ffn(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1, self.self_attn = nn.LayerNorm(cfg.n_embd), MultiHeadAttention(cfg)
        self.ln2, self.cross_attn = nn.LayerNorm(cfg.n_embd), MultiHeadAttention(cfg)
        self.ln3, self.ffn = nn.LayerNorm(cfg.n_embd), FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, tgt_mask, src_mask) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, tgt_mask))  # look at previous target tokens
        x = x + self.drop(self.cross_attn(self.ln2(x), memory, src_mask))  # look at the source
        return x + self.drop(self.ffn(self.ln3(x)))


class Translator(nn.Module):
    """Encoder-decoder Transformer for sequence-to-sequence translation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=cfg.pad_id)
        self.register_buffer("positions", sinusoidal_positions(cfg.max_len, cfg.n_embd), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))
        self.decoder = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.enc_ln = nn.LayerNorm(cfg.n_embd)
        self.dec_ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.cfg.n_embd**-0.5)
        with torch.no_grad():
            self.embed.weight[self.cfg.pad_id].zero_()
        self.lm_head.weight = self.embed.weight  # weight tying

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- masks -------------------------------------------------------------------------------

    def src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """(B, S) ids -> (B, 1, 1, S): every query may attend to every non-pad source token."""
        return (src != self.cfg.pad_id)[:, None, None, :]

    def tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """(B, T) ids -> (B, 1, T, T): non-pad AND not in the future (causal)."""
        T = tgt.size(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=tgt.device).tril()
        return (tgt != self.cfg.pad_id)[:, None, None, :] & causal

    # ---- forward -----------------------------------------------------------------------------

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        if T > self.cfg.max_len:
            raise ValueError(f"sequence of {T} tokens is longer than max_len={self.cfg.max_len}")
        return self.drop(self.embed(ids) * math.sqrt(self.cfg.n_embd) + self.positions[:T])

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """(B, S) source ids -> (B, S, C) memory."""
        mask = self.src_mask(src)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_ln(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """(B, T) target ids so far + encoder memory -> (B, T, V) next-token logits."""
        tgt_mask, src_mask = self.tgt_mask(tgt), self.src_mask(src)
        x = self._embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.lm_head(self.dec_ln(x))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: ``tgt_in`` is ``<s> y1 ... yn``; logits predict ``y1 ... yn </s>``."""
        return self.decode(tgt_in, self.encode(src), src)
