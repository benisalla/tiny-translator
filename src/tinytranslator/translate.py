"""Decoding: greedy (batched) and beam search, plus the ``tinytranslator-translate`` CLI.

Usage::

    tinytranslator-translate --checkpoint checkpoints/run/best.pt --text "How are you?"
    tinytranslator-translate --checkpoint checkpoints/run/best.pt --input sentences.en --output out.ar
    echo "Good morning" | tinytranslator-translate --checkpoint checkpoints/run/best.pt
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .model import Translator
from .tokenizer import Tokenizer


def _encode_sources(model: Translator, tokenizer: Tokenizer, texts: list[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    limit = model.cfg.max_len - 1  # keep room for </s>
    ids = [torch.tensor(tokenizer.encode(t)[:limit] + [tokenizer.eos_id]) for t in texts]
    return pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_id).to(device)


def _max_new_tokens(model: Translator, src: torch.Tensor, max_len: int | None) -> int:
    # a translation is rarely more than ~2x the source length; never beyond the position table
    default = int(src.size(1) * 2) + 10
    return min(max_len or default, model.cfg.max_len - 1)


@torch.no_grad()
def greedy_translate(
    model: Translator, tokenizer: Tokenizer, texts: list[str], max_len: int | None = None
) -> list[str]:
    """Translate a batch of sentences, always picking the most likely next token."""
    model.eval()
    src = _encode_sources(model, tokenizer, texts)
    memory = model.encode(src)
    B = src.size(0)
    out = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(_max_new_tokens(model, src, max_len)):
        next_tok = model.decode(out, memory, src)[:, -1].argmax(-1)
        next_tok = next_tok.masked_fill(done, tokenizer.pad_id)  # finished rows only get padding
        out = torch.cat([out, next_tok[:, None]], dim=1)
        done |= next_tok == tokenizer.eos_id
        if done.all():
            break
    return [tokenizer.decode(_strip(row.tolist(), tokenizer)) for row in out[:, 1:]]


def _strip(ids: list[int], tokenizer: Tokenizer) -> list[int]:
    return ids[: ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids


@torch.no_grad()
def beam_translate(
    model: Translator,
    tokenizer: Tokenizer,
    text: str,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_len: int | None = None,
) -> tuple[str, list[tuple[str, float]]]:
    """Beam search for one sentence. Returns ``(best, [(hypothesis, score), ...])``.

    Scores are summed log-probabilities divided by the GNMT length penalty
    ``((5 + len) / 6) ** length_penalty`` so long translations are not unfairly punished.
    """
    model.eval()
    src = _encode_sources(model, tokenizer, [text])  # (1, S)
    memory = model.encode(src)  # (1, S, C)
    device = src.device

    hyps = torch.full((1, 1), tokenizer.bos_id, dtype=torch.long, device=device)  # (k, t) alive hypotheses
    scores = torch.zeros(1, device=device)  # (k,) their log-probabilities
    finished: list[tuple[list[int], float]] = []

    for step in range(1, _max_new_tokens(model, src, max_len) + 1):
        k = hyps.size(0)
        logits = model.decode(hyps, memory.expand(k, -1, -1), src.expand(k, -1))[:, -1]  # (k, V)
        cand = (scores[:, None] + F.log_softmax(logits, dim=-1)).view(-1)  # (k * V,)
        top_scores, top_idx = cand.topk(min(beam_size, cand.numel()))
        V = logits.size(-1)
        parent, token = top_idx // V, top_idx % V
        hyps = torch.cat([hyps[parent], token[:, None]], dim=1)
        is_eos = token == tokenizer.eos_id
        norm = ((5 + step) / 6) ** length_penalty
        for h, s in zip(hyps[is_eos].tolist(), top_scores[is_eos].tolist()):
            finished.append((h[1:-1], s / norm))
        hyps, scores = hyps[~is_eos], top_scores[~is_eos]
        if len(finished) >= beam_size or hyps.size(0) == 0:
            break

    if not finished:  # nothing reached </s>: fall back to the unfinished beams
        norm = ((5 + hyps.size(1) - 1) / 6) ** length_penalty
        finished = [(h[1:], s / norm) for h, s in zip(hyps.tolist(), scores.tolist())]

    ranked = sorted(((tokenizer.decode(h), s) for h, s in finished), key=lambda x: -x[1])
    return ranked[0][0], ranked


def translate(model, tokenizer, texts: list[str], beam_size: int = 4, batch_size: int = 32, **kw) -> list[str]:
    """Translate a list of sentences: greedy (batched) when ``beam_size == 1``, beam search otherwise."""
    if beam_size <= 1:
        return [
            y
            for i in range(0, len(texts), batch_size)
            for y in greedy_translate(model, tokenizer, texts[i : i + batch_size], **kw)
        ]
    return [beam_translate(model, tokenizer, t, beam_size=beam_size, **kw)[0] for t in texts]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    from .checkpoint import load_checkpoint

    p = argparse.ArgumentParser(description="Translate sentences with a trained tiny-translator checkpoint.")
    p.add_argument("--checkpoint", required=True, help="path to best.pt / last.pt / averaged.pt")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", action="append", help="sentence to translate (repeatable)")
    src.add_argument("--input", help="file with one sentence per line (default: read stdin)")
    p.add_argument("--output", help="write translations here (default: stdout)")
    p.add_argument("--beam-size", type=int, default=4, help="1 = greedy decoding")
    p.add_argument("--length-penalty", type=float, default=0.6)
    p.add_argument("--max-len", type=int, default=None, help="max tokens generated per sentence")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=pick_device(args.device))
    if args.text:
        texts = args.text
    elif args.input:
        with open(args.input, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
    else:
        texts = [line.rstrip("\n") for line in sys.stdin]

    kw = {"max_len": args.max_len}
    if args.beam_size > 1:
        kw["length_penalty"] = args.length_penalty
    outputs = translate(model, tokenizer, texts, beam_size=args.beam_size, **kw)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("".join(o + "\n" for o in outputs))
    else:
        for o in outputs:
            print(o)


if __name__ == "__main__":
    main()
