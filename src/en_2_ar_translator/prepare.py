"""Build a training folder from raw parallel text: ``en2ar-prepare``.

Two sources:

1. Your own aligned files (line i of one file translates line i of the other)::

       en2ar-prepare --src-file corpus.en --tgt-file corpus.ar \\
           --src-lang en --tgt-lang ar --out-dir data

2. Any Hugging Face dataset with a ``translation`` column (needs ``pip install datasets``)::

       en2ar-prepare --hf-dataset Helsinki-NLP/opus-100 --hf-config ar-en \\
           --src-lang en --tgt-lang ar --out-dir data --max-pairs 1000000

Steps: clean (strip, drop empty / too long / badly length-matched pairs,
de-duplicate) -> shuffle -> split into train / val / test -> learn the shared
BPE tokenizer on the training split.

3. A built-in synthetic English -> Arabic toy set (no download, trains on CPU)::

       en2ar-prepare --toy --out-dir data/toy
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Iterable, Iterator

from .data import read_lines, split_paths, write_lines, write_toy_dataset
from .tokenizer import Tokenizer


def clean_pairs(
    pairs: Iterable[tuple[str, str]], max_chars: int = 400, max_ratio: float = 3.0
) -> Iterator[tuple[str, str]]:
    """Yield normalised pairs, dropping empty, overly long, mismatched or duplicate ones."""
    seen = set()
    for src, tgt in pairs:
        src, tgt = " ".join(src.split()), " ".join(tgt.split())  # collapse whitespace / newlines
        if not src or not tgt or len(src) > max_chars or len(tgt) > max_chars:
            continue
        if max(len(src), len(tgt)) / min(len(src), len(tgt)) > max_ratio:
            continue  # one side is much longer: likely a misalignment
        if (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        yield src, tgt


def split_pairs(pairs: list, n_val: int, n_test: int, seed: int = 0) -> dict[str, list]:
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    if n_val + n_test >= len(pairs):
        raise ValueError(f"only {len(pairs)} pairs: not enough for {n_val} val + {n_test} test + some training")
    return {"test": pairs[:n_test], "val": pairs[n_test : n_test + n_val], "train": pairs[n_test + n_val :]}


def iter_hf_pairs(name: str, config: str | None, src_lang: str, tgt_lang: str, split: str) -> Iterator[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("--hf-dataset needs the `datasets` package: pip install 'en_2_ar_translator[data]'") from e
    ds = load_dataset(name, config, split=split, streaming=True)
    for row in ds:
        tr = row["translation"]
        yield tr[src_lang], tr[tgt_lang]


def prepare(
    pairs: Iterable[tuple[str, str]],
    out_dir: str,
    src_lang: str,
    tgt_lang: str,
    n_val: int = 2000,
    n_test: int = 2000,
    vocab_size: int = 16000,
    max_pairs: int | None = None,
    max_chars: int = 400,
    max_ratio: float = 3.0,
    seed: int = 0,
) -> dict[str, int]:
    """Clean, split and write ``pairs`` to ``out_dir``, then train ``out_dir/tokenizer.json``."""
    cleaned = []
    for pair in clean_pairs(pairs, max_chars=max_chars, max_ratio=max_ratio):
        cleaned.append(pair)
        if max_pairs and len(cleaned) >= max_pairs:
            break
    splits = split_pairs(cleaned, n_val, n_test, seed)
    os.makedirs(out_dir, exist_ok=True)
    for split, items in splits.items():
        src_path, tgt_path = split_paths(out_dir, split, src_lang, tgt_lang)
        write_lines(src_path, [s for s, _ in items])
        write_lines(tgt_path, [t for _, t in items])
    tokenizer = Tokenizer.train((x for pair in splits["train"] for x in pair), vocab_size=vocab_size)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    return {split: len(items) for split, items in splits.items()} | {"vocab": tokenizer.vocab_size}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Clean, split and tokenize a parallel corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--src-file", help="source-language file, one sentence per line")
    src.add_argument("--tgt-file", help="target-language file, aligned line by line with --src-file")
    src.add_argument("--hf-dataset", help="Hugging Face dataset name, e.g. Helsinki-NLP/opus-100")
    src.add_argument("--hf-config", default=None, help="dataset config, e.g. ar-en")
    src.add_argument("--hf-split", default="train")
    src.add_argument("--toy", action="store_true", help="write the built-in synthetic en -> ar toy dataset")
    p.add_argument("--src-lang", help="source language code (file suffix), e.g. en")
    p.add_argument("--tgt-lang", help="target language code (file suffix), e.g. ar")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--max-pairs", type=int, default=None, help="stop after this many clean pairs")
    p.add_argument("--max-chars", type=int, default=400, help="drop sentences longer than this")
    p.add_argument("--max-ratio", type=float, default=3.0, help="drop pairs whose length ratio exceeds this")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.toy:
        write_toy_dataset(args.out_dir, seed=args.seed)
        print(f"wrote the toy en -> ar dataset to {args.out_dir}")
        return
    if not (args.src_lang and args.tgt_lang):
        p.error("--src-lang and --tgt-lang are required")
    if args.hf_dataset:
        pairs = iter_hf_pairs(args.hf_dataset, args.hf_config, args.src_lang, args.tgt_lang, args.hf_split)
    elif args.src_file and args.tgt_file:
        src_lines, tgt_lines = read_lines(args.src_file), read_lines(args.tgt_file)
        if len(src_lines) != len(tgt_lines):
            p.error(f"--src-file has {len(src_lines)} lines but --tgt-file has {len(tgt_lines)}")
        pairs = zip(src_lines, tgt_lines)
    else:
        p.error("give either --src-file and --tgt-file, or --hf-dataset")

    stats = prepare(
        pairs, args.out_dir, args.src_lang, args.tgt_lang, n_val=args.n_val, n_test=args.n_test,
        vocab_size=args.vocab_size, max_pairs=args.max_pairs, max_chars=args.max_chars,
        max_ratio=args.max_ratio, seed=args.seed,
    )  # fmt: skip
    print(f"wrote {args.out_dir}: " + ", ".join(f"{k}={v:,}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
