"""Score a checkpoint on a test split with BLEU and chrF: ``tinytranslator-evaluate``.

    tinytranslator-evaluate --checkpoint checkpoints/run/best.pt --data-dir data --split test

Needs ``sacrebleu`` (``pip install 'tiny-translator[eval]'``). For Arabic and
other morphologically rich languages, chrF is usually more telling than BLEU.
"""

from __future__ import annotations

import argparse
import json

from .checkpoint import load_checkpoint
from .data import read_parallel
from .translate import pick_device, translate


def score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    try:
        import sacrebleu
    except ImportError as e:  # pragma: no cover
        raise SystemExit("evaluation needs sacrebleu: pip install 'tiny-translator[eval]'") from e
    return {
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compute BLEU / chrF of a checkpoint on a data split.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None, help="default: the data dir the checkpoint was trained on")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="only score the first N pairs")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--output", help="also write the translations to this file")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    model, tokenizer, state = load_checkpoint(args.checkpoint, device=pick_device(args.device))
    data_cfg = state["data_cfg"]
    pairs = read_parallel(args.data_dir or data_cfg.data_dir, args.split, data_cfg.src_lang, data_cfg.tgt_lang)
    pairs = pairs[: args.limit] if args.limit else pairs
    hyps = translate(model, tokenizer, [s for s, _ in pairs], beam_size=args.beam_size)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("".join(h + "\n" for h in hyps))
    metrics = score(hyps, [t for _, t in pairs])
    result = {"split": args.split, "pairs": len(pairs), "beam_size": args.beam_size, **metrics}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
