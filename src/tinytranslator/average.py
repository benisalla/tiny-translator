"""Average the weights of several checkpoints: ``tinytranslator-average``.

Averaging the last few checkpoints of a run is a cheap quality boost used
since the original Transformer paper::

    tinytranslator-average checkpoints/run/step_*.pt --output checkpoints/run/averaged.pt
"""

from __future__ import annotations

import argparse

import torch


def average_checkpoints(paths: list[str], output: str) -> None:
    if not paths:
        raise ValueError("no checkpoints to average")
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = states[0]["model"].keys()
    for p, s in zip(paths, states):
        if s["model"].keys() != keys or s["model_cfg"] != states[0]["model_cfg"]:
            raise ValueError(f"{p} has a different architecture than {paths[0]}")
    averaged = {k: sum(s["model"][k].float() for s in states) / len(states) for k in keys}
    out = {k: v for k, v in states[-1].items() if k != "optimizer"}  # configs + tokenizer of the newest
    out["model"] = {k: v.to(states[0]["model"][k].dtype) for k, v in averaged.items()}
    out["averaged_from"] = list(paths)
    torch.save(out, output)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Average the weights of several checkpoints of the same run.")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    average_checkpoints(args.checkpoints, args.output)
    print(f"averaged {len(args.checkpoints)} checkpoints -> {args.output}")


if __name__ == "__main__":
    main()
