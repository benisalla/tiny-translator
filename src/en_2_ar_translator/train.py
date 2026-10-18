"""Training loop and the ``en2ar-train`` CLI.

    en2ar-train --data-dir data --src-lang en --tgt-lang ar --output-dir checkpoints/run

Writes to ``--output-dir``:

- ``last.pt``          latest weights + optimiser state (used by ``--resume``)
- ``best.pt``          weights with the lowest validation loss so far
- ``step_<N>.pt``      the last ``--keep-last`` evaluation checkpoints (for ``en2ar-average``)
- ``log.jsonl``        one JSON line per log / eval event
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .config import DataConfig, ModelConfig, TrainConfig
from .data import Batch, ParallelDataset, read_parallel
from .model import Translator
from .tokenizer import Tokenizer
from .translate import greedy_translate, pick_device


def noam_lr(step: int, n_embd: int, warmup_steps: int, scale: float = 2.0) -> float:
    """Linear warm-up for ``warmup_steps``, then decay as 1/sqrt(step) (Vaswani et al., 2017)."""
    step = max(step, 1)
    return scale * n_embd**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def smoothed_loss(logits: torch.Tensor, target: torch.Tensor, pad_id: int, eps: float) -> torch.Tensor:
    """Label-smoothed cross-entropy averaged over non-pad target tokens."""
    return F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=pad_id, label_smoothing=eps)


@torch.no_grad()
def evaluate_loss(model: Translator, batches: list[Batch], device) -> float:
    """Token-weighted validation cross-entropy (no label smoothing, so exp(loss) is the perplexity)."""
    model.eval()
    total, n = 0.0, 0
    for b in batches:
        b = b.to(device)
        logits = model(b.src, b.tgt_in)
        total += F.cross_entropy(
            logits.flatten(0, 1), b.tgt_out.flatten(), ignore_index=model.cfg.pad_id, reduction="sum"
        ).item()
        n += b.n_tokens
    model.train()
    return total / max(n, 1)


def train(model_cfg: ModelConfig, data_cfg: DataConfig, train_cfg: TrainConfig, resume: str | None = None) -> dict:
    """Train a translator. Returns a summary dict (final step, best validation loss, paths)."""
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    device = pick_device(train_cfg.device)
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(train_cfg.output_dir, "log.jsonl")

    # ---- tokenizer + data ------------------------------------------------------------------
    tokenizer = Tokenizer.load(data_cfg.tokenizer_path())
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.pad_id, model_cfg.bos_id, model_cfg.eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
    max_len = min(data_cfg.max_len, model_cfg.max_len)

    train_pairs = read_parallel(data_cfg.data_dir, "train", data_cfg.src_lang, data_cfg.tgt_lang)
    if train_cfg.overfit:
        train_pairs = train_pairs[: train_cfg.overfit]
        val_pairs = train_pairs
    else:
        val_pairs = read_parallel(data_cfg.data_dir, "val", data_cfg.src_lang, data_cfg.tgt_lang)
    train_set = ParallelDataset(train_pairs, tokenizer, max_len)
    val_set = ParallelDataset(val_pairs, tokenizer, max_len)
    if len(train_set) == 0:
        raise ValueError("no training pairs left after filtering; check --data-dir, languages and --max-len")
    val_batches = val_set.batches(data_cfg.tokens_per_batch)
    samples = train_cfg.sample_sentences or [s for s, _ in val_pairs[:3]]

    # ---- model + optimiser -----------------------------------------------------------------
    step, best_val = 0, math.inf
    if resume:
        model, _, state = load_checkpoint(resume, device=device)
        model.train()
        step, best_val = state["step"], state["best_val"]
    else:
        model = Translator(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=tuple(train_cfg.betas), eps=train_cfg.eps)
    if resume and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    print(
        f"device={device} params={model.num_parameters() / 1e6:.2f}M vocab={tokenizer.vocab_size} "
        f"train_pairs={len(train_set)} (dropped {train_set.n_dropped}) val_pairs={len(val_set)} start_step={step}"
    )

    def log(event: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save(name: str) -> str:
        path = os.path.join(train_cfg.output_dir, name)
        save_checkpoint(path, model, tokenizer, data_cfg, train_cfg, step, best_val, optimizer)
        return path

    # ---- loop ------------------------------------------------------------------------------
    stream = train_set.epochs(data_cfg.tokens_per_batch, seed=train_cfg.seed + step)
    model.train()
    t0, running, running_tokens = time.time(), 0.0, 0
    step_ckpts: list[str] = []
    while step < train_cfg.max_steps:
        lr = noam_lr(step + 1, model.cfg.n_embd, train_cfg.warmup_steps, train_cfg.lr_scale)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.accum_steps):
            b = next(stream).to(device)
            loss = smoothed_loss(model(b.src, b.tgt_in), b.tgt_out, model.cfg.pad_id, train_cfg.label_smoothing)
            (loss / train_cfg.accum_steps).backward()
            running += loss.item() * b.n_tokens
            running_tokens += b.n_tokens
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % train_cfg.log_every == 0 or step == train_cfg.max_steps:
            elapsed = time.time() - t0
            train_loss = running / max(running_tokens, 1)
            print(f"step {step:>7} | loss {train_loss:.4f} | lr {lr:.2e} | {running_tokens / elapsed:,.0f} tok/s")
            log({"step": step, "train_loss": train_loss, "lr": lr})
            t0, running, running_tokens = time.time(), 0.0, 0

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss = evaluate_loss(model, val_batches, device)
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            outputs = greedy_translate(model, tokenizer, samples)
            model.train()
            ppl = math.exp(min(val_loss, 50))
            print(f"  eval | val_loss {val_loss:.4f} | ppl {ppl:.2f}{' | new best' if improved else ''}")
            for src_text, out in zip(samples, outputs):
                print(f"    {src_text}  ->  {out}")
            log({"step": step, "val_loss": val_loss, "samples": dict(zip(samples, outputs))})
            if improved:
                save("best.pt")
            save("last.pt")
            if train_cfg.keep_last > 0:
                step_ckpts.append(save(f"step_{step}.pt"))
                while len(step_ckpts) > train_cfg.keep_last:
                    os.remove(step_ckpts.pop(0))

    return {
        "step": step,
        "best_val": best_val,
        "best": os.path.join(train_cfg.output_dir, "best.pt"),
        "last": os.path.join(train_cfg.output_dir, "last.pt"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Transformer translator on parallel text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d, m, t = DataConfig(), ModelConfig(), TrainConfig()
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=d.data_dir, help="folder with train/val.<lang> files")
    g.add_argument("--src-lang", default=d.src_lang)
    g.add_argument("--tgt-lang", default=d.tgt_lang)
    g.add_argument("--tokenizer", default="", help="tokenizer.json (default: <data-dir>/tokenizer.json)")
    g.add_argument("--tokens-per-batch", type=int, default=d.tokens_per_batch)
    g.add_argument("--max-len", type=int, default=d.max_len, help="skip pairs longer than this many tokens")

    g = p.add_argument_group("model")
    g.add_argument("--preset", choices=["base", "small"], default="base", help="small = 2 layers x 128 dims")
    g.add_argument("--n-embd", type=int, default=None, help=f"override the preset (base: {m.n_embd})")
    g.add_argument("--n-heads", type=int, default=None)
    g.add_argument("--n-layers", type=int, default=None)
    g.add_argument("--dropout", type=float, default=None)

    g = p.add_argument_group("training")
    g.add_argument("--output-dir", default=t.output_dir)
    g.add_argument("--max-steps", type=int, default=t.max_steps)
    g.add_argument("--accum-steps", type=int, default=t.accum_steps, help="batches per optimiser step")
    g.add_argument("--warmup-steps", type=int, default=t.warmup_steps)
    g.add_argument("--lr-scale", type=float, default=t.lr_scale)
    g.add_argument("--label-smoothing", type=float, default=t.label_smoothing)
    g.add_argument("--grad-clip", type=float, default=t.grad_clip)
    g.add_argument("--log-every", type=int, default=t.log_every)
    g.add_argument("--eval-every", type=int, default=t.eval_every)
    g.add_argument("--keep-last", type=int, default=t.keep_last)
    g.add_argument("--device", default=t.device)
    g.add_argument("--seed", type=int, default=t.seed)
    g.add_argument("--overfit", type=int, default=0, help="sanity check: train/validate on the first N pairs")
    g.add_argument("--sample", action="append", default=[], help="sentence translated at every eval (repeatable)")
    g.add_argument("--resume", default=None, help="continue from a last.pt checkpoint")
    return p


def configs_from_args(args) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    data_cfg = DataConfig(
        data_dir=args.data_dir, src_lang=args.src_lang, tgt_lang=args.tgt_lang, tokenizer=args.tokenizer,
        tokens_per_batch=args.tokens_per_batch, max_len=args.max_len,
    )  # fmt: skip
    base = ModelConfig.small(vocab_size=1) if args.preset == "small" else ModelConfig()
    overrides = {
        k: getattr(args, k) for k in ("n_embd", "n_heads", "n_layers", "dropout") if getattr(args, k) is not None
    }
    model_cfg = ModelConfig(**{**base.__dict__, **overrides, "max_len": max(base.max_len, args.max_len)})
    train_cfg = TrainConfig(
        output_dir=args.output_dir, max_steps=args.max_steps, accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps, lr_scale=args.lr_scale, label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip, log_every=args.log_every, eval_every=args.eval_every, keep_last=args.keep_last,
        device=args.device, seed=args.seed, overfit=args.overfit, sample_sentences=args.sample,
    )  # fmt: skip
    return model_cfg, data_cfg, train_cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = train(*configs_from_args(args), resume=args.resume)
    print(f"done: step {summary['step']}, best val loss {summary['best_val']:.4f} -> {summary['best']}")


if __name__ == "__main__":
    main()
