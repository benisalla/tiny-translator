# Training guide

```bash
tinytranslator-train --data-dir data/en-ar --src-lang en --tgt-lang ar --output-dir checkpoints/en-ar
```

## 1. Before a long run: the overfit check

If the model cannot memorise 8 pairs, something is wrong with the data or the
setup, and a long run would be wasted:

```bash
tinytranslator-train --data-dir data/en-ar --src-lang en --tgt-lang ar \
    --preset tiny --overfit 8 --max-steps 500 --warmup-steps 50 --eval-every 100 \
    --output-dir checkpoints/overfit
```

The validation loss (computed on the same 8 pairs) should fall towards 0 and
the printed sample translations should match the references.

## 2. Hardware

| Setup | Command |
|-------|---------|
| CPU | `--preset tiny --device cpu` (fine for the toy set and small experiments) |
| 1 GPU | default (`--device auto` picks CUDA, then Apple MPS, then CPU) |
| small GPU / out of memory | lower `--tokens-per-batch` and raise `--accum-steps` to keep the same effective batch |

Effective batch size = `tokens_per_batch × accum_steps`. The original
Transformer used ~25k tokens per step; `--tokens-per-batch 4000
--accum-steps 6` gets close to that while keeping each batch small enough for a
mid-range GPU.

Install a CUDA build of PyTorch first if `pip` gave you a CPU-only one, see
<https://pytorch.org/get-started/locally/>.

## 3. What gets written

```
checkpoints/en-ar/
  best.pt         lowest validation loss so far (use this to translate)
  last.pt         latest weights + optimiser state (use this to resume)
  step_<N>.pt     the last --keep-last eval checkpoints (for averaging)
  log.jsonl       {"step", "train_loss", "lr"} and {"step", "val_loss", "samples"} lines
```

Every checkpoint is self-contained: weights, configs **and the tokenizer**.
`tinytranslator-translate --checkpoint best.pt` needs nothing else.

At every evaluation the console shows the validation loss, the perplexity and
the translation of a few sentences (the first validation sentences, or your own
with `--sample "How are you?" --sample "Where is the station?"`).

## 4. Resume

```bash
tinytranslator-train ... --resume checkpoints/en-ar/last.pt --max-steps 200000
```

The step counter, the learning-rate schedule, the optimiser state and the best
validation loss all continue from the checkpoint. `--max-steps` is the *total*
number of steps, not the number of extra steps.

## 5. After training

```bash
# average the last checkpoints (a small, free quality gain)
tinytranslator-average checkpoints/en-ar/step_*.pt --output checkpoints/en-ar/averaged.pt

# score on the test split (pip install -e '.[eval]')
tinytranslator-evaluate --checkpoint checkpoints/en-ar/averaged.pt --beam-size 4
```

## 6. Every flag

| Flag | Default | Meaning |
|------|---------|---------|
| **data** | | |
| `--data-dir` | `data` | folder with `train/val.<lang>` files |
| `--src-lang`, `--tgt-lang` | `en`, `ar` | file suffixes of the two languages |
| `--tokenizer` | `<data-dir>/tokenizer.json` | shared BPE tokenizer |
| `--tokens-per-batch` | 4000 | tokens (incl. padding) per batch |
| `--max-len` | 128 | pairs longer than this many tokens are skipped |
| **model** | | |
| `--preset` | `base` | `base` (6 × 512, ~52M) or `tiny` (2 × 128, ~3M) |
| `--n-embd`, `--n-heads`, `--n-layers`, `--dropout` | from preset | override the preset |
| **optimisation** | | |
| `--max-steps` | 100000 | total optimiser steps |
| `--accum-steps` | 1 | batches accumulated per step |
| `--warmup-steps` | 4000 | linear warm-up length |
| `--lr-scale` | 2.0 | multiplies the Noam learning rate |
| `--label-smoothing` | 0.1 | ε of the smoothed cross-entropy |
| `--grad-clip` | 1.0 | max gradient norm (0 = off) |
| **bookkeeping** | | |
| `--output-dir` | `checkpoints/run` | where checkpoints and logs go |
| `--log-every` | 50 | steps between training-loss lines |
| `--eval-every` | 1000 | steps between validation + checkpoint |
| `--keep-last` | 5 | `step_<N>.pt` files kept |
| `--sample` | first 3 val sentences | sentence translated at each eval (repeatable) |
| `--overfit` | 0 | train and validate on the first N pairs only |
| `--resume` | – | continue from a `last.pt` |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps` |
| `--seed` | 0 | data order and weight initialisation |

## 7. Python API

```python
from tinytranslator import DataConfig, ModelConfig, TrainConfig
from tinytranslator.train import train

summary = train(
    ModelConfig(n_embd=256, n_heads=4, n_layers=4),
    DataConfig(data_dir="data/en-ar", src_lang="en", tgt_lang="ar"),
    TrainConfig(output_dir="checkpoints/en-ar", max_steps=50_000),
)
print(summary["best"], summary["best_val"])
```
