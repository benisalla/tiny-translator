# Testing guide

Three levels of checks, from seconds to minutes:

1. **Unit tests** (`pytest`): every module, CPU only, no network.
2. **Toy end-to-end run**: train on the built-in synthetic English -> Arabic set and score it.
3. **Overfit check on your data**: can the model memorise 8 of your pairs?

## 1. Unit tests

```bash
pytest                          # everything (a few minutes on a laptop CPU)
pytest tests/test_model.py      # one file
pytest -k beam -v               # tests whose name contains "beam"
ruff check src tests            # lint
ruff format --check src tests
```

No GPU and no internet needed: the tests build a synthetic English -> Arabic
dataset in a temporary folder (`en_2_ar_translator.data.write_toy_dataset`) and
train small models on it.

| File | What it proves |
|------|----------------|
| `test_config.py` | invalid model shapes are rejected; configs survive the round trip through a checkpoint, unknown keys are ignored |
| `test_tokenizer.py` | special ids are 0-3; English **and Arabic** round-trip exactly (no reversal, logical order kept); special tokens are dropped on decode; save/load gives identical ids, including for CJK text |
| `test_data.py` | misaligned files are rejected; `<s> y` / `y </s>` shifting is right; over-long pairs are dropped; length-bucketed batches cover every pair exactly once and stay under the token budget; shuffling is seeded; cleaning (whitespace, empties, length ratio, duplicates) and splitting; `prepare` writes a complete folder |
| `test_model.py` | output shapes; tied weights; sinusoidal table values; **the decoder is causal** (changing token t never changes earlier logits); **padding is invisible** (a padded row equals the unpadded sentence); the source changes the output; gradients reach every parameter; the base preset has the expected size |
| `test_translate.py` | greedy and beam search return one string per input; generation never overruns the position table; beam size 1 == greedy; beam hypotheses are sorted; batched greedy == one-by-one greedy |
| `test_train.py` | Noam schedule shape; loss ignores padding; a real training run writes `best.pt`, `last.pt`, `log.jsonl`, keeps only the last N step checkpoints and **lowers the validation loss**; **the trained model translates >90 % of unseen toy sentences exactly**; resume continues the step count with the optimiser state; averaging equals the mean of the weights; translate / evaluate / train CLIs end to end |

## 2. Toy end-to-end run

```bash
en2ar-prepare --toy --out-dir data/toy
en2ar-train --data-dir data/toy --preset small --max-steps 1500 --warmup-steps 200 \
    --eval-every 500 --tokens-per-batch 1000 --device cpu --output-dir checkpoints/toy
en2ar-translate --checkpoint checkpoints/toy/best.pt --text "three red bird" --text "blue cat"
en2ar-evaluate --checkpoint checkpoints/toy/best.pt --beam-size 1
```

The toy language is a 16-word lexicon translated word by word, so a working
model reaches near-perfect scores (reference run: BLEU 98.4 / chrF 99.0 with
greedy decoding, see the README). If it does not, the install is broken.

## 3. Overfit check on your data

```bash
en2ar-train --data-dir data/en-ar --src-lang en --tgt-lang ar --preset small \
    --overfit 8 --max-steps 500 --warmup-steps 50 --eval-every 100 --output-dir checkpoints/overfit
```

The validation loss is computed on the same 8 pairs and should approach 0;
the sample translations printed at each evaluation should match the
references word for word.
