# Data guide

This project learns from **parallel text**: pairs of sentences that mean the
same thing in two languages. Everything is plain UTF-8 text, one sentence per
line.

## 1. Folder layout

```
data/
  train.en   train.ar      # line i of train.en  <->  line i of train.ar
  val.en     val.ar        # used for the validation loss and best.pt
  test.en    test.ar       # used by en2ar-evaluate
  tokenizer.json           # shared BPE vocabulary, learned from train.*
```

The suffixes are the language codes you pass as `--src-lang` / `--tgt-lang`.
Nothing is hard-coded to English or Arabic: `train.fr` + `train.de` with
`--src-lang fr --tgt-lang de` works the same way.

## 2. Build the folder with `en2ar-prepare`

`en2ar-prepare` cleans, de-duplicates, shuffles and splits a corpus,
then trains the tokenizer. Three ways to feed it:

**a. The built-in toy set** (no download, synthetic English -> Arabic, 16 words):

```bash
en2ar-prepare --toy --out-dir data/toy
```

Use it to check that your install works and that the model can learn (see
[testing.md](testing.md)). It is not a real translation dataset.

**b. A Hugging Face dataset** with a `translation` column (`pip install -e '.[data]'`):

```bash
# OPUS-100: 1M English-Arabic pairs (subtitles, web, religious and legal text...)
en2ar-prepare --hf-dataset Helsinki-NLP/opus-100 --hf-config ar-en \
    --src-lang en --tgt-lang ar --out-dir data/en-ar

# any other pair, e.g. English -> French
en2ar-prepare --hf-dataset Helsinki-NLP/opus-100 --hf-config en-fr \
    --src-lang en --tgt-lang fr --out-dir data/en-fr
```

Other good sources: `Helsinki-NLP/opus_books`, `facebook/flores` (evaluation
only), `Helsinki-NLP/un_pc`, `Helsinki-NLP/news_commentary`.

**c. Your own aligned files** (e.g. exported subtitles, a TMX export, a CSV split in two):

```bash
en2ar-prepare --src-file corpus.en --tgt-file corpus.ar \
    --src-lang en --tgt-lang ar --out-dir data/mine
```

### Cleaning rules

| Rule | Flag | Default |
|------|------|---------|
| whitespace collapsed, newlines removed | – | always |
| empty sentences dropped | – | always |
| sentences longer than N characters dropped | `--max-chars` | 400 |
| pairs where one side is N× longer than the other dropped (likely misaligned) | `--max-ratio` | 3.0 |
| exact duplicate pairs dropped | – | always |

### Other flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-val`, `--n-test` | 2000 | pairs held out for validation / test |
| `--vocab-size` | 16000 | BPE vocabulary size (shared by both languages) |
| `--max-pairs` | all | stop after this many clean pairs |
| `--seed` | 0 | shuffle seed |

## 3. The tokenizer

One BPE vocabulary covers both languages ([`tokenizer.py`](../src/en_2_ar_translator/tokenizer.py)):

- text is NFKC-normalised (unifies Arabic presentation forms, full-width
  characters, etc.), split on spaces SentencePiece-style (`▁` marks a word
  start), then BPE merges are learned;
- ids 0-3 are reserved: `<pad>`, `<s>`, `</s>`, `<unk>`;
- decoding is the exact inverse, so Arabic (and any right-to-left script) is
  stored and returned in its natural logical order: no reversing.

Rules of thumb for `--vocab-size`: 8k for < 200k pairs, 16k for ~1M pairs,
32k for 10M+ pairs. A shared vocabulary is what lets the model tie its input
and output embeddings (fewer parameters, better results on small data).

## 4. How many pairs do I need?

| Pairs | What to expect |
|-------|----------------|
| 2k (toy) | memorises the toy task perfectly; nothing else |
| 100k | short, common sentences come out right; long ones drift |
| 1M (OPUS-100 en-ar) | usable translations of everyday sentences after ~100k GPU steps |
| 10M+ | needed for strong general-purpose quality; use `--preset base` or larger |

## 5. Batching

While training, pairs are sorted by length and grouped so that each batch holds
about `--tokens-per-batch` tokens (padding included). Short sentences make big
batches, long ones small batches, and very little compute is wasted on padding.
Batch order is reshuffled every epoch. Pairs longer than `--max-len` tokens
(default 128) are skipped when the data is loaded.
