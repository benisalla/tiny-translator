# Inference guide

## 1. Command line

```bash
# one or more sentences
en2ar-translate --checkpoint checkpoints/en-ar/best.pt \
    --text "How are you?" --text "Where is the train station?"

# a file, one sentence per line -> a file
en2ar-translate --checkpoint checkpoints/en-ar/best.pt \
    --input subtitles.en --output subtitles.ar

# stdin -> stdout (pipes)
cat subtitles.en | en2ar-translate --checkpoint checkpoints/en-ar/best.pt > subtitles.ar
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--beam-size` | 4 | 1 = greedy (fast, batched); 4-5 = better quality |
| `--length-penalty` | 0.6 | higher favours longer translations (beam search only) |
| `--max-len` | 2 × source + 10 | max tokens generated per sentence |
| `--device` | auto | `cpu`, `cuda`, `mps` |

The output has exactly one line per input line (empty lines stay empty-ish), so
line-aligned formats like subtitle text keep their alignment.

## 2. Python

```python
from en_2_ar_translator import load_checkpoint, translate, beam_translate

model, tokenizer, _ = load_checkpoint("checkpoints/en-ar/best.pt", device="cpu")

translate(model, tokenizer, ["Good morning!", "Thank you very much."], beam_size=4)
# ['صباح الخير!', 'شكرا جزيلا.']   (output depends on your training data)

best, hypotheses = beam_translate(model, tokenizer, "See you tomorrow.", beam_size=5)
for text, score in hypotheses:
    print(f"{score:.3f}  {text}")
```

`translate(..., beam_size=1, batch_size=64)` runs batched greedy decoding,
which is the fastest option for large files.

## 3. Evaluate

```bash
pip install -e '.[eval]'
en2ar-evaluate --checkpoint checkpoints/en-ar/best.pt --split test --beam-size 4 \
    --output test.hyp.ar
```

Prints a JSON summary with corpus **BLEU** and **chrF** (sacreBLEU defaults).
For Arabic and other morphologically rich languages, chrF (character n-grams)
tracks quality better than BLEU (whole words), because one Arabic word often
packs a pronoun, a preposition and an article together.

## 4. Other language pairs

Nothing in inference is specific to a language: the checkpoint carries its own
tokenizer and the language codes it was trained on. A model trained on
`--src-lang fr --tgt-lang de` is used with the exact same commands.
