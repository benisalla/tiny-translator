import torch

from en_2_ar_translator.config import ModelConfig
from en_2_ar_translator.model import Translator
from en_2_ar_translator.translate import beam_translate, greedy_translate, translate


def untrained(tokenizer, max_len=32):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, n_embd=32, n_heads=4, n_layers=1, dropout=0.0, max_len=max_len)
    return Translator(cfg).eval()


def test_greedy_returns_one_string_per_input(toy_tokenizer):
    model = untrained(toy_tokenizer)
    out = greedy_translate(model, toy_tokenizer, ["one two", "red cat dog blue", ""])
    assert len(out) == 3 and all(isinstance(o, str) for o in out)


def test_generation_never_exceeds_position_table(toy_tokenizer):
    model = untrained(toy_tokenizer, max_len=12)
    long_text = " ".join(["one"] * 40)  # longer than max_len: source is truncated, output is capped
    greedy_translate(model, toy_tokenizer, [long_text])
    beam_translate(model, toy_tokenizer, long_text, beam_size=3)


def test_beam_of_one_matches_greedy(toy_tokenizer):
    model = untrained(toy_tokenizer)
    for text in ["one two three", "green bird"]:
        best, _ = beam_translate(model, toy_tokenizer, text, beam_size=1, length_penalty=0.0)
        assert best == greedy_translate(model, toy_tokenizer, [text])[0]


def test_beam_hypotheses_sorted(toy_tokenizer):
    model = untrained(toy_tokenizer)
    best, ranked = beam_translate(model, toy_tokenizer, "four five six", beam_size=4)
    scores = [s for _, s in ranked]
    assert best == ranked[0][0] and scores == sorted(scores, reverse=True)


def test_batched_greedy_equals_one_by_one(toy_tokenizer):
    model = untrained(toy_tokenizer)
    texts = ["one", "two three four five six", "red"]
    assert translate(model, toy_tokenizer, texts, beam_size=1, batch_size=8) == [
        greedy_translate(model, toy_tokenizer, [t])[0] for t in texts
    ]
