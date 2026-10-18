import pytest
import torch

from en_2_ar_translator.data import ParallelDataset, read_parallel, write_lines
from en_2_ar_translator.prepare import clean_pairs, prepare, split_pairs


def test_read_parallel_checks_alignment(tmp_path):
    write_lines(str(tmp_path / "train.en"), ["a", "b"])
    write_lines(str(tmp_path / "train.ar"), ["x"])
    with pytest.raises(ValueError, match="lines"):
        read_parallel(str(tmp_path), "train", "en", "ar")
    with pytest.raises(FileNotFoundError):
        read_parallel(str(tmp_path), "val", "en", "ar")


def test_collate_shifts_target(toy_dir, toy_tokenizer):
    ds = ParallelDataset(read_parallel(toy_dir, "val", "en", "ar"), toy_tokenizer, max_len=64)
    b = ds.collate(ds.examples[:4])
    tok = toy_tokenizer
    assert (b.tgt_in[:, 0] == tok.bos_id).all()
    for row in range(4):
        src_ids, tgt_ids = ds.examples[row]
        n = len(tgt_ids)
        assert b.tgt_in[row, 1 : n + 1].tolist() == tgt_ids  # <s> y1..yn
        assert b.tgt_out[row, :n].tolist() == tgt_ids and b.tgt_out[row, n] == tok.eos_id  # y1..yn </s>
        assert b.src[row, len(src_ids)] == tok.eos_id
    assert b.tgt_in.shape == b.tgt_out.shape


def test_long_pairs_dropped(toy_tokenizer):
    pairs = [("one two", "واحد اثنان"), (" ".join(["one"] * 50), "واحد")]
    ds = ParallelDataset(pairs, toy_tokenizer, max_len=10)
    assert len(ds) == 1 and ds.n_dropped == 1


def test_batches_cover_every_pair_once_and_respect_budget(toy_dir, toy_tokenizer):
    ds = ParallelDataset(read_parallel(toy_dir, "train", "en", "ar"), toy_tokenizer, max_len=64)
    budget = 200
    batches = ds.batches(budget, shuffle=True, seed=1)
    assert sum(b.src.size(0) for b in batches) == len(ds)
    for b in batches:
        assert b.src.numel() <= budget and b.tgt_in.numel() <= budget
    # same seed -> same order, different seed -> different order
    first = lambda bs: [b.src[0].tolist() for b in bs[:5]]  # noqa: E731
    assert first(batches) == first(ds.batches(budget, shuffle=True, seed=1))
    assert first(batches) != first(ds.batches(budget, shuffle=True, seed=2))


def test_epochs_stream_is_endless(toy_dir, toy_tokenizer):
    ds = ParallelDataset(read_parallel(toy_dir, "val", "en", "ar"), toy_tokenizer, max_len=64)
    n = len(ds.batches(100))
    stream = ds.epochs(100)
    got = [next(stream) for _ in range(n * 2 + 1)]
    assert all(isinstance(b.src, torch.Tensor) for b in got)


def test_clean_pairs():
    raw = [
        ("  hello   world ", "مرحبا  بالعالم"),
        ("hello world", "مرحبا بالعالم"),  # duplicate after whitespace normalisation
        ("", "x"),
        ("a", "a much much much longer sentence"),  # bad length ratio
        ("x" * 500, "y" * 500),  # too long
    ]
    assert list(clean_pairs(raw, max_chars=400, max_ratio=3.0)) == [("hello world", "مرحبا بالعالم")]


def test_split_pairs_sizes_and_disjoint():
    pairs = [(str(i), str(i)) for i in range(100)]
    s = split_pairs(pairs, n_val=10, n_test=5)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (85, 10, 5)
    assert len(set(s["train"]) | set(s["val"]) | set(s["test"])) == 100
    with pytest.raises(ValueError):
        split_pairs(pairs, n_val=60, n_test=40)


def test_prepare_writes_folder(tmp_path):
    from en_2_ar_translator.data import TOY_LEXICON

    pairs = [(f"{en} {i}", f"{ar} {i}") for i, (en, ar) in enumerate(TOY_LEXICON.items())] * 10
    stats = prepare(pairs, str(tmp_path), "en", "ar", n_val=3, n_test=3, vocab_size=150)
    assert stats["train"] == len(TOY_LEXICON) - 6  # duplicates removed
    for name in ["train.en", "train.ar", "val.en", "val.ar", "test.en", "test.ar", "tokenizer.json"]:
        assert (tmp_path / name).is_file()
    assert len(read_parallel(str(tmp_path), "val", "en", "ar")) == 3
