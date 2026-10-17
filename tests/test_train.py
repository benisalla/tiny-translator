import json
import math
import os

import pytest
import torch

from tinytranslator import average, evaluate
from tinytranslator.checkpoint import load_checkpoint
from tinytranslator.config import DataConfig, ModelConfig, TrainConfig
from tinytranslator.data import read_parallel
from tinytranslator.train import main as train_main
from tinytranslator.train import noam_lr, smoothed_loss
from tinytranslator.train import train as run_train
from tinytranslator.translate import beam_translate, translate
from tinytranslator.translate import main as translate_main


def test_noam_schedule_warms_up_then_decays():
    lrs = [noam_lr(s, 512, warmup_steps=100) for s in (1, 50, 100, 400)]
    assert lrs[0] < lrs[1] < lrs[2] and lrs[3] == pytest.approx(lrs[2] / 2)


def test_smoothed_loss_ignores_padding():
    logits = torch.randn(2, 3, 10)
    target = torch.tensor([[5, 6, 0], [7, 0, 0]])
    full = smoothed_loss(logits, target, pad_id=0, eps=0.0)
    manual = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 0], logits[0, 1], logits[1, 0]]), torch.tensor([5, 6, 7])
    )
    assert full.item() == pytest.approx(manual.item(), rel=1e-5)


def tiny_run(toy_dir, out_dir, max_steps, **train_kw):
    model_cfg = ModelConfig(vocab_size=1, n_embd=64, n_heads=4, n_layers=2, dropout=0.0, max_len=64)
    data_cfg = DataConfig(data_dir=toy_dir, tokens_per_batch=600, max_len=64)
    train_cfg = TrainConfig(
        output_dir=str(out_dir), max_steps=max_steps, warmup_steps=50, lr_scale=1.0, log_every=50,
        eval_every=100, keep_last=3, device="cpu", **train_kw,
    )  # fmt: skip
    return run_train(model_cfg, data_cfg, train_cfg)


@pytest.fixture(scope="module")
def trained(toy_dir, tmp_path_factory):
    """Train a tiny model on the toy English -> Arabic data (~30 s on CPU)."""
    out = tmp_path_factory.mktemp("run")
    summary = tiny_run(toy_dir, out, max_steps=600)
    return out, summary


def test_training_writes_artifacts(trained):
    out, summary = trained
    assert summary["step"] == 600
    for name in ["best.pt", "last.pt", "log.jsonl"]:
        assert (out / name).is_file()
    steps = sorted(int(f[5:-3]) for f in os.listdir(out) if f.startswith("step_"))
    assert steps == [400, 500, 600]  # keep_last=3
    events = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    val = [e["val_loss"] for e in events if "val_loss" in e]
    assert val[-1] < val[0] and min(val) == pytest.approx(summary["best_val"])


def test_model_learns_to_translate(trained, toy_dir):
    """The toy task is word-by-word: a trained model should get almost every test sentence exactly right."""
    out, _ = trained
    model, tokenizer, _ = load_checkpoint(str(out / "best.pt"))
    pairs = read_parallel(toy_dir, "test", "en", "ar")
    hyps = translate(model, tokenizer, [s for s, _ in pairs], beam_size=1)
    exact = sum(h == t for h, (_, t) in zip(hyps, pairs)) / len(pairs)
    assert exact > 0.9, list(zip(hyps, pairs))[:5]
    assert beam_translate(model, tokenizer, "red cat", beam_size=4)[0] == "أحمر قطة"


def test_resume_continues_step_count(trained, toy_dir, tmp_path):
    out, _ = trained
    model_cfg = ModelConfig(vocab_size=1, n_embd=64, n_heads=4, n_layers=2, dropout=0.0, max_len=64)
    data_cfg = DataConfig(data_dir=toy_dir, tokens_per_batch=600, max_len=64)
    train_cfg = TrainConfig(output_dir=str(tmp_path), max_steps=650, warmup_steps=50, lr_scale=1.0,
                            eval_every=50, keep_last=0, device="cpu")  # fmt: skip
    summary = run_train(model_cfg, data_cfg, train_cfg, resume=str(out / "last.pt"))
    assert summary["step"] == 650
    _, _, state = load_checkpoint(summary["last"])
    assert state["step"] == 650 and "optimizer" in state


def test_average_checkpoints(trained, tmp_path):
    out, _ = trained
    paths = sorted(str(out / f) for f in os.listdir(out) if f.startswith("step_"))
    dst = str(tmp_path / "avg.pt")
    average.main([*paths, "--output", dst])
    avg_model, _, state = load_checkpoint(dst)
    models = [load_checkpoint(p)[0] for p in paths]
    name = "encoder.0.attn.q_proj.weight"
    expected = sum(m.state_dict()[name] for m in models) / len(models)
    assert torch.allclose(avg_model.state_dict()[name], expected, atol=1e-6)
    assert "optimizer" not in state and state["averaged_from"] == paths


def test_translate_cli(trained, tmp_path, capsys):
    out, _ = trained
    ckpt = str(out / "best.pt")
    translate_main(["--checkpoint", ckpt, "--text", "one two", "--text", "green dog", "--beam-size", "1"])
    assert capsys.readouterr().out.splitlines() == ["واحد اثنان", "أخضر كلب"]

    src = tmp_path / "in.en"
    src.write_text("three blue bird\nten\n", encoding="utf-8")
    dst = tmp_path / "out.ar"
    translate_main(["--checkpoint", ckpt, "--input", str(src), "--output", str(dst)])
    assert dst.read_text(encoding="utf-8").splitlines() == ["ثلاثة أزرق طائر", "عشرة"]


def test_evaluate_cli(trained, capsys):
    pytest.importorskip("sacrebleu")
    out, _ = trained
    evaluate.main(["--checkpoint", str(out / "best.pt"), "--limit", "30", "--beam-size", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["pairs"] == 30 and result["bleu"] > 80 and result["chrf"] > 80


def test_overfit_and_train_cli(toy_dir, tmp_path, capsys):
    out = tmp_path / "overfit"
    train_main([
        "--data-dir", toy_dir, "--preset", "tiny", "--n-embd", "64", "--overfit", "8", "--max-steps", "150",
        "--warmup-steps", "20", "--eval-every", "150", "--log-every", "50", "--device", "cpu",
        "--output-dir", str(out), "--sample", "one two",
    ])  # fmt: skip
    text = capsys.readouterr().out
    assert "train_pairs=8" in text and "one two  ->" in text
    _, _, state = load_checkpoint(str(out / "best.pt"))
    assert state["model_cfg"]["n_embd"] == 64 and state["model_cfg"]["n_layers"] == 2
    assert math.isfinite(state["best_val"])
