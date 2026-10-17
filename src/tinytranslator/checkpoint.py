"""Save / load self-contained checkpoints.

A checkpoint is one ``.pt`` file holding the model weights, the configs *and*
the tokenizer (as JSON), so ``translate`` needs nothing else.
"""

from __future__ import annotations

import torch
from tokenizers import Tokenizer as _HFTokenizer

from .config import DataConfig, ModelConfig, TrainConfig, from_dict, to_dict
from .model import Translator
from .tokenizer import Tokenizer


def save_checkpoint(path, model, tokenizer, data_cfg, train_cfg, step, best_val, optimizer=None) -> None:
    state = {
        "model": model.state_dict(),
        "model_cfg": to_dict(model.cfg),
        "data_cfg": to_dict(data_cfg),
        "train_cfg": to_dict(train_cfg),
        "tokenizer": tokenizer.hf.to_str(),
        "step": step,
        "best_val": best_val,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, device="cpu") -> tuple[Translator, Tokenizer, dict]:
    """Return ``(model in eval mode, tokenizer, raw checkpoint dict)``."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = Translator(from_dict(ModelConfig, state["model_cfg"]))
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tokenizer = Tokenizer(_HFTokenizer.from_str(state["tokenizer"]))
    state["data_cfg"] = from_dict(DataConfig, state["data_cfg"])
    state["train_cfg"] = from_dict(TrainConfig, state["train_cfg"])
    return model, tokenizer, state
