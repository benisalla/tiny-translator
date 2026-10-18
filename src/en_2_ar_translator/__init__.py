"""English-to-Arabic Translator: neural machine translation with a compact Transformer (any language pair)."""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import DataConfig, ModelConfig, TrainConfig
from .model import Translator
from .tokenizer import Tokenizer
from .translate import beam_translate, greedy_translate, translate

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "Tokenizer",
    "Translator",
    "beam_translate",
    "greedy_translate",
    "load_checkpoint",
    "save_checkpoint",
    "translate",
]
