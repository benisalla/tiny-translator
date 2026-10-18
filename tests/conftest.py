import pytest
import torch

from en_2_ar_translator.config import ModelConfig
from en_2_ar_translator.data import write_toy_dataset
from en_2_ar_translator.model import Translator


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    """A synthetic English -> Arabic dataset (train/val/test + tokenizer.json), built once per session."""
    path = tmp_path_factory.mktemp("toy_data")
    write_toy_dataset(str(path), n_train=1500, n_val=60, n_test=60)
    return str(path)


@pytest.fixture(scope="session")
def toy_tokenizer(toy_dir):
    from en_2_ar_translator.tokenizer import Tokenizer

    return Tokenizer.load(f"{toy_dir}/tokenizer.json")


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, n_embd=32, n_heads=4, n_layers=2, dropout=0.0, max_len=64)
    return Translator(cfg).eval()
