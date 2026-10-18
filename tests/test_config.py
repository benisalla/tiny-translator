import pytest

from en_2_ar_translator.config import DataConfig, ModelConfig, TrainConfig, from_dict, to_dict


def test_head_size():
    assert ModelConfig(n_embd=512, n_heads=8).head_size == 64


@pytest.mark.parametrize(
    "kwargs",
    [{"n_embd": 100, "n_heads": 3}, {"dropout": 1.0}, {"dropout": -0.1}, {"n_layers": 0}, {"vocab_size": 0}],
)
def test_invalid_model_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_round_trip_ignores_unknown_keys():
    for cls in (ModelConfig, DataConfig, TrainConfig):
        d = to_dict(cls())
        assert from_dict(cls, {**d, "not_a_field": 1}) == cls()


def test_tokenizer_path_default():
    assert DataConfig(data_dir="data/").tokenizer_path() == "data/tokenizer.json"
    assert DataConfig(tokenizer="x.json").tokenizer_path() == "x.json"
