import math

import pytest
import torch

from en_2_ar_translator.config import ModelConfig
from en_2_ar_translator.model import Translator, sinusoidal_positions


def rand_ids(B, T, vocab=50, low=4):
    return torch.randint(low, vocab, (B, T))


def test_output_shape(small_model):
    logits = small_model(rand_ids(3, 7), rand_ids(3, 5))
    assert logits.shape == (3, 5, 50)


def test_weights_are_tied(small_model):
    assert small_model.lm_head.weight.data_ptr() == small_model.embed.weight.data_ptr()


def test_positions_table():
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4)) and torch.allclose(pe[0, 1::2], torch.ones(4))
    assert math.isclose(pe[3, 0].item(), math.sin(3), rel_tol=1e-5)


def test_decoder_is_causal(small_model):
    """Changing target token t must not change the logits at positions < t."""
    src, tgt = rand_ids(2, 6), rand_ids(2, 8)
    base = small_model(src, tgt)
    tgt2 = tgt.clone()
    tgt2[:, 5] = (tgt2[:, 5] + 1 - 4) % 46 + 4
    changed = small_model(src, tgt2)
    assert torch.allclose(base[:, :5], changed[:, :5], atol=1e-5)
    assert not torch.allclose(base[:, 5:], changed[:, 5:], atol=1e-5)


def test_padding_does_not_change_results(small_model):
    """A padded batch row gives the same logits as the same sentence alone."""
    src, tgt = rand_ids(1, 5), rand_ids(1, 4)
    alone = small_model(src, tgt)
    pad = small_model.cfg.pad_id
    src_padded = torch.cat([src, torch.full((1, 3), pad)], dim=1)
    tgt_padded = torch.cat([tgt, torch.full((1, 2), pad)], dim=1)
    batched = small_model(torch.cat([src_padded, rand_ids(1, 8)]), torch.cat([tgt_padded, rand_ids(1, 6)]))
    assert torch.allclose(alone[0], batched[0, :4], atol=1e-5)


def test_source_matters(small_model):
    tgt = rand_ids(1, 4)
    assert not torch.allclose(small_model(rand_ids(1, 5), tgt), small_model(rand_ids(1, 5), tgt))


def test_too_long_sequence_rejected(small_model):
    with pytest.raises(ValueError, match="max_len"):
        small_model(rand_ids(1, 65), rand_ids(1, 3))


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    model = Translator(ModelConfig(vocab_size=30, n_embd=16, n_heads=2, n_layers=1, dropout=0.1, max_len=16))
    model(rand_ids(2, 5, 30), rand_ids(2, 4, 30)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_base_model_size():
    """The default config is the Transformer-base shape (~52M params with a 16k shared vocabulary)."""
    n = Translator(ModelConfig()).num_parameters()
    assert 50e6 < n < 55e6
