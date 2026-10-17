"""A shared source + target BPE tokenizer.

One vocabulary for both languages lets the encoder embedding, the decoder
embedding and the output projection share a single weight matrix (see
``model.py``). The tokenizer is trained from your own parallel text, so it
works for any language pair, any script (Latin, Arabic, CJK...).

Text is NFKC-normalised and split on whitespace SentencePiece-style (``▁``
marks the start of a word), then BPE merges are learned on top. Decoding
reverses this exactly, so Arabic and other right-to-left scripts come back in
their natural (logical) order: no string reversal hacks needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]  # ids 0, 1, 2, 3


class Tokenizer:
    """Thin wrapper around a Hugging Face ``tokenizers`` BPE model."""

    def __init__(self, hf: _HFTokenizer):
        self.hf = hf
        self.pad_id = hf.token_to_id(PAD)
        self.bos_id = hf.token_to_id(BOS)
        self.eos_id = hf.token_to_id(EOS)
        self.unk_id = hf.token_to_id(UNK)
        if None in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
            raise ValueError(f"tokenizer is missing one of the special tokens {SPECIAL_TOKENS}")

    @property
    def vocab_size(self) -> int:
        return self.hf.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Text -> token ids, without <s> / </s>."""
        return self.hf.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.hf.encode_batch(texts)]

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids -> text. Special tokens are dropped."""
        return self.hf.decode(list(ids), skip_special_tokens=True)

    def save(self, path: str) -> None:
        self.hf.save(path)

    @classmethod
    def load(cls, path: str) -> Tokenizer:
        return cls(_HFTokenizer.from_file(path))

    @classmethod
    def train(cls, texts: Iterable[str] | Iterator[str], vocab_size: int = 16000, min_frequency: int = 2) -> Tokenizer:
        """Learn a BPE vocabulary from an iterator of sentences (both languages mixed)."""
        hf = _HFTokenizer(models.BPE(unk_token=UNK))
        hf.normalizer = normalizers.NFKC()
        hf.pre_tokenizer = pre_tokenizers.Metaspace()
        hf.decoder = decoders.Metaspace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        hf.train_from_iterator(texts, trainer=trainer)
        return cls(hf)
