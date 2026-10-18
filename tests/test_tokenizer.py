from en_2_ar_translator.tokenizer import Tokenizer


def test_special_token_ids(toy_tokenizer):
    assert (toy_tokenizer.pad_id, toy_tokenizer.bos_id, toy_tokenizer.eos_id, toy_tokenizer.unk_id) == (0, 1, 2, 3)


def test_round_trip_english_and_arabic(toy_tokenizer):
    for text in ["one two three red cat", "واحد اثنان ثلاثة أحمر قطة"]:
        ids = toy_tokenizer.encode(text)
        assert ids and all(i >= 4 for i in ids)  # no special / unknown tokens for in-vocabulary text
        assert toy_tokenizer.decode(ids) == text  # Arabic comes back in logical order, untouched


def test_decode_drops_special_tokens(toy_tokenizer):
    ids = toy_tokenizer.encode("blue dog")
    assert toy_tokenizer.decode([toy_tokenizer.bos_id, *ids, toy_tokenizer.eos_id, 0, 0]) == "blue dog"


def test_train_save_load(tmp_path):
    texts = ["hello world", "bonjour le monde", "مرحبا بالعالم", "你好世界"] * 20
    tok = Tokenizer.train(texts, vocab_size=120, min_frequency=1)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    again = Tokenizer.load(path)
    for t in set(texts):
        assert again.encode(t) == tok.encode(t)
        assert again.decode(again.encode(t)) == t
