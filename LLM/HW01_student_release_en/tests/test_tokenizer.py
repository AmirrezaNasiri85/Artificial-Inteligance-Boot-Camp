import pytest

from src.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, decode_tokens, encode_text


@pytest.mark.parametrize("text", ["Hello, MiniGPT!", "café naïve", "emoji: 🤖", "", "line\nnext"])
def test_round_trip(text):
    assert decode_tokens(encode_text(text)) == text


def test_encoding_is_deterministic_and_special_tokens_are_added():
    assert encode_text("abc") == encode_text("abc")
    encoded = encode_text("x", add_bos=True, add_eos=True)
    assert encoded[0] == BOS_TOKEN_ID
    assert encoded[-1] == EOS_TOKEN_ID


def test_empty_input_with_special_tokens():
    assert encode_text("") == []
    assert encode_text("", add_bos=True, add_eos=True) == [BOS_TOKEN_ID, EOS_TOKEN_ID]


def test_decoding_special_or_invalid_tokens_is_rejected():
    with pytest.raises(ValueError):
        decode_tokens([BOS_TOKEN_ID])
    with pytest.raises(ValueError):
        decode_tokens([999])
