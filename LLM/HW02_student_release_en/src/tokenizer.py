"""Deterministic UTF-8 byte tokenizer."""

from __future__ import annotations

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260
SPECIAL_TOKEN_NAMES = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}


def encode_text(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    """Encode text as shifted UTF-8 bytes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    ids = [value + BYTE_OFFSET for value in text.encode("utf-8")]
    if add_bos:
        ids.insert(0, BOS_TOKEN_ID)
    if add_eos:
        ids.append(EOS_TOKEN_ID)
    return ids


def decode_tokens(token_ids: list[int]) -> str:
    """Decode byte tokens; reject special and invalid IDs."""
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list")
    raw = bytearray()
    for token_id in token_ids:
        if not isinstance(token_id, int):
            raise TypeError("all token IDs must be integers")
        if token_id in SPECIAL_TOKEN_NAMES:
            raise ValueError(f"cannot decode special token {SPECIAL_TOKEN_NAMES[token_id]}")
        if not BYTE_OFFSET <= token_id < VOCAB_SIZE:
            raise ValueError(f"token ID out of range: {token_id}")
        raw.append(token_id - BYTE_OFFSET)
    return raw.decode("utf-8", errors="replace")


def token_display(token_id: int) -> str:
    """Return a compact display form for one token."""
    if token_id in SPECIAL_TOKEN_NAMES:
        return SPECIAL_TOKEN_NAMES[token_id]
    if not BYTE_OFFSET <= token_id < VOCAB_SIZE:
        return f"<INVALID:{token_id}>"
    decoded = bytes([token_id - BYTE_OFFSET]).decode("utf-8", errors="replace")
    return {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(decoded, decoded)

