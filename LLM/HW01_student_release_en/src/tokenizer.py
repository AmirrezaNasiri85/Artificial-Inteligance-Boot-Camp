"""Student TODO: deterministic UTF-8 byte tokenizer."""

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260
SPECIAL_TOKEN_NAMES = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

def encode_text(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    # TODO(HW01): Implement this function.
    raise NotImplementedError("TODO(HW01)")

def decode_tokens(token_ids: list[int]) -> str:
    # TODO(HW01): Implement this function.
    raise NotImplementedError("TODO(HW01)")

def token_display(token_id: int) -> str:
    if token_id in SPECIAL_TOKEN_NAMES:
        return SPECIAL_TOKEN_NAMES[token_id]
    if BYTE_OFFSET <= token_id < VOCAB_SIZE:
        value = bytes([token_id - BYTE_OFFSET]).decode("utf-8", errors="replace")
        return {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(value, value)
    return f"<INVALID:{token_id}>"
