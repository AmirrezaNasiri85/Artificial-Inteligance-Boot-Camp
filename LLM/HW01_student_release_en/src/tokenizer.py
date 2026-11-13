"""Student TODO: deterministic UTF-8 byte tokenizer."""

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260
SPECIAL_TOKEN_NAMES = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

# raise NotImplementedError("TODO(HW01)")


def encode_text(text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    # TODO(HW01): Implement this function.
    ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]
    if add_bos:
        ids = [BOS_TOKEN_ID] + ids
    if add_eos:
        ids = ids + [EOS_TOKEN_ID]
        
    return ids

def decode_tokens(token_ids: list[int]) -> str:
    # TODO(HW01): Implement this function.
    # In this function, we remove the codes which do not carry any important concepts.
    # then the other codes are collected and gathered to be decoded.
    raw = bytearray()
    for tokenid in token_ids:
        if not BYTE_OFFSET <= tokenid < VOCAB_SIZE:
            raise ValueError(f"Invalid token id: {tokenid}")
        raw.append(tokenid - BYTE_OFFSET)
    return raw.decode("utf-8", errors="replace")

def token_display(token_id: int) -> str:
    if token_id in SPECIAL_TOKEN_NAMES:
        return SPECIAL_TOKEN_NAMES[token_id]
    if BYTE_OFFSET <= token_id < VOCAB_SIZE:
        value = bytes([token_id - BYTE_OFFSET]).decode("utf-8", errors="replace")
        return {"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(value, value)
    return f"<INVALID:{token_id}>"

def tokenizer_test(text: str, add_bos: bool = False, add_eos: bool = False):
    coded_text = encode_text(text, add_bos, add_eos)
    print(coded_text)

    decoded_text = decode_tokens(coded_text)
    print(decoded_text)
    for obj in coded_text:
        print(token_display(obj))


if __name__ == "__main__":
    tokenizer_test("hello world")