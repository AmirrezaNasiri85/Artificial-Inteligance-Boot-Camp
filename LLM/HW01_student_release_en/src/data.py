"""Student TODO: causal language-model data helpers."""

from pathlib import Path

import torch

from .tokenizer import encode_text


def create_lm_example(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 1 or token_ids.numel() < 2:
        raise ValueError("token_ids must be 1-D with at least two tokens")
    inputs = token_ids[:-1]
    labels = token_ids[1:]
    return inputs, labels

def load_corpus_tokens(path: str | Path) -> torch.Tensor:
    return torch.tensor(encode_text(Path(path).read_text(encoding="utf-8"), add_eos=True), dtype=torch.long)

def sample_batch(tokens: torch.Tensor, batch_size: int, sequence_length: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1 or tokens.numel() <= sequence_length:
        raise ValueError("corpus must contain more tokens than sequence_length")
    starts = torch.randint(0, tokens.numel() - sequence_length, (batch_size,), generator=generator)
    chunks = torch.stack([tokens[start:start + sequence_length + 1] for start in starts])
    return chunks[:, :-1], chunks[:, 1:]

def split_tokens(tokens: torch.Tensor, validation_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    split = int(tokens.numel() * (1.0 - validation_fraction))
    return tokens[:split], tokens[split:]
