from __future__ import annotations

import torch

from src.decoding import select_entropy_adaptive


def test_adaptive_takes_both_branches() -> None:
    generator = torch.Generator().manual_seed(4)
    low = select_entropy_adaptive(torch.tensor([12.0, 0.0, 0.0]), 0.5, 0.9, generator)
    high = select_entropy_adaptive(torch.zeros(5), 0.5, 0.9, generator)
    assert low.strategy == "entropy_adaptive_greedy"
    assert low.candidate_count == 1
    assert high.strategy == "entropy_adaptive_top_p"
    assert high.candidate_count == 5

