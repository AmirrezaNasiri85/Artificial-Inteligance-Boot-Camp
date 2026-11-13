from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.sequence_search import (
    exhaustive_best_sequence,
    greedy_tree_sequence,
    sequence_log_probability,
)

ASSET = Path(__file__).resolve().parents[1] / "assets" / "probability_tree.json"


def test_sequence_log_probability_and_invalid_values() -> None:
    assert sequence_log_probability([0.5, 0.25]) == pytest.approx(math.log(0.125))
    assert sequence_log_probability([]) == 0
    for values in [[0], [-0.1], [1.1]]:
        with pytest.raises(ValueError):
            sequence_log_probability(values)


def test_greedy_and_exhaustive_paths_are_consistent() -> None:
    tree = json.loads(ASSET.read_text(encoding="utf-8"))
    greedy = greedy_tree_sequence(tree, tree["start_state"], tree["depth"])
    optimal = exhaustive_best_sequence(tree, tree["start_state"], tree["depth"])
    assert greedy["tokens"] == tree["expected_greedy_tokens"]
    assert optimal["tokens"] == tree["expected_optimal_tokens"]
    assert optimal["sequence_probability"] > greedy["sequence_probability"]
    for result in [greedy, optimal]:
        assert result["sequence_probability"] == pytest.approx(math.prod(result["transition_probabilities"]))
        assert result["sequence_log_probability"] == pytest.approx(math.log(result["sequence_probability"]))


def test_tree_depth_validation() -> None:
    with pytest.raises(ValueError):
        greedy_tree_sequence({}, "root", 0)
