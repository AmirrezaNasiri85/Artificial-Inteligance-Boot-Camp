"""Student TODOs for tiny-tree sequence search."""

from __future__ import annotations

import math


def sequence_log_probability(transition_probabilities: list[float]) -> float:
    total = 0.0
    for p in transition_probabilities:
        p = float(p)
        if not math.isfinite(p) or p <= 0.0 or p > 1.0:
            raise ValueError(f"invalid transition probability: {p}")
        total += math.log(p) 
    return total


def greedy_tree_sequence(tree: dict, start_state: str, depth: int) -> dict:
    table = tree
    for key in ("transitions", "nodes", "states"):
        if isinstance(tree.get(key), dict):
            table = tree[key]
            break

    state, tokens, probs = start_state, [], []
    for _ in range(depth):
        node = table.get(state)
        if not node:
            break

        if isinstance(node, dict):
            options = [(t, float(p), t) for t, p in node.items()]
        else:
            options = [
                (o["token"], float(o["probability"]), o.get("next_state", o["token"]))
                for o in node
            ]
        sequence_log_probability([p for _, p, _ in options])  
        token, p, state = max(options, key=lambda o: o[1]) 
        tokens.append(token)
        probs.append(p)

    log_p = sequence_log_probability(probs)
    return {
        "tokens": tokens,
        "transition_probabilities": probs,
        "log_probability": log_p,
        "sequence_probability": math.exp(log_p),
    }


def exhaustive_best_sequence(tree: dict, start_state: str, depth: int) -> dict:
    table = tree
    for key in ("transitions", "nodes", "states"):
        if isinstance(tree.get(key), dict):
            table = tree[key]
            break

    frontier = [(start_state, [], [])]
    for _ in range(depth):
        expanded = []
        for state, tokens, probs in frontier:
            node = table.get(state)
            if not node:
                expanded.append((state, tokens, probs)) 
                continue
            if isinstance(node, dict):
                options = [(t, float(p), t) for t, p in node.items()]
            else:
                options = [
                    (o["token"], float(o["probability"]), o.get("next_state", o["token"]))
                    for o in node
                ]
            for token, p, nxt in options:
                expanded.append((nxt, tokens + [token], probs + [p]))
        frontier = expanded

    best = None
    for _, tokens, probs in frontier:
        log_p = sequence_log_probability(probs)
        if best is None or log_p > best["log_probability"]:
            best = {
                "tokens": tokens,
                "transition_probabilities": probs,
                "log_probability": log_p,
                "sequence_probability": math.exp(log_p),
            }
    return best
