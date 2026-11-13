"""Checkpoint and course-asset loading helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .config import MiniGPTConfig, count_parameters
from .model import MiniGPT

EXPECTED_CHECKPOINT_SHA256 = "7151108ca27ebfed3077aed89b4d093696bf09a715152fce99d3bf64167fea32"
EXPECTED_PARAMETER_COUNT = 644_736


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(asset_dir: str | Path) -> dict[str, object]:
    """Verify all fixed assets and return parsed metadata."""
    root = Path(asset_dir)
    required = [
        "minigpt_pretrained.pt",
        "tokenizer_config.json",
        "benchmark_prompts.json",
        "probability_tree.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing course assets: " + ", ".join(missing))
    digest = sha256_file(root / "minigpt_pretrained.pt")
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 mismatch")
    tokenizer = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    prompts = json.loads((root / "benchmark_prompts.json").read_text(encoding="utf-8"))
    tree = json.loads((root / "probability_tree.json").read_text(encoding="utf-8"))
    if tokenizer.get("vocab_size") != 260 or len(prompts) < 6:
        raise ValueError("invalid course asset metadata")
    return {"checkpoint_sha256": digest, "tokenizer": tokenizer, "prompts": prompts, "tree": tree}


def load_pretrained_model(asset_dir: str | Path) -> MiniGPT:
    """Load the verified checkpoint on CPU using weights_only=True."""
    root = Path(asset_dir)
    verify_assets(root)
    model = MiniGPT(MiniGPTConfig()).cpu()
    payload = torch.load(root / "minigpt_pretrained.pt", map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    if count_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("unexpected parameter count")
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("model parameters must be on CPU")
    return model
