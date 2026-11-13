#!/usr/bin/env python3
"""Validate an HW02 submission archive, manifest, artifacts, and source isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from package_submission import FORBIDDEN_EXTENSIONS, FORBIDDEN_PARTS, MAX_FILE_BYTES, MAX_ZIP_BYTES

REQUIRED_FILES = {
    "HW02.ipynb", "README.md", "report.md", "submission_manifest.json",
    "src/probability.py", "src/decoding.py", "src/generation.py", "src/sequence_search.py", "src/validation.py",
    "tests/test_probability.py", "tests/test_decoding.py", "tests/test_generation.py", "tests/test_sequence_search.py", "tests/test_adaptive.py", "tests/test_artifacts.py", "tests/test_packaging.py",
    "figures/next_token_distribution.png", "figures/temperature_distributions.png", "figures/second_best_probability_decay.png", "figures/entropy_by_step.png", "figures/diversity_vs_probability.png",
    "results/environment.json", "results/top_tokens.csv", "results/softmax_stability.json", "results/temperature_metrics.csv", "results/deterministic_choices.json", "results/sampling_candidate_sets.csv", "results/generation_trace.csv", "results/second_best_comparison.csv", "results/toy_sequence_search.json", "results/generations.csv", "results/decoding_metrics.csv", "results/adaptive_decoder_results.csv", "results/decoding_strategy_summary.csv",
}
MENTOR_NAMES = {"HW02_solution.ipynb", "mentor_notes.md", "common_mistakes.md", "reference_report.md", "reproducibility_report.md", "instructor"}
FORBIDDEN_TEXT = {"scripts.notebook_workflow", "model.generate(", "HW02_solution", "mentor_notes", "mentor/src", "reference_report"}
COURSE_ASSET_NAMES = {"minigpt_pretrained.pt", "tokenizer_config.json", "benchmark_prompts.json", "probability_tree.json"}


def _relative(name: str, top: str) -> str:
    if "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive path: {name}")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts) or len(path.parts) < 2 or path.parts[0] != top:
        raise ValueError(f"unsafe archive path: {name}")
    return PurePosixPath(*path.parts[1:]).as_posix()


def validate_submission_zip(zip_path: str | Path) -> dict[str, object]:
    path = Path(zip_path)
    if not path.is_file() or path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("submission ZIP is missing or exceeds 20 MiB")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise ValueError("invalid ZIP") from error
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        tops = {PurePosixPath(info.filename).parts[0] for info in infos}
        if len(tops) != 1:
            raise ValueError("archive must contain exactly one top-level directory")
        top = next(iter(tops))
        entries: dict[str, zipfile.ZipInfo] = {}
        folded: set[str] = set()
        for info in infos:
            relative = _relative(info.filename, top)
            if relative.casefold() in folded:
                raise ValueError(f"duplicate normalized path: {relative}")
            folded.add(relative.casefold())
            entries[relative] = info
            parts = PurePosixPath(relative).parts
            if any(part in FORBIDDEN_PARTS or part in MENTOR_NAMES for part in parts):
                raise ValueError(f"forbidden content: {relative}")
            if PurePosixPath(relative).suffix.lower() in FORBIDDEN_EXTENSIONS or PurePosixPath(relative).name in COURSE_ASSET_NAMES:
                raise ValueError(f"forbidden course asset or extension: {relative}")
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"suspiciously large file: {relative}")
        missing = sorted(REQUIRED_FILES - set(entries))
        if missing:
            raise ValueError("missing required files: " + ", ".join(missing))
        manifest = json.loads(archive.read(entries["submission_manifest.json"]).decode("utf-8"))
        if manifest.get("assignment") != "QBC12-AI-HW02" or not re.fullmatch(r"[0-9]{4,12}", str(manifest.get("student_id", ""))):
            raise ValueError("invalid manifest header")
        rows = manifest.get("files")
        if not isinstance(rows, list) or not rows:
            raise ValueError("manifest files must be a nonempty list")
        declared: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
                raise ValueError("invalid manifest row")
            name = str(row["path"])
            if name in declared or name.casefold() in {item.casefold() for item in declared} or PurePosixPath(name).as_posix() != name or ".." in PurePosixPath(name).parts:
                raise ValueError("unsafe or duplicate manifest path")
            declared[name] = row
        actual = set(entries) - {"submission_manifest.json"}
        if set(declared) != actual:
            raise ValueError("manifest declared/actual file mismatch")
        for name, row in declared.items():
            payload = archive.read(entries[name])
            if len(payload) != row["size_bytes"]:
                raise ValueError(f"manifest size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"manifest hash mismatch: {name}")
            if name == "HW02.ipynb" or name.startswith("src/") or name in {"README.md", "report.md"}:
                text = payload.decode("utf-8")
                if any(marker in text for marker in FORBIDDEN_TEXT):
                    raise ValueError(f"mentor-only or prohibited source content: {name}")
        json.loads(archive.read(entries["HW02.ipynb"]).decode("utf-8"))
        for name in [item for item in actual if item.startswith("results/") and item.endswith(".json")]:
            json.loads(archive.read(entries[name]).decode("utf-8"))
        for name in [item for item in actual if item.startswith("results/") and item.endswith(".csv")]:
            rows_csv = list(csv.DictReader(io.StringIO(archive.read(entries[name]).decode("utf-8"))))
            if not rows_csv:
                raise ValueError(f"empty CSV: {name}")
        for name in [item for item in actual if item.startswith("figures/") and item.endswith(".png")]:
            payload = archive.read(entries[name])
            if payload[:8] != b"\x89PNG\r\n\x1a\n" or len(payload) >= 500 * 1024:
                raise ValueError(f"invalid or oversized PNG: {name}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "file_count": len(infos), "student_id": manifest["student_id"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_submission_zip(args.zip_path), indent=2))


if __name__ == "__main__":
    main()
