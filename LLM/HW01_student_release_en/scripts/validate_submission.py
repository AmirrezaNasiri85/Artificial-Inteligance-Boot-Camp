#!/usr/bin/env python3
"""Strictly validate a completed-student submission ZIP and its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from package_submission import (
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_PARTS,
    MAX_FILE_BYTES,
    MAX_ZIP_BYTES,
)

REQUIRED_SUBMISSION_FILES = {
    "HW01.ipynb",
    "README.md",
    "report.md",
    "src/__init__.py",
    "src/tokenizer.py",
    "src/data.py",
    "src/masking.py",
    "src/training.py",
    "src/generation.py",
    "src/analysis.py",
    "tests/test_tokenizer.py",
    "tests/test_data.py",
    "tests/test_masking.py",
    "tests/test_training.py",
    "tests/test_generation.py",
    "tests/test_analysis.py",
    "tests/test_packaging.py",
    "scripts/package_submission.py",
    "scripts/validate_submission.py",
    "figures/causal_mask.png",
    "figures/training_loss.png",
    "figures/error_propagation.png",
    "results/tokenizer_examples.json",
    "results/token_shift_table.csv",
    "results/training_step_metrics.json",
    "results/training_metrics.json",
    "results/generation_trace.csv",
    "results/error_propagation.csv",
    "results/training_vs_inference.csv",
    "results/eval_vs_inference_mode.csv",
}
MENTOR_ONLY_NAMES = {
    "HW01_solution.ipynb",
    "mentor_notes.md",
    "common_mistakes.md",
    "reference_report.md",
    "reproducibility_report.md",
    "instructor",
}
FORBIDDEN_NOTEBOOK_TEXT = {
    "scripts.notebook_workflow",
    "HW01_solution",
    "mentor_notes",
}


def _normalized_relative(name: str, top_level: str) -> str:
    if "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive path: {name}")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name}")
    if not path.parts or path.parts[0] != top_level or len(path.parts) < 2:
        raise ValueError("archive must contain exactly one top-level submission directory")
    return PurePosixPath(*path.parts[1:]).as_posix()


def _manifest_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("assignment") != "QBC12-AI-HW01":
        raise ValueError("invalid assignment identifier in manifest")
    if not re.fullmatch(r"[0-9]{4,12}", str(manifest.get("student_id", ""))):
        raise ValueError("invalid student ID in manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest files must be a nonempty list")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError("manifest row has an invalid schema")
        path = str(row["path"])
        normalized = PurePosixPath(path).as_posix()
        if path != normalized or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ValueError(f"unsafe manifest path: {path}")
        if path in mapped or path.casefold() in {item.casefold() for item in mapped}:
            raise ValueError(f"duplicate manifest path: {path}")
        if not isinstance(row["size_bytes"], int) or row["size_bytes"] < 0:
            raise ValueError(f"invalid manifest size: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise ValueError(f"invalid manifest SHA-256: {path}")
        mapped[path] = row
    return mapped


def validate_submission_zip(zip_path: str | Path) -> dict[str, Any]:
    """Validate paths, forbidden content, manifest integrity, and required files."""
    path = Path(zip_path)
    if not path.is_file():
        raise ValueError(f"submission ZIP does not exist: {path}")
    if path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("submission exceeds 20 MiB")

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise ValueError("submission is not a valid ZIP") from error
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise ValueError("submission archive is empty")
        raw_names = [info.filename for info in infos]
        top_levels = {PurePosixPath(name).parts[0] for name in raw_names if name}
        if len(top_levels) != 1:
            raise ValueError("archive must contain one top-level directory")
        top_level = next(iter(top_levels))
        normalized_to_info: dict[str, zipfile.ZipInfo] = {}
        seen_casefold: set[str] = set()
        for info in infos:
            relative = _normalized_relative(info.filename, top_level)
            folded = relative.casefold()
            if relative in normalized_to_info or folded in seen_casefold:
                raise ValueError(f"duplicate normalized archive path: {relative}")
            normalized_to_info[relative] = info
            seen_casefold.add(folded)
            relative_path = PurePosixPath(relative)
            if any(part in FORBIDDEN_PARTS for part in relative_path.parts):
                raise ValueError(f"forbidden archive entry: {relative}")
            if relative_path.suffix.lower() in FORBIDDEN_EXTENSIONS:
                raise ValueError(f"forbidden extension: {relative}")
            if any(part in MENTOR_ONLY_NAMES for part in relative_path.parts):
                raise ValueError(f"mentor-only content: {relative}")
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"suspiciously large file: {relative}")

        manifest_name = "submission_manifest.json"
        if manifest_name not in normalized_to_info:
            raise ValueError("missing submission_manifest.json")
        manifest = json.loads(archive.read(normalized_to_info[manifest_name]).decode("utf-8"))
        declared = _manifest_map(manifest)
        actual = set(normalized_to_info) - {manifest_name}
        if set(declared) != actual:
            missing = sorted(set(declared) - actual)
            undeclared = sorted(actual - set(declared))
            raise ValueError(f"manifest mismatch; missing={missing}, undeclared={undeclared}")

        missing_required = sorted(REQUIRED_SUBMISSION_FILES - actual)
        if missing_required:
            raise ValueError("missing required submission files: " + ", ".join(missing_required))
        for relative, row in declared.items():
            payload = archive.read(normalized_to_info[relative])
            if len(payload) != row["size_bytes"]:
                raise ValueError(f"manifest size mismatch: {relative}")
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"manifest SHA-256 mismatch: {relative}")

        notebook_text = archive.read(normalized_to_info["HW01.ipynb"]).decode("utf-8")
        json.loads(notebook_text)
        for forbidden in FORBIDDEN_NOTEBOOK_TEXT:
            if forbidden in notebook_text:
                raise ValueError(f"forbidden notebook dependency: {forbidden}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_count": len(infos),
        "student_id": manifest["student_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    arguments = parser.parse_args()
    result = validate_submission_zip(arguments.zip_path)
    print(f"Validated {result['path']}: {result['file_count']} files, {result['size_bytes']} bytes")


if __name__ == "__main__":
    main()
