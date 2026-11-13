#!/usr/bin/env python3
"""Build a deterministic, asset-free completed HW02 submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".safetensors", ".gguf", ".zip"}
FORBIDDEN_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "dist", "build", "assets"}
REQUIRED_ITEMS = {
    "README.md", "report.md", "src", "tests", "scripts/package_submission.py", "scripts/validate_submission.py",
    "figures/next_token_distribution.png", "figures/temperature_distributions.png", "figures/second_best_probability_decay.png", "figures/entropy_by_step.png", "figures/diversity_vs_probability.png",
    "results/environment.json", "results/top_tokens.csv", "results/softmax_stability.json", "results/temperature_metrics.csv", "results/deterministic_choices.json", "results/sampling_candidate_sets.csv", "results/generation_trace.csv", "results/second_best_comparison.csv", "results/toy_sequence_search.json", "results/generations.csv", "results/decoding_metrics.csv", "results/adaptive_decoder_results.csv", "results/decoding_strategy_summary.csv",
}


def validate_student_id(student_id: str) -> str:
    if not re.fullmatch(r"[0-9]{4,12}", student_id):
        raise ValueError("student ID must contain 4 to 12 ASCII digits")
    return student_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_file(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        raise ValueError(f"forbidden directory or file: {relative}")
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"forbidden extension: {relative}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"suspiciously large file: {relative}")


def write_deterministic_zip(source_parent: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source_parent.rglob("*") if item.is_file()):
            relative = path.relative_to(source_parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def package_submission(project_root: Path, student_id: str, output: Path) -> Path:
    student_id = validate_student_id(student_id)
    notebook = project_root / "HW02.ipynb"
    if not notebook.is_file():
        notebook = project_root / "HW02_student.ipynb"
    missing = sorted(item for item in REQUIRED_ITEMS if not (project_root / item).exists())
    if not notebook.is_file():
        missing.append("HW02.ipynb")
    if missing:
        raise ValueError("missing required files: " + ", ".join(missing))
    from src.validation import validate_artifacts

    validate_artifacts(project_root)
    package_name = f"HW02_{student_id}"
    with tempfile.TemporaryDirectory(prefix="hw02-package-") as temporary:
        staging = Path(temporary) / package_name
        staging.mkdir()
        for name in ["README.md", "report.md", "src", "tests", "figures", "results", "scripts", "pyproject.toml", "uv.lock", ".python-version"]:
            source = project_root / name
            if not source.exists():
                continue
            destination = staging / name
            if source.is_dir():
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*FORBIDDEN_PARTS))
            else:
                shutil.copy2(source, destination)
        shutil.copy2(notebook, staging / "HW02.ipynb")
        rows = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            validate_candidate_file(path, staging)
            rows.append({"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {"assignment": "QBC12-AI-HW02", "student_id": student_id, "files": rows}
        (staging / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_deterministic_zip(staging.parent, output)
    if output.stat().st_size >= MAX_ZIP_BYTES:
        output.unlink()
        raise ValueError("submission ZIP exceeds 20 MiB")
    from validate_submission import validate_submission_zip

    validate_submission_zip(output)
    print(f"Created: {output.resolve()}")
    print(f"Size: {output.stat().st_size} bytes")
    print(f"SHA-256: {sha256_file(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project_root / f"HW02_{args.student_id}.zip"
    package_submission(args.project_root.resolve(), args.student_id, output.resolve())


if __name__ == "__main__":
    main()

