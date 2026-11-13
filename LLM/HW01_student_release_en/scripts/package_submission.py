#!/usr/bin/env python3
"""Validate and create a deterministic completed-student submission ZIP."""

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
FORBIDDEN_PARTS = {
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "dist",
    "build",
    "assets",
}
REQUIRED_ITEMS = {
    "README.md",
    "report.md",
    "src",
    "tests",
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
    "scripts/package_submission.py",
    "scripts/validate_submission.py",
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


def write_deterministic_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def package_submission(project_root: Path, student_id: str, output: Path) -> Path:
    student_id = validate_student_id(student_id)
    notebook = project_root / "HW01.ipynb"
    if not notebook.exists():
        notebook = project_root / "HW01_student.ipynb"
    required = set(REQUIRED_ITEMS)
    if not notebook.exists():
        required.add("HW01.ipynb")
    missing = [item for item in sorted(required) if not (project_root / item).exists()]
    if missing:
        raise ValueError("missing required files: " + ", ".join(missing))

    package_name = f"HW01_{student_id}"
    with tempfile.TemporaryDirectory(prefix="hw01-package-") as temporary:
        staging = Path(temporary) / package_name
        staging.mkdir()
        included_roots = [
            "README.md",
            "report.md",
            "src",
            "tests",
            "figures",
            "results",
            "scripts",
        ]
        for name in included_roots:
            source = project_root / name
            destination = staging / name
            if source.is_dir():
                ignored_names = list(FORBIDDEN_PARTS)
                shutil.copytree(
                    source,
                    destination,
                    ignore=shutil.ignore_patterns(*ignored_names),
                )
            else:
                shutil.copy2(source, destination)
        shutil.copy2(notebook, staging / "HW01.ipynb")

        manifest_files = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            validate_candidate_file(path, staging)
            manifest_files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "assignment": "QBC12-AI-HW01",
            "student_id": student_id,
            "files": manifest_files,
        }
        manifest_path = staging / "submission_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_deterministic_zip(staging.parent, output)

    size = output.stat().st_size
    if size >= MAX_ZIP_BYTES:
        output.unlink()
        raise ValueError(f"ZIP is {size} bytes; limit is {MAX_ZIP_BYTES} bytes")
    print(f"Created: {output.resolve()}")
    print(f"Size: {size} bytes")
    print(f"SHA-256: {sha256_file(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    destination = arguments.output or arguments.project_root / "dist" / "HW01_submission.zip"
    package_submission(
        arguments.project_root.resolve(), arguments.student_id, destination.resolve()
    )


if __name__ == "__main__":
    main()
