#!/usr/bin/env python3
"""Deterministic asset-free HW03 submission packager."""

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
    "instructor",
}
ARTIFACTS = {
    "results/environment.json",
    "results/uncached_trace.csv",
    "results/kv_cache_shapes.csv",
    "results/prefill_decode_trace.csv",
    "results/cache_equivalence.json",
    "results/cached_generation_trace.csv",
    "results/inference_benchmark.csv",
    "results/context_rebuild_trace.csv",
    "results/quantization_summary.json",
    "results/quantization_tensor_metrics.csv",
    "results/quantization_quality.csv",
    "results/capacity_plan.csv",
    "results/inference_strategy_summary.csv",
    "figures/kv_cache_growth.png",
    "figures/processed_token_positions.png",
    "figures/prefill_decode_latency.png",
    "figures/quantization_error.png",
    "figures/memory_capacity.png",
}
REQUIRED_ITEMS = {
    "README.md",
    "report.md",
    "src",
    "tests",
    "scripts/package_submission.py",
    "scripts/validate_submission.py",
} | ARTIFACTS


def validate_student_id(x):
    if not re.fullmatch(r"[0-9]{4,12}", x):
        raise ValueError("student ID must contain 4 to 12 ASCII digits")
    return x


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def write_zip(parent, dest):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(x for x in parent.rglob("*") if x.is_file()):
            i = zipfile.ZipInfo(p.relative_to(parent).as_posix(), (2024, 1, 1, 0, 0, 0))
            i.compress_type = zipfile.ZIP_DEFLATED
            i.external_attr = 0o100644 << 16
            z.writestr(i, p.read_bytes(), compresslevel=9)


def package_submission(project_root: Path, student_id: str, output: Path):
    student_id = validate_student_id(student_id)
    notebook = project_root / "HW03.ipynb"
    if not notebook.is_file():
        notebook = project_root / "HW03_student.ipynb"
    missing = sorted(x for x in REQUIRED_ITEMS if not (project_root / x).exists())
    if not notebook.is_file():
        missing.append("HW03.ipynb")
    if missing:
        raise ValueError("missing required files: " + ", ".join(missing))
    from src.validation import validate_artifacts

    validate_artifacts(project_root)
    with tempfile.TemporaryDirectory(prefix="hw03-package-") as t:
        stage = Path(t) / f"HW03_{student_id}"
        stage.mkdir()
        for name in [
            "README.md",
            "report.md",
            "rubric.md",
            "src",
            "tests",
            "figures",
            "results",
            "scripts",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
        ]:
            s = project_root / name
            if not s.exists():
                continue
            d = stage / name
            if s.is_dir():
                shutil.copytree(s, d, ignore=shutil.ignore_patterns(*FORBIDDEN_PARTS))
            else:
                shutil.copy2(s, d)
        shutil.copy2(notebook, stage / "HW03.ipynb")
        rows = []
        for p in sorted(x for x in stage.rglob("*") if x.is_file()):
            rel = p.relative_to(stage)
            if (
                any(x in FORBIDDEN_PARTS for x in rel.parts)
                or p.suffix.lower() in FORBIDDEN_EXTENSIONS
                or p.stat().st_size > MAX_FILE_BYTES
            ):
                raise ValueError(f"forbidden/suspicious file: {rel}")
            rows.append(
                {"path": rel.as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
            )
        (stage / "submission_manifest.json").write_text(
            json.dumps(
                {"assignment": "QBC12-AI-HW03", "student_id": student_id, "files": rows}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_zip(stage.parent, output)
    if output.stat().st_size >= MAX_ZIP_BYTES:
        output.unlink()
        raise ValueError("submission ZIP exceeds 20 MiB")
    from scripts.validate_submission import validate_submission_zip

    validate_submission_zip(output)
    print(f"Created: {output.resolve()}")
    print(f"Size: {output.stat().st_size} bytes")
    print(f"SHA-256: {sha256_file(output)}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-id", required=True)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.project_root.resolve()
    package_submission(
        root, a.student_id, (a.output or root / f"HW03_{a.student_id}.zip").resolve()
    )


if __name__ == "__main__":
    main()
