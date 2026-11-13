#!/usr/bin/env python3
"""Static validator for completed HW03 submission archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from scripts.package_submission import (
    ARTIFACTS,
    FORBIDDEN_EXTENSIONS,
    FORBIDDEN_PARTS,
    MAX_FILE_BYTES,
    MAX_ZIP_BYTES,
)

BASE = {
    "HW03.ipynb",
    "README.md",
    "report.md",
    "submission_manifest.json",
    "src/accounting.py",
    "src/cache.py",
    "src/config.py",
    "src/generation.py",
    "src/model.py",
    "src/model_io.py",
    "src/quantization.py",
    "src/types.py",
    "src/validation.py",
    "tests/test_contracts.py",
    "tests/test_accounting.py",
    "tests/test_cache.py",
    "tests/test_model_cache.py",
    "tests/test_generation.py",
    "tests/test_quantization.py",
    "tests/test_artifacts.py",
    "tests/test_packaging.py",
    "scripts/package_submission.py",
    "scripts/validate_submission.py",
}
REQUIRED = BASE | ARTIFACTS
COURSE = {
    "minigpt_pretrained.pt",
    "tokenizer_config.json",
    "benchmark_prompts.json",
    "probability_tree.json",
    "minigpt_int8.pt",
}
FORBIDDEN_TEXT = {
    "scripts.notebook_workflow",
    "model.generate(",
    "HW03_solution",
    "mentor_notes",
    "mentor/src",
    "reference_report",
    "bitsandbytes",
    "torchao",
    "torch.ao",
}


def validate_submission_zip(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size >= MAX_ZIP_BYTES:
        raise ValueError("submission ZIP missing/oversized")
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        tops = {PurePosixPath(i.filename).parts[0] for i in infos}
        if len(tops) != 1:
            raise ValueError("archive must contain one top-level directory")
        top = next(iter(tops))
        entries = {}
        folded = set()
        for i in infos:
            if "\\" in i.filename:
                raise ValueError("unsafe path")
            p = PurePosixPath(i.filename)
            if p.is_absolute() or ".." in p.parts or len(p.parts) < 2 or p.parts[0] != top:
                raise ValueError("unsafe path")
            rel = PurePosixPath(*p.parts[1:]).as_posix()
            key = rel.casefold()
            if key in folded:
                raise ValueError("duplicate normalized path")
            folded.add(key)
            entries[rel] = i
            parts = PurePosixPath(rel).parts
            if (
                any(x in FORBIDDEN_PARTS or x == "instructor" for x in parts)
                or PurePosixPath(rel).suffix.lower() in FORBIDDEN_EXTENSIONS
                or PurePosixPath(rel).name in COURSE
            ):
                raise ValueError(f"forbidden content: {rel}")
            if i.file_size > MAX_FILE_BYTES:
                raise ValueError(f"suspiciously large file: {rel}")
        missing = sorted(REQUIRED - set(entries))
        if missing:
            raise ValueError("missing required files: " + ", ".join(missing))
        manifest = json.loads(z.read(entries["submission_manifest.json"]).decode())
        if manifest.get("assignment") != "QBC12-AI-HW03" or not re.fullmatch(
            r"[0-9]{4,12}", str(manifest.get("student_id", ""))
        ):
            raise ValueError("invalid manifest header")
        rows = manifest.get("files")
        declared = {}
        if not isinstance(rows, list) or not rows:
            raise ValueError("invalid manifest files")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
                raise ValueError("invalid manifest row")
            name = str(row["path"])
            if name.casefold() in {x.casefold() for x in declared}:
                raise ValueError("duplicate manifest path")
            declared[name] = row
        actual = set(entries) - {"submission_manifest.json"}
        if set(declared) != actual:
            raise ValueError("manifest declared/actual mismatch")
        for name, row in declared.items():
            data = z.read(entries[name])
            if len(data) != row["size_bytes"]:
                raise ValueError(f"manifest size mismatch: {name}")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"manifest hash mismatch: {name}")
            if (
                name == "HW03.ipynb"
                or name.startswith("src/")
                or name in {"README.md", "report.md"}
            ):
                text = data.decode("utf-8")
                if any(m in text for m in FORBIDDEN_TEXT):
                    raise ValueError(f"prohibited source content: {name}")
        json.loads(z.read(entries["HW03.ipynb"]).decode())
        for name in [x for x in ARTIFACTS if x.endswith(".json")]:
            json.loads(z.read(entries[name]).decode())
        for name in [x for x in ARTIFACTS if x.endswith(".csv")]:
            if not list(csv.DictReader(io.StringIO(z.read(entries[name]).decode()))):
                raise ValueError(f"empty CSV: {name}")
        for name in [x for x in ARTIFACTS if x.endswith(".png")]:
            data = z.read(entries[name])
            if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) >= 500 * 1024:
                raise ValueError(f"invalid PNG: {name}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_count": len(infos),
        "student_id": manifest["student_id"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip_path", type=Path)
    a = p.parse_args()
    print(json.dumps(validate_submission_zip(a.zip_path), indent=2))


if __name__ == "__main__":
    main()
