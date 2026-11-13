import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "scripts").is_dir()
)
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_submission.py"
PACKAGE_SPEC = importlib.util.spec_from_file_location("package_submission", PACKAGE_SCRIPT)
PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
assert PACKAGE_SPEC.loader is not None
sys.modules["package_submission"] = PACKAGE
PACKAGE_SPEC.loader.exec_module(PACKAGE)

VALIDATOR_SCRIPT = PROJECT_ROOT / "scripts" / "validate_submission.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validate_submission", VALIDATOR_SCRIPT)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def _make_project(root: Path) -> None:
    for relative in VALIDATOR.REQUIRED_SUBMISSION_FILES:
        if relative == "HW01.ipynb":
            path = root / "HW01_student.ipynb"
            payload = json.dumps(
                {
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ).encode()
        else:
            path = root / relative
            payload = b"test evidence"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()
        }


def _write_entries(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def _update_notebook_manifest(entries: dict[str, bytes], top: str) -> None:
    manifest_name = f"{top}/submission_manifest.json"
    notebook_name = f"{top}/HW01.ipynb"
    manifest = json.loads(entries[manifest_name])
    row = next(item for item in manifest["files"] if item["path"] == "HW01.ipynb")
    row["size_bytes"] = len(entries[notebook_name])
    row["sha256"] = hashlib.sha256(entries[notebook_name]).hexdigest()
    entries[manifest_name] = json.dumps(manifest).encode()


@pytest.fixture
def valid_zip(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    _make_project(project)
    output = tmp_path / "submission.zip"
    PACKAGE.package_submission(project, "123456", output)
    return output


def test_student_id_validation():
    assert PACKAGE.validate_student_id("123456") == "123456"
    with pytest.raises(ValueError):
        PACKAGE.validate_student_id("12/34")


def test_forbidden_paths_and_extensions_are_detected(tmp_path):
    forbidden = tmp_path / "model.pt"
    forbidden.write_bytes(b"weights")
    with pytest.raises(ValueError, match="forbidden"):
        PACKAGE.validate_candidate_file(forbidden, tmp_path)


def test_deterministic_zip_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("same", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    PACKAGE.write_deterministic_zip(source, first)
    PACKAGE.write_deterministic_zip(source, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt"]


def test_valid_manifest_is_fully_verified(valid_zip):
    result = VALIDATOR.validate_submission_zip(valid_zip)
    assert result["student_id"] == "123456"
    assert result["file_count"] > 20


@pytest.mark.parametrize(
    "corruption, expected",
    [
        ("hash", "SHA-256 mismatch"),
        ("size", "size mismatch"),
        ("missing", "manifest mismatch"),
        ("extra", "manifest mismatch"),
        ("model", "forbidden"),
        ("notebook_import", "forbidden notebook dependency"),
    ],
)
def test_corrupted_packages_are_rejected(valid_zip, tmp_path, corruption, expected):
    entries = _entries(valid_zip)
    top = next(iter(entries)).split("/", maxsplit=1)[0]
    manifest_name = f"{top}/submission_manifest.json"
    manifest = json.loads(entries[manifest_name])
    if corruption == "hash":
        manifest["files"][0]["sha256"] = "0" * 64
        entries[manifest_name] = json.dumps(manifest).encode()
    elif corruption == "size":
        manifest["files"][0]["size_bytes"] += 1
        entries[manifest_name] = json.dumps(manifest).encode()
    elif corruption == "missing":
        entries.pop(f"{top}/results/training_metrics.json")
    elif corruption == "extra":
        entries[f"{top}/undeclared.txt"] = b"extra"
    elif corruption == "model":
        entries[f"{top}/model.pt"] = b"weights"
    elif corruption == "notebook_import":
        name = f"{top}/HW01.ipynb"
        notebook = json.loads(entries[name])
        notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from scripts.notebook_workflow import *"],
            }
        )
        entries[name] = json.dumps(notebook).encode()
        _update_notebook_manifest(entries, top)
    broken = tmp_path / f"{corruption}.zip"
    _write_entries(broken, entries)
    with pytest.raises(ValueError, match=expected):
        VALIDATOR.validate_submission_zip(broken)
