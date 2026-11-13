#!/usr/bin/env python3
"""Fail when Persian or Arabic text appears in release-facing text files."""

from __future__ import annotations

import argparse
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff]")
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".csv", ".ipynb", ".toml"}
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".ipynb_checkpoints",
    "__pycache__",
}


def _find_violations(name: str, payload: bytes) -> list[str]:
    text = payload.decode("utf-8")
    return [
        f"{name}:{line_number}: {line.strip()}"
        for line_number, line in enumerate(text.splitlines(), start=1)
        if ARABIC_SCRIPT.search(line)
    ]


def _directory_files(path: Path) -> Iterable[tuple[str, bytes]]:
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if candidate.suffix.lower() in TEXT_SUFFIXES:
            yield str(candidate), candidate.read_bytes()


def scan_path(path: Path) -> tuple[int, list[str]]:
    scanned = 0
    violations: list[str] = []
    if path.is_dir():
        entries = _directory_files(path)
    elif path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)

        def archive_entries() -> Iterable[tuple[str, bytes]]:
            with archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    name = PurePosixPath(info.filename)
                    if not info.is_dir() and name.suffix.lower() in TEXT_SUFFIXES:
                        yield f"{path}!{info.filename}", archive.read(info)

        entries = archive_entries()
    elif path.suffix.lower() in TEXT_SUFFIXES:
        entries = [(str(path), path.read_bytes())]
    else:
        entries = []
    for name, payload in entries:
        scanned += 1
        violations.extend(_find_violations(name, payload))
    return scanned, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("HW01"), Path("HW02")])
    arguments = parser.parse_args()
    scanned = 0
    violations: list[str] = []
    for path in arguments.paths:
        count, found = scan_path(path)
        scanned += count
        violations.extend(found)
    if violations:
        raise SystemExit("English-only audit failed:\n" + "\n".join(violations))
    print(f"English-only audit passed: {scanned} text files scanned; 0 violations.")


if __name__ == "__main__":
    main()
