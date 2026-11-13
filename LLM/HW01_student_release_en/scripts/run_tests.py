#!/usr/bin/env python3
"""Run the public test suite from the project root."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
test_path = "tests/public" if (root / "tests" / "public").is_dir() else "tests"
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", test_path], cwd=root))
