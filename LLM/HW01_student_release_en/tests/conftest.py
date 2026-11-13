"""Make the assignment project root importable regardless of pytest working directory."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))
