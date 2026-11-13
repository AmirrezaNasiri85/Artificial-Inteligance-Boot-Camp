#!/usr/bin/env python3
"""Run the public HW02 suite from the package root."""

from __future__ import annotations

import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"]))

