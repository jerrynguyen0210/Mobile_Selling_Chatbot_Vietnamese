"""Root conftest.py — adds back-end/ to sys.path so ``app.*`` is importable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "back-end"))
