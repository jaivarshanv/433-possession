"""Make `possession` importable in tests. Tests use SYNTHETIC data only, never match data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
