"""Ensure the project root is on sys.path so that both `ml.*` and `app.*` can be imported."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # /Users/dell/Documents/ClaimSight
BACKEND_ROOT = PROJECT_ROOT / "backend"

for p in (str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
