"""Bootstraps the vendored FinRL-Trading checkout onto sys.path.

FinRL-Trading is vendored as a git submodule (vendor/FinRL-Trading/) rather
than pip-installed -- see PROJECT_PLAN.md's "Vendoring" decision for why.
Its own source uses absolute imports like `from src.data.data_fetcher import
...`, which only resolve when its repo root is on sys.path (so that its
`src/` directory is importable as the top-level package `src`). Call
ensure_finrl_on_path() before importing anything under `src.*`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_FINRL_ROOT = REPO_ROOT / "vendor" / "FinRL-Trading"


def ensure_finrl_on_path() -> None:
    path_str = str(VENDOR_FINRL_ROOT)
    if path_str not in sys.path:
        if not VENDOR_FINRL_ROOT.is_dir():
            raise FileNotFoundError(
                f"Vendored FinRL-Trading checkout not found at {VENDOR_FINRL_ROOT}. "
                "Did you run `git submodule update --init`?"
            )
        sys.path.insert(0, path_str)
