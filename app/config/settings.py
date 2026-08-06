"""Shared config: environment-driven where it matters, sensible local
defaults otherwise."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_mlflow_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI env var if set (e.g. Postgres in Phase 5's
    docker-compose), else a local SQLite file. MLflow's plain filesystem
    backend ('./mlruns') is in maintenance mode as of MLflow 3.x and
    refuses to run without an explicit opt-out."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
