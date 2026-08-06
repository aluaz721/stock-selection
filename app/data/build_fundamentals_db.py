"""Loads FinRL-Trading's bundled fundamentals CSV into a local SQLite DB.

vendor/FinRL-Trading/data/fundamental_data_full.csv ships with the vendored
repo (see PROJECT_PLAN.md's "Bundled historical data" decision) -- someone
else already spent an FMP budget building it: 22,909 records, 715 tickers,
64 columns, 2015-Q2 through 2026-Q1. This loads it into the shape
src/strategies/ml_bucket_selection.py expects (a SQLite `fundamental_data`
table), in our own data/ directory rather than inside vendor/, so the
vendored checkout stays untouched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app.config.vendor_path import VENDOR_FINRL_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_CSV_PATH = VENDOR_FINRL_ROOT / "data" / "fundamental_data_full.csv"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "finrl_trading.db"


def build_fundamentals_db(csv_path: Path = BUNDLED_CSV_PATH, db_path: Path = DEFAULT_DB_PATH) -> Path:
    """(Re)builds the `fundamental_data` table at db_path from csv_path.
    Replaces any existing table, so reruns don't duplicate rows."""
    df = pd.read_csv(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("fundamental_data", conn, if_exists="replace", index=False)
    finally:
        conn.close()
    return db_path


def main() -> None:
    db_path = build_fundamentals_db()
    conn = sqlite3.connect(db_path)
    try:
        n_records, n_tickers = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ticker) FROM fundamental_data"
        ).fetchone()
    finally:
        conn.close()
    print(f"Built {db_path} -- {n_records} records, {n_tickers} tickers")


if __name__ == "__main__":
    main()
