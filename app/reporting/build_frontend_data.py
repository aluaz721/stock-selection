"""Builds the committed JSON files streamlit_app/app.py reads:
data/predictions/history.json (strategy vs. SPY/QQQ cumulative performance,
from the verified 2017-2025 backtest) and data/predictions/latest.json
(today's top picks per bucket, from a fresh mixed-vintage run).

This is a backfill/seed generator, not the live daily-refresh mechanism --
that's still blocked on live data (see PROJECT_PLAN.md's Live refresh
decision). Re-run this manually to refresh both files with whatever the
bundled/local DB currently has; once refresh_fundamentals.py exists, the
daily workflow will call something equivalent automatically.

Uses top-3-per-bucket, the configuration locked in after the position-count
sweep and two-half robustness check (see PROJECT_PLAN.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from app.config.vendor_path import VENDOR_FINRL_ROOT, ensure_finrl_on_path

ensure_finrl_on_path()

from src.strategies.base_strategy import StrategyConfig  # noqa: E402

from app.data.fetch_prices import drop_tickers_with_price_gaps, fetch_daily_prices  # noqa: E402
from app.reporting.portfolio_history import (  # noqa: E402
    compute_buy_and_hold_history,
    compute_portfolio_history,
    compute_stats,
)
from app.strategy.ml_bucket_strategy import MLBucketStrategy  # noqa: E402
from backtests.run_backtest import build_weight_signals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "predictions"
ML_BUCKET_SCRIPT = VENDOR_FINRL_ROOT / "src" / "strategies" / "ml_bucket_selection.py"
TOP_N_PER_BUCKET = 3  # locked in -- see PROJECT_PLAN.md's position-count sweep


def build_history(db_path: Path, infer_dates: list[str], work_dir: Path) -> None:
    strategy = MLBucketStrategy(
        StrategyConfig(name="ml_bucket_selector"), db_path=db_path, top_n_per_bucket=TOP_N_PER_BUCKET, work_dir=work_dir
    )
    weight_signals = build_weight_signals(strategy, infer_dates)

    all_tickers = sorted(set(weight_signals.columns) | {"SPY", "QQQ"})
    start_date = weight_signals.index.min().strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_daily_prices(all_tickers, start_date, end_date)
    prices = drop_tickers_with_price_gaps(prices)
    valid = set(prices["tic"])
    weight_signals = weight_signals[[c for c in weight_signals.columns if c in valid]]

    strategy_history = compute_portfolio_history(weight_signals, prices)
    bench_prices = prices[prices["datadate"] >= strategy_history.index.min()]
    spy_history = compute_buy_and_hold_history(bench_prices, "SPY")
    qqq_history = compute_buy_and_hold_history(bench_prices, "QQQ")

    combined = pd.DataFrame({"strategy": strategy_history, "spy": spy_history, "qqq": qqq_history}).dropna()

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": {
            "top_n_per_bucket": TOP_N_PER_BUCKET,
            "start": combined.index.min().strftime("%Y-%m-%d"),
            "end": combined.index.max().strftime("%Y-%m-%d"),
        },
        "summary": {name: compute_stats(combined[name]) for name in ["strategy", "spy", "qqq"]},
        "series": [
            {"date": d.strftime("%Y-%m-%d"), "strategy": round(r.strategy, 4), "spy": round(r.spy, 4), "qqq": round(r.qqq, 4)}
            for d, r in combined.iterrows()
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "history.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'history.json'} ({len(combined)} days, {combined.index.min().date()} -> {combined.index.max().date()})")


def build_latest(db_path: Path, val_cutoff: str, work_dir: Path) -> None:
    """Mixed-vintage: each ticker uses its own latest available quarter --
    the closest thing to 'today's picks' the bundled data supports."""
    out_dir = work_dir / "mixed_vintage"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, str(ML_BUCKET_SCRIPT),
            "--val-cutoff", val_cutoff,
            "--mixed-vintage",
            "--db", str(db_path),
            "--output-dir", str(out_dir) + "/",
        ],
        capture_output=True,
        text=True,
        check=False,  # exit code can be nonzero even on a successful run
        # (e.g. the openpyxl-dependent Excel export failing after everything
        # that matters already saved) -- we inspect the output files ourselves.
    )
    pred_files = sorted(out_dir.glob("sp500_ml_bucket_predictions_*.csv"))
    if not pred_files:
        raise RuntimeError(f"mixed-vintage run produced no output (exit {result.returncode}): {result.stderr[-1000:]}")

    predictions = pd.read_csv(pred_files[-1])
    # `datadate` is the literal string "mixed" in this mode (a marker, not a
    # real date) -- each ticker's actual quarter is in `original_datadate`.
    as_of = predictions["original_datadate"].max()
    vintage_breakdown = predictions["original_datadate"].value_counts().sort_index(ascending=False).to_dict()

    buckets = {}
    for bucket, g in predictions.groupby("bucket"):
        top = g.sort_values("predicted_return", ascending=False).head(TOP_N_PER_BUCKET)
        buckets[bucket] = [
            {"tic": row.tic, "predicted_return": round(row.predicted_return, 4)} for row in top.itertuples()
        ]

    payload = {
        "as_of": str(as_of),
        "vintage_breakdown": {str(k): int(v) for k, v in vintage_breakdown.items()},
        "val_cutoff": val_cutoff,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "buckets": buckets,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'latest.json'} (as_of {as_of}, {sum(len(v) for v in buckets.values())} picks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/predictions/*.json for the frontend")
    parser.add_argument("--db", required=True)
    parser.add_argument("--val-cutoff", required=True, help="for the mixed-vintage 'latest' run")
    parser.add_argument(
        "--infer-dates", nargs="+", required=True, help="quarter-end dates for the history backtest"
    )
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="frontend_data_"))
    build_history(Path(args.db), args.infer_dates, work_dir)
    build_latest(Path(args.db), args.val_cutoff, work_dir)


if __name__ == "__main__":
    main()
