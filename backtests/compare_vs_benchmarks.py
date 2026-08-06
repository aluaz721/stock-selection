"""Compares MLBucketStrategy against an equal-weight baseline over the same
point-in-time universe, plus SPY/QQQ -- run_backtest.py alone isn't enough
to tell whether the model's per-bucket ranking adds value or whether any
result just reflects which stocks were in the S&P 500 that quarter.

The equal-weight control reuses MLBucketStrategy's own cached predictions
CSVs (already written to work_dir by build_weight_signals) rather than
re-invoking ml_bucket_selection.py -- each invocation costs ~2 minutes, so
recomputing per comparison strategy isn't worth it. Run run_backtest first
against the same work_dir, or just use main() here, which sequences both.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from app.config.vendor_path import ensure_finrl_on_path

ensure_finrl_on_path()

import src.backtest.backtest_engine as backtest_engine_module  # noqa: E402
from src.backtest.backtest_engine import BacktestConfig, BacktestEngine  # noqa: E402
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult  # noqa: E402
from src.strategies.ml_bucket_selection import datadate_to_tradedate  # noqa: E402

from app.data.fetch_prices import drop_tickers_with_price_gaps, fetch_daily_prices
from app.strategy.ml_bucket_strategy import MLBucketStrategy
from backtests.run_backtest import build_weight_signals, fetch_price_data_compat


class EqualWeightUniverseStrategy(BaseStrategy):
    """No model, no selection -- equal-weights the full point-in-time S&P
    500 universe ml_bucket_selection.py considered that quarter (read from
    its own predictions CSV, already produced by MLBucketStrategy for the
    same quarter in the same work_dir). The control: if MLBucketStrategy
    doesn't beat this, its per-bucket ranking isn't adding anything over
    just holding the index."""

    def __init__(self, config: StrategyConfig, work_dir: Path):
        super().__init__(config)
        self.work_dir = Path(work_dir)

    def generate_weights(self, data, target_date: str | None = None) -> StrategyResult:
        out_dir = self.work_dir / target_date
        pred_files = sorted(out_dir.glob("sp500_ml_bucket_predictions_*.csv"))
        if not pred_files:
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=["tic", "weight"]),
                metadata={"as_of": target_date, "n_selected": 0, "reason": "no cached predictions for this quarter"},
            )

        tics = pd.read_csv(pred_files[-1])["tic"].unique()
        weights = pd.DataFrame({"tic": tics, "weight": 1.0 / len(tics)})
        return StrategyResult(
            strategy_name=self.config.name,
            weights=weights,
            metadata={"as_of": target_date, "n_selected": len(weights)},
        )


def run_comparison(
    db_path: Path, infer_dates: list[str], top_n_per_bucket: int = 5, work_dir: Path | None = None
) -> dict[str, "backtest_engine_module.BacktestResult"]:
    backtest_engine_module.fetch_price_data = fetch_price_data_compat
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ml_bucket_"))

    ml_bucket = MLBucketStrategy(
        StrategyConfig(name="ml_bucket_selector"),
        db_path=db_path,
        top_n_per_bucket=top_n_per_bucket,
        work_dir=work_dir,
    )
    ml_bucket_signals = build_weight_signals(ml_bucket, infer_dates)

    equal_weight = EqualWeightUniverseStrategy(StrategyConfig(name="equal_weight"), work_dir=work_dir)
    equal_weight_signals = build_weight_signals(equal_weight, infer_dates)

    all_tickers = sorted(set(ml_bucket_signals.columns) | set(equal_weight_signals.columns))
    start_date = min(ml_bucket_signals.index.min(), equal_weight_signals.index.min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_daily_prices(all_tickers, start_date, end_date)
    prices = drop_tickers_with_price_gaps(prices)
    valid_tickers = set(prices["tic"])
    ml_bucket_signals = ml_bucket_signals[[c for c in ml_bucket_signals.columns if c in valid_tickers]]
    equal_weight_signals = equal_weight_signals[[c for c in equal_weight_signals.columns if c in valid_tickers]]

    engine = BacktestEngine(BacktestConfig(start_date=start_date, end_date=end_date))

    results = {
        "ml_bucket_selector": engine.run_backtest("ml_bucket_selector", prices, ml_bucket_signals),
        "equal_weight": engine.run_backtest("equal_weight", prices, equal_weight_signals),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLBucketStrategy vs. equal-weight and SPY/QQQ")
    parser.add_argument("--db", required=True)
    parser.add_argument("--infer-dates", nargs="+", required=True)
    parser.add_argument("--top-n-per-bucket", type=int, default=5)
    parser.add_argument(
        "--work-dir", default=None, help="where to cache per-quarter ml_bucket_selection.py output (kept after the run)"
    )
    args = parser.parse_args()

    results = run_comparison(
        Path(args.db), args.infer_dates, top_n_per_bucket=args.top_n_per_bucket, work_dir=args.work_dir
    )

    rows = {}
    for name, result in results.items():
        rows[name] = {
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.metrics.get("sharpe_ratio"),
            "max_drawdown": result.metrics.get("max_drawdown"),
        }
    any_result = next(iter(results.values()))
    for bm, ann in any_result.benchmark_annualized.items():
        rows[bm] = {
            "annualized_return": ann,
            "sharpe_ratio": any_result.benchmark_metrics.get(bm, {}).get("sharpe_ratio"),
            "max_drawdown": any_result.benchmark_metrics.get(bm, {}).get("max_drawdown"),
        }

    table = pd.DataFrame(rows).T
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
