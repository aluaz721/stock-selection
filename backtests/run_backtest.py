"""CLI: run FinRL-Trading's BacktestEngine against MLBucketStrategy across
a sequence of quarters.

Each rebalance is a full ml_bucket_selection.py invocation (~2 minutes --
retrains the per-bucket model competition fresh for that quarter's
point-in-time universe, measured directly during development). Keep the
quarter list bounded accordingly.

FinRL-Trading's BacktestEngine fetches its SPY/QQQ benchmark prices via its
own FMP-backed fetch_price_data. We monkeypatch that to our yfinance
fetcher (app/data/fetch_prices.py) so benchmarking doesn't need an FMP key
-- the vendored module itself is untouched, only the name it resolves
fetch_price_data through at call time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.config.vendor_path import ensure_finrl_on_path

ensure_finrl_on_path()

import src.backtest.backtest_engine as backtest_engine_module  # noqa: E402
from src.backtest.backtest_engine import BacktestConfig, BacktestEngine  # noqa: E402
from src.strategies.base_strategy import StrategyConfig  # noqa: E402
from src.strategies.ml_bucket_selection import datadate_to_tradedate  # noqa: E402

from app.data.fetch_prices import drop_tickers_with_price_gaps, fetch_daily_prices  # noqa: E402
from app.strategy.ml_bucket_strategy import MLBucketStrategy  # noqa: E402


def fetch_price_data_compat(tickers, start_date, end_date, preferred_source=None) -> pd.DataFrame:
    """Signature-compatible with FinRL-Trading's fetch_price_data, backed by
    our yfinance fetcher."""
    return fetch_daily_prices(list(tickers), start_date, end_date)


def build_weight_signals(strategy: MLBucketStrategy, infer_dates: list[str]) -> pd.DataFrame:
    """One row per quarter, indexed by the actual TRADEDATE (when the picks
    would have been actionable) -- not the quarter-end datadate itself,
    which per ML_STOCK_SELECTION.md's CRITICAL SPEC 1 isn't public/
    actionable until ~2 months later."""
    rows = {}
    for infer_date in infer_dates:
        result = strategy.generate_weights({}, target_date=infer_date)
        tradedate = pd.Timestamp(datadate_to_tradedate(infer_date))
        rows[tradedate] = (
            pd.Series(dtype=float) if result.weights.empty else result.weights.set_index("tic")["weight"]
        )

    weight_signals = pd.DataFrame(rows).T.sort_index()
    weight_signals.index.name = "date"
    return weight_signals.fillna(0.0)


def run_backtest(
    db_path: Path, infer_dates: list[str], top_n_per_bucket: int = 5
) -> backtest_engine_module.BacktestResult:
    backtest_engine_module.fetch_price_data = fetch_price_data_compat

    strategy = MLBucketStrategy(
        StrategyConfig(name="ml_bucket_selector"), db_path=db_path, top_n_per_bucket=top_n_per_bucket
    )
    weight_signals = build_weight_signals(strategy, infer_dates)

    all_tickers = weight_signals.columns.tolist()
    start_date = weight_signals.index.min().strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    prices = fetch_daily_prices(all_tickers, start_date, end_date)
    prices = drop_tickers_with_price_gaps(prices)
    weight_signals = weight_signals[[c for c in weight_signals.columns if c in set(prices["tic"])]]

    engine = BacktestEngine(BacktestConfig(start_date=start_date, end_date=end_date))
    return engine.run_backtest("ml_bucket_selector", prices, weight_signals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest MLBucketStrategy via FinRL-Trading's BacktestEngine")
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--infer-dates", nargs="+", required=True, help="quarter-end dates, e.g. 2024-03-31 2024-06-30"
    )
    parser.add_argument("--top-n-per-bucket", type=int, default=5)
    args = parser.parse_args()

    result = run_backtest(Path(args.db), args.infer_dates, top_n_per_bucket=args.top_n_per_bucket)

    print(f"\nStrategy: {result.strategy_name}")
    print(f"Annualized return: {result.annualized_return:.2%}")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value}")
    print("\nBenchmarks:")
    for bm, ann in result.benchmark_annualized.items():
        print(f"  {bm}: {ann:.2%} annualized")


if __name__ == "__main__":
    main()
