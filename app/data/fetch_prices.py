"""Daily OHLCV price data via yfinance.

FinRL-Trading's BacktestEngine needs daily prices to mark a portfolio
between rebalances (ml_bucket_selection.py itself only works with quarterly
fundamentals/prices -- see app/strategy/ml_bucket_strategy.py). Reshapes
yfinance's output to match FinRL-Trading's own price schema (PRICE_COLUMNS)
so BacktestEngine doesn't care which source supplied it.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = [
    "gvkey", "datadate", "tic", "prccd", "prcod", "prchd", "prcld", "cshtrd", "adj_close",
]


def to_yfinance_symbol(tic: str) -> str:
    """yfinance expects dash-separated class shares (BRK-B), but
    FinRL-Trading's bundled DB uses dots (BRK.B) -- see
    ML_STOCK_SELECTION.md's "Known Quirks": "FMP uses `-` (BRK-B), DB uses
    `.` (BRK.B)". Without this, symbols like BRK.B/BF.B fail with
    "possibly delisted; no timezone found" even though they trade fine."""
    return tic.replace(".", "-")


def fetch_daily_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Long-format daily OHLCV for `tickers` between start_date and end_date.
    Output `tic` values match the input (DB) notation, not yfinance's."""
    if not tickers:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    yf_to_original = {to_yfinance_symbol(t): t for t in tickers}
    raw = yf.download(list(yf_to_original), start=start_date, end=end_date, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    long = raw.stack(level="Ticker", future_stack=True).reset_index()
    long.columns.name = None
    long["Ticker"] = long["Ticker"].map(yf_to_original)
    long = long.rename(
        columns={
            "Date": "datadate",
            "Ticker": "tic",
            "Close": "prccd",
            "Open": "prcod",
            "High": "prchd",
            "Low": "prcld",
            "Volume": "cshtrd",
            "Adj Close": "adj_close",
        }
    )
    long["gvkey"] = long["tic"]
    long["datadate"] = pd.to_datetime(long["datadate"])
    return long.dropna(subset=["prccd"])[PRICE_COLUMNS].reset_index(drop=True)


def drop_tickers_with_price_gaps(prices: pd.DataFrame) -> pd.DataFrame:
    """Drops any ticker whose price series still has NaNs after forward-fill
    (most commonly a leading gap -- no valid price yet at the start of the
    window, e.g. a stock that went private/was acquired mid-window and
    yfinance only has a partial history). FinRL-Trading's BacktestEngine
    hard-crashes (`bt.core.SecurityBase.allocate`) if asked to allocate
    capital to a security with a NaN price on a rebalance date, so this has
    to happen before prices reach it -- confirmed via a real crash on EA
    during Phase 2 development (likely its 2025 take-private deal)."""
    if prices.empty:
        return prices
    wide = prices.pivot(index="datadate", columns="tic", values="adj_close").sort_index().ffill()
    bad_tickers = set(wide.columns[wide.isna().any()])
    if not bad_tickers:
        return prices
    return prices[~prices["tic"].isin(bad_tickers)].reset_index(drop=True)
