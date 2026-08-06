"""Computes portfolio cumulative-return histories directly from prices and
weight_signals, bypassing FinRL-Trading's `bt`-based BacktestEngine
entirely for this.

Not a style choice -- `bt`'s Sharpe/Sortino has been caught reporting
impossible values twice (see PROJECT_PLAN.md known risks): once on its
buy-and-hold benchmark path, once on a normal RunOnDate-rebalanced
strategy with a large position count. Return and max_drawdown matched
independent calculation closely both times, but rather than keep
re-deriving "is this specific Sharpe trustworthy" per report, this module
sidesteps `bt` for the whole computation so every number it produces is
verified-methodology by construction.
"""

from __future__ import annotations

import pandas as pd


def compute_portfolio_history(weight_signals: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Cumulative return (starting at 1.0) of a rebalanced portfolio.

    Between rebalance dates, weights are held fixed (drift with individual
    stock prices, not continuously re-targeted) -- `weight_signals` gets
    forward-filled to the daily index, then each day's return uses the
    *previous* day's weights (shift(1)), matching how a real portfolio
    only earns tomorrow's return on today's holdings.
    """
    wide = prices.pivot(index="datadate", columns="tic", values="adj_close").sort_index().ffill()
    start = weight_signals.index.min()
    idx = wide.loc[start:].index

    ws = weight_signals.reindex(idx).ffill().fillna(0.0)
    ws = ws[[c for c in ws.columns if c in wide.columns]]
    row_sum = ws.sum(axis=1)
    nonzero = row_sum > 0
    ws.loc[nonzero] = ws.loc[nonzero].div(row_sum[nonzero], axis=0)

    daily_returns = wide[ws.columns].loc[start:].pct_change().fillna(0.0)
    port_returns = (ws.shift(1).fillna(0.0) * daily_returns).sum(axis=1)
    port_returns = port_returns.iloc[1:]  # first day has no prior weight yet

    return (1 + port_returns).cumprod()


def compute_buy_and_hold_history(prices: pd.DataFrame, tic: str) -> pd.Series:
    """Cumulative return (starting at 1.0) of holding a single ticker
    throughout `prices`' date range -- e.g. an SPY/QQQ benchmark series."""
    g = prices[prices["tic"] == tic].sort_values("datadate")
    return pd.Series(g["adj_close"].values / g["adj_close"].iloc[0], index=g["datadate"].values)


def compute_stats(cumulative_return: pd.Series) -> dict:
    """annualized_return, sharpe_ratio, max_drawdown from a cumulative-return series."""
    returns = cumulative_return.pct_change().dropna()
    total_return = cumulative_return.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * (252**0.5)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    max_dd = ((cumulative_return / cumulative_return.cummax()) - 1).min()
    return {"annualized_return": float(ann_return), "sharpe_ratio": float(sharpe), "max_drawdown": float(max_dd)}
