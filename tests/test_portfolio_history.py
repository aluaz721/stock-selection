import numpy as np
import pandas as pd
import pytest

from app.reporting.portfolio_history import (
    compute_buy_and_hold_history,
    compute_portfolio_history,
    compute_stats,
)


def make_prices(tic: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"datadate": dates, "tic": tic, "adj_close": closes})


class TestComputePortfolioHistory:
    def test_single_ticker_full_weight_matches_buy_and_hold(self):
        prices = make_prices("AAA", [100, 110, 121, 133.1])
        weight_signals = pd.DataFrame({"AAA": [1.0]}, index=[prices["datadate"].iloc[0]])

        history = compute_portfolio_history(weight_signals, prices)

        # the rebalance-date's own return is dropped (no prior weight to
        # apply it to yet), so history[0] is the *next* day's return (+10%,
        # applying the rebalance date's weight) and tracks AAA's 10%/day
        # growth from there
        assert history.iloc[0] == pytest.approx(1.10)
        assert history.iloc[-1] == pytest.approx(1.331, rel=1e-3)

    def test_equal_weight_two_tickers_averages_returns(self):
        aaa = make_prices("AAA", [100, 110])  # +10%
        bbb = make_prices("BBB", [100, 90])  # -10%
        prices = pd.concat([aaa, bbb], ignore_index=True)
        weight_signals = pd.DataFrame({"AAA": [0.5], "BBB": [0.5]}, index=[aaa["datadate"].iloc[0]])

        history = compute_portfolio_history(weight_signals, prices)

        assert history.iloc[-1] == pytest.approx(1.0, abs=1e-9)  # +10% and -10% average out

    def test_rebalance_updates_weights_at_new_date(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame(
            {"datadate": list(dates) * 2, "tic": ["AAA"] * 6 + ["BBB"] * 6, "adj_close": [100] * 12}
        )
        weight_signals = pd.DataFrame(
            {"AAA": [1.0, 0.0], "BBB": [0.0, 1.0]}, index=[dates[0], dates[3]]
        )

        history = compute_portfolio_history(weight_signals, prices)

        assert len(history) == 5  # 6 days minus the first (no prior weight yet)


class TestComputeBuyAndHoldHistory:
    def test_normalizes_to_one_at_start(self):
        prices = make_prices("SPY", [400, 420, 440])
        history = compute_buy_and_hold_history(prices, "SPY")
        assert history.iloc[0] == pytest.approx(1.0)
        assert history.iloc[-1] == pytest.approx(1.10)


class TestComputeStats:
    def test_flat_series_has_zero_return_and_drawdown(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        flat = pd.Series([1.0] * 10, index=idx)
        stats = compute_stats(flat)
        assert stats["annualized_return"] == pytest.approx(0.0, abs=1e-9)
        assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_steady_growth_has_positive_sharpe(self):
        idx = pd.bdate_range("2024-01-01", periods=252)
        steady = pd.Series(np.linspace(1.0, 1.20, 252), index=idx)
        stats = compute_stats(steady)
        assert stats["annualized_return"] > 0
        assert stats["sharpe_ratio"] > 0

    def test_drawdown_is_negative_after_a_drop(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        series = pd.Series([1.0, 1.2, 0.9, 1.1], index=idx)
        stats = compute_stats(series)
        assert stats["max_drawdown"] == pytest.approx((0.9 / 1.2) - 1)
