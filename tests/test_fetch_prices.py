from unittest.mock import patch

import pandas as pd
import pytest

import app.data.fetch_prices as fetch_prices_module
from app.data.fetch_prices import PRICE_COLUMNS, drop_tickers_with_price_gaps, fetch_daily_prices, to_yfinance_symbol


def make_yfinance_download_output(tics_and_bases: dict[str, float], n_days: int = 3) -> pd.DataFrame:
    """Matches yf.download()'s real return shape: MultiIndex columns
    (Price, Ticker), one row per date, index named 'Date' -- verified
    against a live call during Phase 1."""
    dates = pd.bdate_range("2026-01-02", periods=n_days, name="Date")
    cols = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "High", "Low", "Open", "Volume"], list(tics_and_bases)],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for tic, base in tics_and_bases.items():
        for i in range(n_days):
            raw.loc[dates[i], ("Open", tic)] = base + i
            raw.loc[dates[i], ("High", tic)] = base + i + 1
            raw.loc[dates[i], ("Low", tic)] = base + i - 1
            raw.loc[dates[i], ("Close", tic)] = base + i + 0.5
            raw.loc[dates[i], ("Adj Close", tic)] = base + i + 0.5
            raw.loc[dates[i], ("Volume", tic)] = 1_000 + i
    return raw


class TestFetchDailyPrices:
    def test_empty_ticker_list_returns_empty_frame_with_columns(self):
        out = fetch_daily_prices([], "2026-01-02", "2026-01-06")
        assert list(out.columns) == PRICE_COLUMNS
        assert len(out) == 0

    def test_reshapes_yfinance_output_to_price_columns(self):
        raw = make_yfinance_download_output({"AAA": 100.0, "BBB": 50.0})
        with patch.object(fetch_prices_module.yf, "download", return_value=raw) as mock_download:
            out = fetch_daily_prices(["AAA", "BBB"], "2026-01-02", "2026-01-06")

        mock_download.assert_called_once()
        assert list(out.columns) == PRICE_COLUMNS
        assert len(out) == 6
        assert set(out["tic"]) == {"AAA", "BBB"}

        aaa_first = out[out["tic"] == "AAA"].sort_values("datadate").iloc[0]
        assert aaa_first["prcod"] == pytest.approx(100.0)
        assert aaa_first["prccd"] == pytest.approx(100.5)
        assert aaa_first["gvkey"] == "AAA"

    def test_empty_download_result_still_has_price_columns(self):
        with patch.object(fetch_prices_module.yf, "download", return_value=pd.DataFrame()):
            out = fetch_daily_prices(["ZZZZ"], "2026-01-02", "2026-01-06")
        assert list(out.columns) == PRICE_COLUMNS
        assert len(out) == 0

    def test_dot_tickers_converted_for_yfinance_but_original_in_output(self):
        # yf.download is called with dash notation ("BRK-B"); the mock's
        # MultiIndex columns must match what fetch_daily_prices actually
        # requests, so build it from the converted symbol.
        raw = make_yfinance_download_output({"BRK-B": 400.0})
        with patch.object(fetch_prices_module.yf, "download", return_value=raw) as mock_download:
            out = fetch_daily_prices(["BRK.B"], "2026-01-02", "2026-01-06")

        requested = mock_download.call_args[0][0]
        assert requested == ["BRK-B"]
        assert set(out["tic"]) == {"BRK.B"}  # output stays in original (DB) notation


class TestToYfinanceSymbol:
    def test_converts_dot_to_dash(self):
        assert to_yfinance_symbol("BRK.B") == "BRK-B"
        assert to_yfinance_symbol("BF.B") == "BF-B"

    def test_leaves_plain_tickers_unchanged(self):
        assert to_yfinance_symbol("AAPL") == "AAPL"


class TestDropTickersWithPriceGaps:
    def test_drops_ticker_with_leading_gap(self):
        dates = pd.bdate_range("2026-01-02", periods=4)
        prices = pd.concat(
            [
                pd.DataFrame({"datadate": dates, "tic": "GOOD", "adj_close": [10.0, 11.0, 12.0, 13.0]}),
                # BAD has no price until day 3 -- ffill can't back-fill a leading gap
                pd.DataFrame({"datadate": dates[2:], "tic": "BAD", "adj_close": [20.0, 21.0]}),
            ],
            ignore_index=True,
        )

        out = drop_tickers_with_price_gaps(prices)

        assert set(out["tic"]) == {"GOOD"}

    def test_keeps_tickers_with_no_gaps(self):
        dates = pd.bdate_range("2026-01-02", periods=3)
        prices = pd.DataFrame({"datadate": list(dates) * 2, "tic": ["A"] * 3 + ["B"] * 3, "adj_close": range(6)})

        out = drop_tickers_with_price_gaps(prices)

        assert set(out["tic"]) == {"A", "B"}

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["datadate", "tic", "adj_close"])
        out = drop_tickers_with_price_gaps(empty)
        assert out.empty
