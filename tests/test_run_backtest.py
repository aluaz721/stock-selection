import pandas as pd

from backtests.run_backtest import build_weight_signals


class FakeResult:
    def __init__(self, weights: pd.DataFrame):
        self.weights = weights


class FakeStrategy:
    """Returns fixed weights keyed by the target_date (a datadate) it's
    asked for, so the test controls exactly what build_weight_signals
    should assemble."""

    def __init__(self, weights_by_infer_date: dict[str, pd.DataFrame]):
        self.weights_by_infer_date = weights_by_infer_date

    def generate_weights(self, data, target_date=None):
        weights = self.weights_by_infer_date.get(target_date, pd.DataFrame(columns=["tic", "weight"]))
        return FakeResult(weights)


class TestBuildWeightSignals:
    def test_indexed_by_tradedate_not_datadate(self):
        """A quarter-end datadate maps to a tradedate ~2 months later
        (ML_STOCK_SELECTION.md's CRITICAL SPEC 1) -- the weight_signals
        index must be the actionable tradedate, not the datadate itself."""
        strategy = FakeStrategy({"2025-09-30": pd.DataFrame({"tic": ["AAA"], "weight": [1.0]})})

        signals = build_weight_signals(strategy, ["2025-09-30"])

        assert len(signals) == 1
        # 2025-09-30 -> tradedate 2025-12-01 per DATADATE_TO_TRADEDATE_MAP
        assert signals.index[0] == pd.Timestamp("2025-12-01")
        assert signals.loc[pd.Timestamp("2025-12-01"), "AAA"] == 1.0

    def test_one_row_per_quarter(self):
        strategy = FakeStrategy(
            {
                "2024-12-31": pd.DataFrame({"tic": ["AAA"], "weight": [1.0]}),
                "2025-03-31": pd.DataFrame({"tic": ["BBB"], "weight": [1.0]}),
            }
        )

        signals = build_weight_signals(strategy, ["2024-12-31", "2025-03-31"])

        assert len(signals) == 2
        assert set(signals.columns) == {"AAA", "BBB"}

    def test_empty_weights_become_zero_row(self):
        strategy = FakeStrategy({})  # no matching date -> FakeStrategy returns empty weights

        signals = build_weight_signals(strategy, ["2025-09-30"])

        assert len(signals) == 1
        assert (signals.iloc[0] == 0.0).all() if len(signals.columns) else True

    def test_rows_sorted_by_date(self):
        strategy = FakeStrategy(
            {
                "2025-03-31": pd.DataFrame({"tic": ["AAA"], "weight": [1.0]}),
                "2024-12-31": pd.DataFrame({"tic": ["AAA"], "weight": [1.0]}),
            }
        )

        # passed out of order -> output must still be chronological
        signals = build_weight_signals(strategy, ["2025-03-31", "2024-12-31"])

        assert list(signals.index) == sorted(signals.index)
