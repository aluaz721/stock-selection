import pandas as pd

from app.config.vendor_path import ensure_finrl_on_path

ensure_finrl_on_path()
from src.strategies.base_strategy import StrategyConfig  # noqa: E402

from backtests.compare_vs_benchmarks import EqualWeightUniverseStrategy  # noqa: E402


class TestEqualWeightUniverseStrategy:
    def test_equal_weights_full_cached_universe(self, tmp_path):
        out_dir = tmp_path / "2025-12-31"
        out_dir.mkdir(parents=True)
        preds = pd.DataFrame({"tic": ["A", "B", "C", "D"], "bucket": ["x"] * 4, "predicted_return": [0.1] * 4})
        preds.to_csv(out_dir / "sp500_ml_bucket_predictions_x.csv", index=False)

        strategy = EqualWeightUniverseStrategy(StrategyConfig(name="equal_weight"), work_dir=tmp_path)
        result = strategy.generate_weights({}, target_date="2025-12-31")

        assert set(result.weights["tic"]) == {"A", "B", "C", "D"}
        assert (result.weights["weight"] == 0.25).all()

    def test_missing_cache_returns_empty_with_reason(self, tmp_path):
        strategy = EqualWeightUniverseStrategy(StrategyConfig(name="equal_weight"), work_dir=tmp_path)
        result = strategy.generate_weights({}, target_date="2025-12-31")

        assert result.weights.empty
        assert "reason" in result.metadata

    def test_dedupes_repeated_tickers(self, tmp_path):
        out_dir = tmp_path / "2025-12-31"
        out_dir.mkdir(parents=True)
        preds = pd.DataFrame({"tic": ["A", "A", "B"], "bucket": ["x", "y", "z"], "predicted_return": [0.1] * 3})
        preds.to_csv(out_dir / "sp500_ml_bucket_predictions_x.csv", index=False)

        strategy = EqualWeightUniverseStrategy(StrategyConfig(name="equal_weight"), work_dir=tmp_path)
        result = strategy.generate_weights({}, target_date="2025-12-31")

        assert len(result.weights) == 2
        assert (result.weights["weight"] == 0.5).all()
