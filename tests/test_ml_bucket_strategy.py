from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.strategy.ml_bucket_strategy import MLBucketStrategy, previous_quarter_end
from app.config.vendor_path import ensure_finrl_on_path

ensure_finrl_on_path()
from src.strategies.base_strategy import StrategyConfig  # noqa: E402


class TestPreviousQuarterEnd:
    def test_within_year(self):
        assert previous_quarter_end("2025-12-31") == "2025-09-30"
        assert previous_quarter_end("2025-09-30") == "2025-06-30"
        assert previous_quarter_end("2025-06-30") == "2025-03-31"

    def test_year_boundary(self):
        assert previous_quarter_end("2025-03-31") == "2024-12-31"


PREDS = pd.DataFrame(
    {
        "tic": ["A", "B", "C", "D", "E", "F"],
        "bucket": ["growth_tech"] * 3 + ["cyclical"] * 3,
        "predicted_return": [0.05, 0.03, 0.01, 0.08, 0.02, 0.09],
    }
)


class TestMLBucketStrategy:
    def test_cache_hit_skips_subprocess_and_selects_top_n_per_bucket(self, tmp_path):
        out_dir = tmp_path / "2025-12-31"
        out_dir.mkdir(parents=True)
        PREDS.to_csv(out_dir / "sp500_ml_bucket_predictions_x.csv", index=False)

        strategy = MLBucketStrategy(
            StrategyConfig(name="test"), db_path=tmp_path / "db.sqlite", top_n_per_bucket=2, work_dir=tmp_path
        )
        with patch("app.strategy.ml_bucket_strategy.subprocess.run") as mock_run:
            result = strategy.generate_weights({}, target_date="2025-12-31")

        mock_run.assert_not_called()
        # top 2 per bucket by predicted_return: growth_tech {A(.05),B(.03)}, cyclical {D(.08),F(.09)}
        assert set(result.weights["tic"]) == {"A", "B", "D", "F"}
        assert (result.weights["weight"] == 0.25).all()
        assert result.metadata["val_cutoff"] == "2025-09-30"

    def test_cache_miss_invokes_subprocess_with_correct_args(self, tmp_path):
        strategy = MLBucketStrategy(
            StrategyConfig(name="test"), db_path=tmp_path / "db.sqlite", top_n_per_bucket=2, work_dir=tmp_path
        )

        def fake_subprocess_run(cmd, **kwargs):
            # simulate ml_bucket_selection.py writing its output file
            out_dir = tmp_path / "2025-12-31"
            PREDS.to_csv(out_dir / "sp500_ml_bucket_predictions_x.csv", index=False)
            return MagicMock(returncode=0, stderr="")

        with patch("app.strategy.ml_bucket_strategy.subprocess.run", side_effect=fake_subprocess_run) as mock_run:
            result = strategy.generate_weights({}, target_date="2025-12-31")

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert called_args[called_args.index("--val-cutoff") + 1] == "2025-09-30"
        assert called_args[called_args.index("--infer-date") + 1] == "2025-12-31"
        assert called_args[called_args.index("--db") + 1] == str(tmp_path / "db.sqlite")
        assert set(result.weights["tic"]) == {"A", "B", "D", "F"}

    def test_missing_target_date_raises(self, tmp_path):
        strategy = MLBucketStrategy(StrategyConfig(name="test"), db_path=tmp_path / "db.sqlite")
        with pytest.raises(ValueError, match="target_date"):
            strategy.generate_weights({}, target_date=None)

    def test_no_output_produced_returns_empty_with_reason(self, tmp_path):
        strategy = MLBucketStrategy(
            StrategyConfig(name="test"), db_path=tmp_path / "db.sqlite", work_dir=tmp_path
        )
        with patch("app.strategy.ml_bucket_strategy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="boom")
            result = strategy.generate_weights({}, target_date="2025-12-31")

        assert result.weights.empty
        assert list(result.weights.columns) == ["tic", "weight"]
        assert result.metadata["n_selected"] == 0
        assert "reason" in result.metadata
