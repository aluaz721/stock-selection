from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd

from app.training.run_selection import run_selection


class TestRunSelection:
    def test_logs_expected_mlflow_run(self, tmp_path):
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        preds = pd.DataFrame(
            {
                "tic": ["A", "B", "C"],
                "bucket": ["growth_tech", "growth_tech", "cyclical"],
                "predicted_return": [0.05, 0.02, 0.01],
            }
        )
        preds.to_csv(work_dir / "sp500_ml_bucket_predictions_x.csv", index=False)
        model_results = pd.DataFrame(
            {
                "bucket": ["growth_tech", "growth_tech", "cyclical"],
                "model": ["RF", "Ridge", "Stacking"],
                "val_mse": [0.05, 0.04, 0.02],
            }
        )
        model_results.to_csv(work_dir / "sp500_ml_bucket_model_results_x.csv", index=False)

        with patch("app.training.run_selection.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            run_id = run_selection(db_path=tmp_path / "db.sqlite", val_cutoff="2025-09-30", work_dir=work_dir)

        called_args = mock_run.call_args[0][0]
        assert called_args[called_args.index("--val-cutoff") + 1] == "2025-09-30"
        assert "--mixed-vintage" in called_args
        assert called_args[called_args.index("--db") + 1] == str(tmp_path / "db.sqlite")

        run = mlflow.get_run(run_id)
        assert run.data.params["val_cutoff"] == "2025-09-30"
        assert run.data.params["growth_tech_best_model"] == "Ridge"  # lowest val_mse for growth_tech
        assert run.data.metrics["n_stocks_ranked"] == 3.0
        assert run.data.metrics["n_buckets"] == 2.0
        assert run.data.metrics["growth_tech_best_val_mse"] == 0.04

        client = mlflow.tracking.MlflowClient()
        artifact_paths = {a.path for a in client.list_artifacts(run_id)}
        assert "predictions" in artifact_paths
        assert "model_results" in artifact_paths

    def test_raises_when_no_predictions_produced(self, tmp_path):
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        with patch("app.training.run_selection.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="boom")
            try:
                run_selection(db_path=tmp_path / "db.sqlite", val_cutoff="2025-09-30", work_dir=work_dir)
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "boom" in str(e)
