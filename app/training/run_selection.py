"""Runs ml_bucket_selection.py in mixed-vintage mode (per ticker: use the
latest available quarter -- see ML_STOCK_SELECTION.md) and logs its
outputs to MLflow as a tracked run.

Per the Reproducibility decision (PROJECT_PLAN.md), invokes the vendored
script as a subprocess rather than reimplementing its logic -- same
pattern app/strategy/ml_bucket_strategy.py uses for backtesting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from app.config.settings import get_mlflow_tracking_uri
from app.config.vendor_path import VENDOR_FINRL_ROOT

ML_BUCKET_SCRIPT = VENDOR_FINRL_ROOT / "src" / "strategies" / "ml_bucket_selection.py"


def run_selection(db_path: Path, val_cutoff: str, work_dir: Path | None = None) -> str:
    """Runs ml_bucket_selection.py --mixed-vintage against db_path, logs its
    outputs to MLflow, returns the MLflow run ID.

    Relies on the ambient MLflow tracking URI (set by the caller, e.g.
    main() via get_mlflow_tracking_uri(), or a test) rather than setting
    one itself -- same convention as app/training/train.py (retired, but
    the pattern's still the right one).
    """
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ml_bucket_selection_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, str(ML_BUCKET_SCRIPT),
            "--val-cutoff", val_cutoff,
            "--mixed-vintage",
            "--db", str(db_path),
            "--output-dir", str(work_dir) + "/",
        ],
        capture_output=True,
        text=True,
        check=False,  # exit code can be nonzero even on a successful run
        # (e.g. the openpyxl-dependent Excel export failing after everything
        # that matters already saved) -- we inspect the output files ourselves.
    )

    pred_files = sorted(work_dir.glob("sp500_ml_bucket_predictions_*.csv"))
    if not pred_files:
        raise RuntimeError(
            f"ml_bucket_selection.py produced no predictions (exit {result.returncode}): {result.stderr[-2000:]}"
        )
    model_result_files = sorted(work_dir.glob("sp500_ml_bucket_model_results_*.csv"))
    importance_files = sorted(work_dir.glob("sp500_ml_feature_importance_*.csv"))

    predictions = pd.read_csv(pred_files[-1])

    mlflow.set_experiment("ml-bucket-selection")
    with mlflow.start_run() as run:
        mlflow.log_params({"val_cutoff": val_cutoff, "mixed_vintage": True, "db_path": str(db_path)})
        mlflow.log_metric("n_stocks_ranked", len(predictions))
        mlflow.log_metric("n_buckets", predictions["bucket"].nunique())

        if model_result_files:
            model_results = pd.read_csv(model_result_files[-1])
            best_per_bucket = model_results.loc[model_results.groupby("bucket")["val_mse"].idxmin()]
            for _, row in best_per_bucket.iterrows():
                mlflow.log_metric(f"{row['bucket']}_best_val_mse", row["val_mse"])
                mlflow.log_param(f"{row['bucket']}_best_model", row["model"])

        mlflow.log_artifact(str(pred_files[-1]), artifact_path="predictions")
        if importance_files:
            mlflow.log_artifact(str(importance_files[-1]), artifact_path="feature_importance")
        if model_result_files:
            mlflow.log_artifact(str(model_result_files[-1]), artifact_path="model_results")

        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ml_bucket_selection.py (mixed-vintage) and log to MLflow")
    parser.add_argument("--db", required=True)
    parser.add_argument("--val-cutoff", required=True, help="last quarter-end whose y_return is fully realized")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    run_id = run_selection(
        Path(args.db), args.val_cutoff, work_dir=Path(args.work_dir) if args.work_dir else None
    )
    print(f"Logged MLflow run: {run_id}")


if __name__ == "__main__":
    main()
