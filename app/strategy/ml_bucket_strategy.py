"""BaseStrategy subclass wrapping FinRL-Trading's own ml_bucket_selection.py.

Per the Reproducibility decision (PROJECT_PLAN.md): runs the vendored
script as a subprocess, unmodified, rather than porting its logic into our
own code -- it wasn't written as an importable library (everything lives
inside main()), and editing it would violate "vendor untouched."

Each call retrains the full per-bucket model competition (RF/LGBM/HistGBM/
ExtraTrees/Ridge/Stacking) for one quarter's point-in-time universe --
budget ~2 minutes per generate_weights() call (measured directly).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from app.config.vendor_path import VENDOR_FINRL_ROOT, ensure_finrl_on_path

ensure_finrl_on_path()

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult  # noqa: E402

ML_BUCKET_SCRIPT = VENDOR_FINRL_ROOT / "src" / "strategies" / "ml_bucket_selection.py"

QUARTER_ENDS = ["03-31", "06-30", "09-30", "12-31"]


def previous_quarter_end(datadate: str) -> str:
    """The quarter-end immediately before `datadate` (itself a quarter-end),
    matching ml_bucket_selection.py's own --val-cutoff/--infer-date
    backtesting convention (val-cutoff is always one quarter behind
    infer-date -- see ML_STOCK_SELECTION.md's Backtesting Guide table)."""
    year, mmdd = int(datadate[:4]), datadate[5:]
    idx = QUARTER_ENDS.index(mmdd)
    if idx == 0:
        return f"{year - 1}-12-31"
    return f"{year}-{QUARTER_ENDS[idx - 1]}"


class MLBucketStrategy(BaseStrategy):
    """target_date is a quarter-end datadate (e.g. "2025-12-31"), matching
    ml_bucket_selection.py's --infer-date."""

    def __init__(
        self,
        config: StrategyConfig,
        db_path: Path,
        top_n_per_bucket: int = 5,
        work_dir: Optional[Path] = None,
    ):
        super().__init__(config)
        self.db_path = Path(db_path)
        self.top_n_per_bucket = top_n_per_bucket
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ml_bucket_"))

    def generate_weights(
        self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None
    ) -> StrategyResult:
        if target_date is None:
            raise ValueError("MLBucketStrategy requires target_date (a quarter-end datadate)")

        infer_date = target_date
        val_cutoff = previous_quarter_end(infer_date)
        out_dir = self.work_dir / infer_date
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_files = sorted(out_dir.glob("sp500_ml_bucket_predictions_*.csv"))
        if not pred_files:
            # Each call retrains a full per-bucket model competition (~2
            # minutes) -- skip it if this quarter was already computed into
            # the same work_dir (e.g. a prior run that failed partway
            # through on a *different* quarter shouldn't force redoing
            # every quarter that already succeeded).
            result = subprocess.run(
                [
                    sys.executable, str(ML_BUCKET_SCRIPT),
                    "--val-cutoff", val_cutoff,
                    "--infer-date", infer_date,
                    "--db", str(self.db_path),
                    "--output-dir", str(out_dir) + "/",
                ],
                capture_output=True,
                text=True,
            )
            pred_files = sorted(out_dir.glob("sp500_ml_bucket_predictions_*.csv"))
        else:
            result = None

        if not pred_files:
            return StrategyResult(
                strategy_name=self.config.name,
                weights=pd.DataFrame(columns=["tic", "weight"]),
                metadata={
                    "as_of": infer_date,
                    "val_cutoff": val_cutoff,
                    "n_selected": 0,
                    "reason": f"ml_bucket_selection.py produced no output "
                    f"(exit {result.returncode}): {result.stderr[-500:]}",
                },
            )

        predictions = pd.read_csv(pred_files[-1])
        selected = predictions.groupby("bucket", group_keys=False)[predictions.columns].apply(
            lambda g: g.sort_values("predicted_return", ascending=False).head(self.top_n_per_bucket)
        )

        if selected.empty:
            weights = pd.DataFrame(columns=["tic", "weight"])
        else:
            weights = pd.DataFrame({"tic": selected["tic"].values, "weight": 1.0 / len(selected)})

        return StrategyResult(
            strategy_name=self.config.name,
            weights=weights,
            metadata={"as_of": infer_date, "val_cutoff": val_cutoff, "n_selected": len(weights)},
        )
