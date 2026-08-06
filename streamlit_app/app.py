"""Minimal-tier frontend (Phase 6): today's top picks per bucket + rolling
strategy-vs-S&P-500 performance. Reads data/predictions/{latest,history}.json
-- see app/reporting/build_frontend_data.py for how they're generated.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"

st.set_page_config(page_title="ML Stock Selection", page_icon="\U0001f4c8", layout="wide")

st.title("ML Stock Selection")
st.caption(
    "Replicates FinRL-Trading's per-sector-bucket ML stock selection strategy "
    "(RandomForest / LightGBM / HistGradientBoosting / ExtraTrees / Ridge / "
    "Stacking ensemble on 52 fundamental factors + momentum, with point-in-time "
    "S&P 500 membership) wrapped in a custom MLOps pipeline. **Not investment "
    "advice.**"
)

latest_path = PREDICTIONS_DIR / "latest.json"
history_path = PREDICTIONS_DIR / "history.json"

if not latest_path.exists() or not history_path.exists():
    st.error("No prediction data yet. Run `python -m app.reporting.build_frontend_data` first.")
    st.stop()

latest = json.loads(latest_path.read_text())
history = json.loads(history_path.read_text())

st.subheader("Backtested performance")
summary = history["summary"]
labels = {"strategy": "Strategy (top-3/bucket)", "spy": "SPY", "qqq": "QQQ"}
cols = st.columns(3)
for col, key in zip(cols, ["strategy", "spy", "qqq"]):
    s = summary[key]
    col.metric(labels[key], f"{s['annualized_return']:.1%} / yr", f"Sharpe {s['sharpe_ratio']:.2f}")
    col.caption(f"Max drawdown: {s['max_drawdown']:.1%}")

st.caption(
    f"Backtest window: {history['config']['start']} to {history['config']['end']} "
    "(2 real market crashes included, not just a bull run). Point-in-time S&P 500 "
    "membership throughout -- no survivorship bias. Returns and max drawdown come "
    "directly from prices and portfolio weights, computed independently of "
    "FinRL-Trading's `bt`-based backtest engine (see PROJECT_PLAN.md's known risks "
    "for why: `bt`'s own Sharpe/Sortino calculation has been caught reporting "
    "impossible values twice)."
)

chart_df = pd.DataFrame(history["series"])
chart_df["date"] = pd.to_datetime(chart_df["date"])
chart_df = chart_df.set_index("date")
chart_df.columns = ["Strategy", "SPY", "QQQ"]
st.line_chart(chart_df)

st.subheader("Latest picks")
vintage_note = ", ".join(f"{v} tickers on {k}" for k, v in latest["vintage_breakdown"].items())
st.caption(
    f"As of {latest['as_of']} (mixed vintage: {vintage_note} -- see "
    "ML_STOCK_SELECTION.md's mixed-vintage mode). Model trained through "
    f"{latest['val_cutoff']}."
)

bucket_order = ["growth_tech", "cyclical", "real_assets", "defensive"]
buckets = latest["buckets"]
bucket_cols = st.columns(len(bucket_order))
for col, bucket in zip(bucket_cols, bucket_order):
    with col:
        st.markdown(f"**{bucket.replace('_', ' ').title()}**")
        for pick in buckets.get(bucket, []):
            st.write(f"{pick['tic']}  `{pick['predicted_return']:+.1%}`")

st.caption(
    "Fundamentals are quarterly, filed on scattered dates during earnings season -- "
    "picks staying the same for stretches at a time is expected behavior, not a "
    "stalled pipeline."
)

with st.expander("Methodology & known limitations"):
    st.markdown(
        """
- **Model**: FinRL-Trading's own `ml_bucket_selection.py`
  (`AI4Finance-Foundation/FinRL-Trading`), run as close to unmodified as
  possible -- not a from-scratch reimplementation. See that repo's
  `ML_STOCK_SELECTION.md` for the original spec.
- **This is a portfolio/engineering project, not investment advice.** The
  headline numbers above beat SPY and QQQ on both return and Sharpe over the
  backtest window, in aggregate and in each half of it independently checked
  -- but every backtest number here validates a *pipeline*, and 9 years is
  still one historical sample, not a guarantee of future performance.
- **Known weak point**: this strategy's max drawdown is consistently *worse*
  than the index's, in both halves of the backtest independently -- a
  concentrated, sector-bucketed book takes broad market crashes harder than
  a diversified index does, even though it recovers and wins on return.
- **Live data is currently frozen at the bundled dataset's snapshot**
  (through Q1 2026) -- daily refresh needs either a paid Financial Modeling
  Prep plan or a custom data adapter, neither built yet. "Latest picks"
  above reflect the most current data available, not necessarily today.

Full methodology, every bug found and fixed along the way, and all scope
decisions are documented in `PROJECT_PLAN.md` in the repo.
        """
    )
