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
    "(2 real market crashes included, not just a bull run)."
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
    "Fundamentals are quarterly, filed on scattered dates during earnings season."
)
