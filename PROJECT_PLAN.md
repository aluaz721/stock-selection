# ML Stock Selection MLOps Pipeline — Project Plan

## Summary

Replicate FinRL-Trading's own per-sector-bucket ML stock selection
strategy (`ml_bucket_selection.py` — a RandomForest/LightGBM/
HistGradientBoosting/ExtraTrees/Ridge/Stacking ensemble predicting
quarterly forward returns from 52 fundamental factors plus momentum, with
point-in-time S&P 500 membership) as close to its original implementation
as possible, then wrap it in a novel MLOps loop: scheduled retrain-and-rank
→ tracking → validation → containerized deployment → CI/CD → a
live-updating public frontend showing today's picks and rolling
performance vs. the S&P 500.

This project is **independent from Aquila** (the C++ limit order book engine)
— no shared code or integration between the two.

Built as a layer on top of **FinRL-Trading**
(`AI4Finance-Foundation/FinRL-Trading`), vendored as a git submodule rather
than pip-installed (see Vendoring decision below). Phases 1-3 originally
built a custom LightGBM classifier on NASDAQ-100 technical features as a
"signal filter" — 52% test accuracy, no demonstrated edge, and a backtest
universe later found to have real survivorship bias. **This plan
supersedes that approach.** FinRL-Trading already ships a more
sophisticated, fundamentals-driven strategy
(`src/strategies/ml_bucket_selection.py`) with its own bundled historical
dataset and documentation (`ML_STOCK_SELECTION.md`) — verified working by
actually running it end-to-end against the vendored data. The novelty here
is entirely the MLOps layer wrapped around that strategy, not the model
itself — see the Reproducibility decision below. The retired Phase 1-3
code isn't deleted, just no longer the active build target.

## Explicit scope decisions

- **Model: replicate, don't reimplement.** Use FinRL-Trading's own
  `src/strategies/ml_bucket_selection.py` as close to unmodified as
  possible — a per-GICS-sector-bucket (`growth_tech` / `cyclical` /
  `real_assets` / `defensive`) ensemble of 6-7 regressors (RandomForest,
  LightGBM, HistGradientBoosting, ExtraTrees, Ridge, Stacking, optionally
  XGBoost if installed) predicting quarterly forward log-return from 52
  fundamental factors plus momentum features, picked per bucket by
  validation MSE. Verified working this session: built a local SQLite DB
  from the vendored `fundamental_data_full.csv` and ran the script
  directly for a real backtest date — it produced sensible, differentiated
  per-bucket rankings (e.g. materials/mining names topping `real_assets`,
  healthcare topping `defensive`) with real MSE values, using only
  already-installed dependencies (`scikit-learn`, `lightgbm`; `xgboost` is
  a graceful optional import, currently skipped).
- **Reproducibility over reimplementation.** The point of this pivot is to
  plug a *proven, already-implemented* strategy into a novel MLOps
  pipeline, not to prove a from-scratch model works. Concretely: prefer
  invoking FinRL-Trading's own scripts (`ml_bucket_selection.py`,
  `fetch_and_store_fundamentals.py`, `backfill_historical_sp500.py`,
  `fix_adj_close.py`, `fill_recent_yreturn.py`) directly — as subprocesses
  or direct module calls — over porting their logic into our own code.
  New code is limited to orchestration/tracking/serving glue: an
  MLflow-logging wrapper around the script's outputs, a `BaseStrategy`
  adapter for backtesting (the vendored script isn't `BaseStrategy`-shaped
  itself, so *some* new glue is unavoidable there), and the
  scheduling/serving layer.
- **Universe: S&P 500, not NASDAQ-100.** Matches the bundled data
  (`vendor/FinRL-Trading/data/sp500_historical_constituents.csv`,
  `fundamental_data_full.csv`) and `ml_bucket_selection.py`'s default. The
  NASDAQ-100 snapshot and yfinance price pipeline from Phases 1-3 are
  retired for the model itself.
- **Bundled historical data eliminates FMP for backtesting/initial
  training.** `vendor/FinRL-Trading/data/fundamental_data_full.csv`
  (22,909 records, 715 tickers, 64 columns, 2015-Q2 → 2026-Q1) and
  `sp500_historical_constituents.csv` (2,709 point-in-time daily
  membership snapshots, 1996-2026) are already bundled in the vendored
  repo — someone else already spent an FMP budget building this. Loading
  the CSV into a local SQLite DB and running `ml_bucket_selection.py`
  needs zero new API calls. **FMP re-enters the picture for live
  refresh** (next bullet) — a deliberate, new dependency for this phase,
  distinct from Phases 1-3's decision to avoid it.
- **Live refresh needs FMP — and the free tier doesn't cover it (confirmed,
  not a volume problem).** "Output new stocks every X" means extending
  the bundled dataset past Q1 2026 as new fundamentals get filed —
  exactly what FinRL-Trading's own `fetch_and_store_fundamentals.py` /
  `backfill_historical_sp500.py` / `fix_adj_close.py` /
  `fill_recent_yreturn.py` do, and they're built around FMP. The original
  assumption here was that a free key's 250-requests/day cap would be the
  constraint (fine for incremental refresh, tight for a full backfill).
  Tested with a real key: **that assumption was wrong.** The key
  authenticates fine and basic price/quote endpoints return 200, but all
  four fundamentals endpoints `fetch_and_store_fundamentals.py` needs —
  `income-statement`, `balance-sheet-statement`, `cash-flow-statement`,
  `ratios` — return **402 Payment Required** on the free tier, at any
  volume. This isn't a rate-limit issue to budget around; the free tier
  doesn't include these endpoints at all. A yfinance-based fundamentals
  adapter (mirroring the Phase 1 price adapter, mapping `.info`/
  `.financials`/`.balance_sheet`/`.cashflow` to the 52-factor schema) is
  no longer just a fallback — it's the realistic path unless a paid FMP
  plan is worth the cost. Revisit once that tradeoff is decided.
- **Retraining is scheduled, not drift-triggered** (supersedes the
  original plan's central mechanism). `ml_bucket_selection.py` always
  retrains from scratch on the full available window every time it's
  invoked — there's no persistent "Production model" sitting still
  between drift checks the way the original plan assumed. That changes
  what "drift-triggered retraining" can honestly mean here: there's no
  natural trigger to gate *whether* to retrain. What's left of drift
  monitoring becomes an *observability* layer (rolling validation MSE,
  realized-return tracking of past picks, feature-distribution checks)
  surfaced on the frontend/MLflow rather than something that decides
  whether a retrain happens — demoted to a stretch item (Phase 4).
  Flagging this clearly since it's a real change from the original
  premise, not a silent scope cut — worth confirming you're fine with it.
- **Retrain/inference cadence: daily, not hourly.** Fundamentals are
  quarterly, filed on scattered dates during earnings season — an hourly
  cron would mostly re-output identical rankings and burn FMP quota for
  nothing. Daily matches `ml_bucket_selection.py`'s existing
  `--mixed-vintage` mode (per ticker: use the latest available quarter),
  which naturally picks up newly-filed fundamentals as they land. State
  this plainly on the frontend — picks staying the same for stretches at
  a time is expected behavior, not a broken pipeline.
- **Kubernetes**: still explicitly out of scope — single model, no
  traffic to scale, Docker Compose is the honest footprint.
- **Prometheus/Grafana**: still optional stretch, not core, per the
  original reasoning — the real monitoring signal is model/portfolio
  metrics, domain-specific tools would just visualize them.
- **A/B testing**: yes, but the comparison target shifts with the model —
  new run's picks vs. the previous run's, and the ranked-portfolio
  strategy vs. an equal-weight S&P 500 baseline and SPY itself, evaluated
  through FinRL-Trading's `BacktestEngine` (Phase 2's integration carries
  over, pointed at the new strategy wrapper instead of the retired
  classifier). Framed as directional comparisons, not significance-tested
  results, per the original reasoning.
- **Orchestration**: GitHub Actions cron, not Airflow — still holds, one
  daily-cadence chain doesn't justify a webserver/scheduler stack.
- **This is a portfolio piece with a deadline**: unchanged — reaching a
  live, demonstrable link still beats a fuller architecture no one sees.
  This pivot is *in service of* that: a proven strategy means less time
  spent chasing signal, more time on the MLOps loop and frontend that are
  actually the resume story.
- **Vendoring, not pip-installing, FinRL-Trading**: unchanged (the
  published PyPI package is broken; git submodule + `PYTHONPATH` is the
  fix — see original research). This pivot leans on the vendored checkout
  even more directly than before.
- **Our own source root is `app/`, not `src/`**: unchanged, same
  collision-avoidance reasoning as before (FinRL-Trading's checkout has
  its own top-level `src/__init__.py`).

## Repo structure

```
drift-retrain-trading/
├── README.md
├── requirements.txt              # our code's deps + FinRL-Trading's vendored
│                                  #   source's, phased in per project phase
├── .env.example                  # MLflow tracking URI, DB creds, FMP_API_KEY
│                                  #   (now required — see Live refresh decision)
├── docker-compose.yml            # mlflow server + postgres
├── Dockerfile                    # training/serving image
├── .github/
│   └── workflows/
│       ├── ci.yml                # lint, unit tests, backtest validation on PR
│       └── daily-selection.yml   # scheduled cron: refresh data -> run
│                                  #   selection -> log to MLflow -> write
│                                  #   output for the frontend
│
├── vendor/
│   └── FinRL-Trading/             # git submodule, upstream untouched
│       ├── src/strategies/
│       │   └── ml_bucket_selection.py  # THE model -- run as close to
│       │                          #   unmodified as possible (see Model /
│       │                          #   Reproducibility decisions above)
│       ├── src/data/               # fetch_and_store_fundamentals.py,
│       │   │                       #   backfill_historical_sp500.py,
│       │   │                       #   fix_adj_close.py,
│       │   │                       #   fill_recent_yreturn.py -- run
│       │   │                       #   directly for live refresh
│       │   └── data_fetcher.py     # fetch_sp500_tickers, FMP-backed
│       ├── data/
│       │   ├── fundamental_data_full.csv        # bundled: 2015-Q2~2026-Q1,
│       │   │                                     #   715 tickers, 64 cols
│       │   └── sp500_historical_constituents.csv # bundled: point-in-time
│       │                                          #   membership, 1996-2026
│       └── ML_STOCK_SELECTION.md   # the original pipeline's own docs --
│                                  #   the spec this plan is replicating
│
├── app/                           # our source root (not `src/` -- see
│   │                              #   naming-collision decision above)
│   ├── config/
│   │   ├── vendor_path.py         # ensure_finrl_on_path(): puts
│   │   │                          #   vendor/FinRL-Trading/ on sys.path
│   │   └── settings.py            # paths, MLflow URI
│   │
│   ├── data/
│   │   ├── build_fundamentals_db.py  # loads fundamental_data_full.csv
│   │   │                          #   (or the refreshed version) into a
│   │   │                          #   local data/finrl_trading.db --
│   │   │                          #   OUTSIDE vendor/, keeps it untouched
│   │   └── refresh_fundamentals.py   # thin sequencer around FinRL-Trading's
│   │                              #   own fetch/backfill/fix scripts for
│   │                              #   the daily live-refresh step
│   │
│   ├── strategy/
│   │   └── ml_bucket_strategy.py  # BaseStrategy subclass: runs (or reads
│   │                              #   the output of) ml_bucket_selection.py
│   │                              #   for a target_date, converts per-bucket
│   │                              #   rankings into portfolio weights
│   │
│   ├── training/
│   │   └── run_selection.py       # invokes ml_bucket_selection.py, logs
│   │                              #   its predictions / feature-importance /
│   │                              #   per-bucket MSE to MLflow as a run
│   │
│   └── drift/                     # (stretch, Phase 4 -- see Retraining
│       ├── monitors.py            #   decision above: observability, not
│       │                          #   a retrain gate)
│       └── backtest_trigger.py
│
│   [retired, kept but not the active build target -- see Model decision]
│   ├── features/pipeline.py, features/labels.py   # NASDAQ-100 + yfinance
│   │                                                #   technical features
│   ├── strategy/model.py                           # the LightGBM classifier
│   └── strategy/rolling_selector_strategy.py       # BaseStrategy wrapper
│                                                    #   around that classifier
│
├── streamlit_app/
│   └── app.py                     # minimal-tier live frontend (Phase 6
│                                  #   target): today's top picks per
│                                  #   bucket + rolling strategy-vs-S&P 500
│                                  #   performance chart
│
├── api/                           # fuller-tier serving (Phase 6 stretch)
│   ├── main.py                    # FastAPI app
│   ├── routers/
│   │   └── predictions.py         # GET /picks/latest  GET /picks/history
│   │                              #   GET /metrics/performance
│   └── db.py                      # Supabase/Neon Postgres client
│
├── frontend/                      # fuller-tier serving (Phase 6 stretch)
│   ├── package.json               # React + Vite (or Next.js)
│   ├── src/
│   │   ├── App.tsx                # today's picks by bucket + rolling
│   │   │                          #   performance-vs-S&P 500 chart
│   │   └── api.ts                 # fetch wrapper for the FastAPI backend
│   └── vite.config.ts
│
├── backtests/
│   ├── run_backtest.py            # CLI: run BacktestEngine on
│   │                              #   ml_bucket_strategy (Phase 2)
│   └── compare_vs_benchmarks.py   # vs. equal-weight S&P 500 / SPY
│
├── data/
│   ├── finrl_trading.db           # local SQLite build of the fundamentals
│   │                              #   dataset (gitignored) -- built by
│   │                              #   app/data/build_fundamentals_db.py
│   └── predictions/               # committed JSON history (minimal-tier
│                                  #   serving path, see Phase 6)
│
├── notebooks/
│   ├── 01_explore_ml_bucket_selection.ipynb
│   └── 02_results_dashboard.ipynb
│
└── tests/
    ├── test_ml_bucket_strategy.py
    └── test_run_selection.py
```

## FinRL-Trading integration points

| Your file | FinRL-Trading dependency |
|---|---|
| `app/data/build_fundamentals_db.py` | `vendor/FinRL-Trading/data/fundamental_data_full.csv` (loaded, not imported) |
| `app/data/refresh_fundamentals.py` | `src/data/fetch_and_store_fundamentals.py`, `backfill_historical_sp500.py`, `fix_adj_close.py`, `fill_recent_yreturn.py` (invoked directly) |
| `app/training/run_selection.py` | `src/strategies/ml_bucket_selection.py` (invoked directly) |
| `app/strategy/ml_bucket_strategy.py` | `src.strategies.base_strategy.BaseStrategy`, `StrategyResult` |
| `backtests/run_backtest.py`, `compare_vs_benchmarks.py` | `src.backtest.backtest_engine.BacktestEngine`, `BacktestConfig` |
| (retired) `app/strategy/rolling_selector_strategy.py` | same `BaseStrategy` import, now unused in the active build |

Vendored as a git submodule at `vendor/FinRL-Trading/`, not pip-installed
(see Vendoring decision above). Every module that touches it calls
`app.config.vendor_path.ensure_finrl_on_path()` before importing anything
under `src.*` — that `src` refers to FinRL-Trading's own package, reachable
because its repo root is on `PYTHONPATH`, not to our own code (which lives
under `app/`). Per the Reproducibility decision, prefer running
`ml_bucket_selection.py` and the `src/data/*.py` scripts as-is (subprocess
or direct `main()` call) over importing their internals piecemeal — they
weren't written as an importable library, and refactoring them to be one
would mean editing the vendored source.

## Build phases

Supersedes the original plan's Phases 1-4; Phases 5-7 carry over with
updated content.

### Phase 1 — Validate the replicated strategy
- `app/data/build_fundamentals_db.py`: load the bundled
  `fundamental_data_full.csv` into a local `data/finrl_trading.db`.
- Run `ml_bucket_selection.py` directly for a couple of the documented
  backtest dates (e.g. `--val-cutoff 2025-09-30 --infer-date 2025-12-31`);
  confirm sensible, differentiated per-bucket rankings and real MSE values
  — already done once this session as a proof of concept, formalize into
  a repeatable, tested step.
- Sanity-check the point-in-time S&P 500 filter is genuinely active
  (per-bucket/per-date sample counts should shrink appropriately, not
  just use today's full membership retroactively).

### Phase 2 — Backtest via BaseStrategy/BacktestEngine
- `app/strategy/ml_bucket_strategy.py`: `BaseStrategy` subclass that
  converts a given date's per-bucket rankings into portfolio weights
  (e.g. top-N per bucket, equal- or inverse-MSE-weighted).
- Point `backtests/run_backtest.py` / `compare_vs_benchmarks.py` at this
  strategy instead of the retired classifier.
- This time, point-in-time membership means no survivorship bias (unlike
  the original Phase 3 backtest) — get a real read on whether the
  replicated strategy beats SPY / equal-weight S&P 500 over 2015-2026.

### Phase 3 — MLOps orchestration: scheduled retrain-and-rank
- **Done, tested**: `app/training/run_selection.py` invokes
  `ml_bucket_selection.py --mixed-vintage`, logs predictions,
  feature-importance, model-results, and per-bucket best-model/val-MSE to
  MLflow as a tracked run. Verified with a real run against the bundled
  DB (503 stocks, 4 buckets, real MLflow params/metrics/artifacts).
- **Done, structural**: `.github/workflows/daily-selection.yml` — the
  `build_fundamentals_db` → `run_selection` chain is real; the refresh
  step and frontend-output step are documented placeholders (see below).
- **Deferred by choice, not blocked**: `app/data/refresh_fundamentals.py`
  isn't built. A real FMP key is configured (`.env`, gitignored) and does
  authenticate, but its free tier returns 402 Payment Required on every
  fundamentals endpoint `fetch_and_store_fundamentals.py` needs (see Live
  refresh decision above) — confirmed with a live call, not assumed. The
  realistic path is a yfinance-based fundamentals adapter or a paid FMP
  plan; deliberately deferred in favor of Phase 4/5/6, which don't need
  fresher-than-Q1-2026 data to demonstrate the full MLOps loop. Revisit
  when live refresh actually matters more than shipping the rest.
- **Blocked on Phase 6 design**: writing predictions to `data/predictions/`
  in whatever shape the frontend consumes.
- **Deferred**: MLflow tracking is currently ephemeral in CI (no
  persistent backend yet) and `--val-cutoff` is hardcoded rather than
  auto-advancing each quarter — both depend on work not yet done (Phase 5
  deployment; a "which quarter is fully realized" helper). Noted directly
  in the workflow file.

### Phase 4 — Monitoring (stretch, demoted from the original plan's core)
- `app/drift/monitors.py`: rolling validation MSE per bucket over time,
  realized-return tracking of past picks vs. what actually happened —
  observability, not a retraining gate (see Retraining decision above).
- Optional Prometheus/Grafana visualization of the above.

### Phase 5 — Deployment
- `Dockerfile` for training/serving.
- `docker-compose.yml`: MLflow server + Postgres backend (tracking store
  only).
- `.github/workflows/ci.yml`: lint, unit tests, backtest-validation run on
  PRs touching strategy/selection code.

### Phase 6 — Live prediction serving
Portfolio-piece-with-a-deadline framing unchanged: minimal tier is the
target, fuller tier is a stretch upgrade after something is live.

**Minimal tier (target) — done, verified**:
- `app/reporting/portfolio_history.py`: portfolio cumulative-return math,
  computed directly from prices + weight_signals rather than through `bt`
  (see the known-risk entry on `bt`'s Sharpe bugs) — tested offline.
- `app/reporting/build_frontend_data.py`: backfill/seed generator (not the
  live daily job yet — that's still `.github/workflows/daily-selection.yml`'s
  deferred pieces) that writes `data/predictions/history.json` (the
  verified 2017-2025 top-3/bucket backtest vs. SPY/QQQ, 2306 trading days)
  and `data/predictions/latest.json` (a real mixed-vintage run — 425
  tickers on Q4 2025 data, 78 already on Q1 2026, transparently labeled).
- `streamlit_app/app.py`: renders both — performance metrics, the
  strategy-vs-benchmark chart, today's top-3-per-bucket picks, and a
  methodology/limitations expander that states the drawdown weakness and
  the frozen-data caveat plainly rather than burying them.
- Verified with `streamlit.testing.v1.AppTest` (no headless browser
  available — Chromium's download risked the disk budget at 441MB free):
  zero exceptions, all 3 metrics/4 bucket columns/expander render with
  correct values matching the JSON inputs exactly.
- Not yet done: deploying to Streamlit Community Cloud (needs a GitHub
  push, not done this session) and wiring daily regeneration into the
  scheduled workflow (needs live refresh unblocked first).

**Fuller tier (stretch, post-deadline)**:
- Same cron job, writes to Supabase/Neon Postgres instead of committed
  JSON — also a place to log realized outcomes for performance tracking.
- `api/`: FastAPI app exposing `/picks/latest`, `/picks/history`,
  `/metrics/performance`. Deploy on Render or Fly.io free tier.
- `frontend/`: React (Vite) or Next.js app on Vercel/Netlify.

Frame the page around today's picks + rolling performance vs. the S&P
500, and say plainly that picks update daily but often stay the same for
stretches (fundamentals are quarterly) — that's what ties it honestly to
the pipeline underneath instead of reading as broken.

### Phase 7 — Polish
- README with architecture diagram, explaining the FinRL-Trading
  dependency boundary, what's original vs. replicated, and what's custom.
- Link the live frontend URL and repo in the README.
- Notebooks cleaned up as a walkthrough (strategy exploration, results
  dashboard) — useful for your own iteration and as an interview reference.

## Known risks to watch for

- **Model choice is a replicated strategy, not a claimed novel edge** —
  the resume story is the MLOps system (scheduled retraining, tracked,
  validated, served), wrapped around FinRL-Trading's own implementation.
  Phase 1 confirmed the pipeline runs and produces sensible rankings;
  Phase 2's backtest (see below) is what actually answers whether it
  beats the S&P 500. Keep that framing in the README.
- **`bt`'s Sharpe/Sortino cannot be trusted anywhere without independent
  verification (confirmed bug, broader than first thought; vendor stays
  untouched per the Reproducibility decision)**: first caught on
  `BacktestEngine`'s buy-and-hold benchmark path (`_get_benchmark_metrics`,
  SPY/QQQ) -- Sharpe ratios of 5.5 and 10.3 over an 8-quarter backtest,
  not plausible for any real index. Initially assumed isolated to that
  buy-and-hold benchmark shape, since the main strategy's own `RunOnDate`
  Sharpe looked sane by comparison on that run. **That assumption was
  wrong**: a later run hit the identical failure mode on `equal_weight` --
  a normal `RunOnDate`-rebalanced strategy, same code path as
  `ml_bucket_selector` -- reporting Sharpe 6.89 over 2022-2025 (implying
  ~1.7% annualized volatility for a ~500-stock portfolio, impossible).
  Root cause not fully isolated; number of positions may be a factor
  (equal_weight holds ~500 names/quarter vs. top3/top5's 12-20), but
  don't assume that's the whole story. **Never trust
  `result.metrics['sharpe_ratio']` or `result.benchmark_metrics[bm]
  ['sharpe_ratio']` without cross-checking** -- compute portfolio returns
  directly from prices + weight_signals (shift weights by one day, multiply
  by that day's per-ticker return, sum) and derive Sharpe as
  `mean*252 / (std*sqrt(252))` from that series instead. Return and
  max_drawdown from `bt`/`BacktestEngine` have matched independent
  calculation closely every time this was checked -- only Sharpe/Sortino
  are suspect.
- **yfinance ticker format**: FinRL-Trading's bundled DB uses dotted class
  shares (`BRK.B`), yfinance expects dashes (`BRK-B`) -- confirmed via a
  real failure (`app/data/fetch_prices.py`'s `to_yfinance_symbol` now
  handles this). Any other ticker-notation mismatch between the DB and a
  live data source is a similar risk to watch for.
- **Corporate actions silently shrink the tradeable universe over time**:
  of ~500 S&P 500 tickers in a typical quarter, ~15-20 reliably fail to
  download from yfinance ("possibly delisted") -- mergers, acquisitions,
  and take-privates that happened between the bundled DB's snapshot and
  today (e.g. Electronic Arts' 2025 take-private deal crashed an early
  Phase 2 backtest with a NaN-price allocation error before
  `drop_tickers_with_price_gaps` was added). This is handled defensively
  now (affected tickers are dropped from that backtest's investable
  universe, weights renormalize among the rest) but will need a real
  answer before Phase 3's live inference -- a stock ml_bucket_selection.py
  ranks highly needs to actually be tradeable.
- **Time-ordered splits only** — no random shuffling anywhere in
  train/val/test, or you'll leak future information into training. This
  is already how `ml_bucket_selection.py` is built; don't undo it if you
  touch the script.
- **Survivorship bias — fixed for this path, not universally.** The
  original Phase 3 backtest (NASDAQ-100 + a static current-day snapshot
  applied retroactively) had confirmed survivorship bias — a random
  30-name sample, equal-weighted with no model, returned 57.7% annualized
  vs. QQQ's 22.6%, and included CoreWeave (IPO'd March 2025, mid-backtest).
  `ml_bucket_selection.py`'s point-in-time S&P 500 filter (verified wired
  into the actual code path, not just documented) fixes this for the new
  strategy. If NASDAQ-100/yfinance data ever gets reused for anything,
  the original bias risk applies again there.
- **FMP dependency reintroduced, deliberately.** Phases 1-3 avoided FMP
  entirely (metered free tier, "Legacy"-tagged endpoints with unconfirmed
  free-tier status). This plan reintroduces it specifically for daily
  live refresh, on the bet that *incremental* updates to ~500 tickers'
  changed fundamentals fit the 250-requests/day free tier better than a
  full historical backfill would have. Confirm this in Phase 3 before
  assuming it's fine — if it isn't, fall back to a yfinance fundamentals
  adapter (bounded, proven-tractable work per the Phase 1 price adapter).
- **Reproducibility drift risk**: the vendored submodule is pinned to a
  fixed commit, so `ml_bucket_selection.py`'s behavior won't silently
  change underneath us — but if the submodule is ever updated
  (`git submodule update --remote`), re-verify Phase 1's proof-of-concept
  run still produces sane output before trusting it again.
- **Cadence honesty**: fundamentals update quarterly on scattered filing
  dates, not continuously — a daily cron will often output identical
  rankings for days or weeks at a stretch. State this on the frontend
  explicitly (see Phase 6) so it reads as expected behavior, not as a
  stalled pipeline.
- **`openpyxl` gap (trivial, noted for later)**: `ml_bucket_selection.py`'s
  optional Excel export failed in testing with `ModuleNotFoundError: No
  module named 'openpyxl'` — all three CSV outputs (predictions,
  feature-importance, model-results) still saved successfully beforehand,
  so this isn't blocking. Install `openpyxl` only if the Excel export is
  actually wanted; the CSVs are sufficient for MLflow logging and the
  frontend.
- **Disk space is still tight** (~600MB free as of this session) — same
  constraint as the original plan. Check before adding any new dependency
  (e.g. `openpyxl`, `xgboost` if you want the full 7-model lineup).
