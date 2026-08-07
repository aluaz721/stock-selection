# Training/inference image for the ML stock selection pipeline.
#
# No FastAPI service exists yet (that's Phase 6's fuller-tier stretch, not
# built) -- this packages the Python pipeline (build the local fundamentals
# DB, run selection, build frontend data, run tests/backtests) to run
# on-demand via `docker run`, not as a long-lived server.
#
# vendor/FinRL-Trading is a git submodule: it must already be checked out
# (`git submodule update --init`) in the build context before `docker
# build` runs.

FROM python:3.12-slim

WORKDIR /app

# build-essential: some deps (e.g. bt, lightgbm) may need to compile from
# source depending on platform/wheel availability. libgomp1: lightgbm's
# runtime OpenMP dependency, not always present on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# No default long-running process. Examples:
#   docker run <image> python -m app.data.build_fundamentals_db
#   docker run <image> python -m app.training.run_selection --db data/finrl_trading.db --val-cutoff 2025-09-30
#   docker run <image> python -m pytest tests/
CMD ["python", "-c", "print('Specify a command, e.g.: python -m app.training.run_selection --db data/finrl_trading.db --val-cutoff 2025-09-30')"]
