# Execution-Aware Backtest

Execution-Aware Backtest is a portfolio-level execution simulation prototype. The project currently provides its repository foundation and a strict loader for synthetic hourly OHLCV CSV data, but no backtest engine.

The planned timing convention is to generate signals at a bar close and permit execution only at the next bar open. Same-bar close execution and execution of final-bar signals are non-goals.

## Setup

Python 3.11 or newer is required. From the repository root, use the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -c "import backtest; print(backtest.__version__)"
.\.venv\Scripts\python.exe -m pytest -q
```

## Repository structure

- `src/backtest/`: installable Python package and strict OHLCV CSV loader
- `tests/fixtures/`: deterministic synthetic test data
- `config/`: future scenario configuration
- `data/raw/`, `data/processed/`, `data/metadata/`: future local data organization
- `notebooks/`: optional exploration, not source-of-truth logic
- `scripts/`: future task-specific scripts
- `results/`: future generated results
- `artifacts/codex_review/`: ignored local review evidence

`tests/fixtures/simple_bars.csv` is synthetic and does not reproduce real market data.

## Data validation

`backtest.data.load_bars_csv` preserves CSV row order and returns immutable bars with UTC-aware timestamps and floating-point OHLCV values. It rejects malformed schemas, invalid OHLCV values, non-chronological or duplicate timestamps, and timestamps that are not exactly one hour apart. Missing bars are rejected rather than sorted, forward-filled, inferred, or repaired.

## Not implemented

Strategies, signals, orders, an event hierarchy, fills, execution logic, portfolio accounting, an engine loop, metrics, configuration loading, a command-line interface, notebooks, market-data ingestion, and backtest results are not implemented. Production readiness and profitability assessment are outside the current scope.
