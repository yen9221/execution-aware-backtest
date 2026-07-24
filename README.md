# Execution-Aware Backtest

Execution-Aware Backtest is a portfolio-level execution simulation prototype. The project currently provides strict synthetic OHLCV loading, price-free CASH/LONG target intents, a minimal deterministic next-bar execution loop, quantity-based fills, and immutable single-asset long/cash accounting. It is not a production backtester.

The planned timing convention is to generate signals at a bar close and permit execution only at the next bar open. Same-bar close execution and execution of final-bar signals are non-goals.

## Setup

Python 3.11 or newer is required. From the repository root, use the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -c "import backtest; print(backtest.__version__)"
.\.venv\Scripts\python.exe -m pytest -q
```

## Repository structure

- `src/backtest/`: strict OHLCV loading, target-to-order conversion, a minimal bar loop, fill mathematics, and portfolio state transitions
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

## Execution mathematics

Quantity-based market orders are represented by immutable records. `backtest.execution.execute_market_order` deterministically calculates a fill from a supplied next-bar open reference price. Directional slippage and proportional fees are retained separately: buys receive a higher fill price and negative cash flow, while sells receive a lower fill price and positive cash flow. The function does not choose orders, enforce bar timing, or read or update portfolio cash and positions.

## Target positioning

`backtest.positioning.PendingTarget` represents an already-determined CASH or LONG target independently from any model output. It is created after a completed decision bar and stores only that bar's timestamp and the target; it contains no future price or quantity. `market_order_for_target_at_open` converts the intent into a quantity-based market order, or `None`, only when the next execution-bar open is observable. All-in buy affordability includes adverse directional slippage and the proportional fee, with a one-float-step conservative sizing buffer. Repeated CASH or LONG targets produce no trade, and an existing long position is not rebalanced. The positioning layer itself contains no model, threshold, strategy, fill execution, or portfolio mutation.

## Minimal engine

`backtest.engine.run_backtest` consumes one precomputed CASH or LONG target for each completed chronological bar. A target created from bar `t` can execute only at bar `t+1` open; the final bar's target is retained as unexecuted because no later open exists. The engine delegates target conversion, fill calculation, and portfolio accounting to their existing modules, records real fills only, and stores one immutable end-of-bar portfolio snapshot per bar. Portfolio value is marked as cash plus long quantity times that bar's close and is only a state observation, not a performance conclusion. The loop does not perform prediction, threshold selection, strategy generation, performance analysis, or output generation.

## Portfolio accounting

`backtest.portfolio.apply_fill` returns a new immutable single-asset portfolio state after applying a fill to cash, long position quantity, and cumulative fees. It rejects insufficient cash and insufficient long position; leverage, margin, and short selling are not supported. Fill prices, fees, and slippage are accepted from the execution layer rather than recalculated.

## Not implemented

Models, prediction-to-target policy, thresholds, strategies, signals, pending-order queues, PnL and other performance metrics, configuration loading, CSV output, a command-line interface, notebooks, market-data ingestion, and production backtest behavior are not implemented. Production readiness and profitability assessment are outside the current scope.
