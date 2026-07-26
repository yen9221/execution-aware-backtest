# Execution-Aware Backtest

Execution-Aware Backtest is a portfolio-level execution simulation prototype. The project currently provides strict synthetic OHLCV loading, a deterministic baseline target rule, price-free CASH/LONG target intents, a minimal deterministic next-bar execution loop, quantity-based fills, immutable single-asset long/cash accounting, and a separate deterministic reporting layer. It is not a production backtester.

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

The positioning layer also defines immutable single-asset `TargetWeight` values constrained to `[0.0, 1.0]`: `0.0` represents fully cash, `1.0` represents fully long, and intermediate values represent intended partial long exposure. `PendingTargetWeight` retains a price-free continuous target until the execution-bar open is observable. `market_order_for_target_weight_at_open` then values the existing position and the pre-trade portfolio at that open and can return a partial BUY or SELL delta order. It does not execute the order, create a fill, or mutate the portfolio; the existing execution and portfolio layers remain responsible for fills and account changes.

Continuous target-weight sizing retains the pre-trade reference-open allocation definition and calculates ideal exposure quantity from that open. For BUY orders, adverse slippage and the proportional fee are included in the effective unit cash cost used to cap quantity to affordability. SELL quantity remains the reference-open exposure difference, capped by the existing position. The execution and portfolio layers still determine actual fills, fees, cash flows, and account changes.

Because transaction costs reduce portfolio value, realized post-cost weight may differ from the intended target; no implicit post-cost exact-target solver is used. Exact target matching is retained only when both rates are zero.
Existing binary TargetPosition conversion and engine behavior remain unchanged, and fractional weights are never silently coerced into that binary path.

Continuous positioning also accepts an explicit absolute rebalance tolerance. Current weight is calculated from pre-trade portfolio value marked at the execution open, and no order is generated when the absolute current-to-target deviation is less than or equal to the tolerance. A triggered order still sizes toward the exact intended target; the tolerance does not change fee, slippage, affordability, or SELL sizing. A zero tolerance retains the prior exact-match-only behavior.

After tolerance and final cost-aware quantity sizing, continuous positioning applies a required non-negative minimum expected trade notional. BUY expected notional uses the adverse slipped BUY price, while SELL expected notional uses the adverse slipped SELL price. Fees and cash flow are excluded. Expected notional below the minimum is suppressed; equality is allowed, and a zero minimum retains prior behavior. Skipped orders are not accumulated. No post-cost exact-target solver is implemented, and binary behavior remains unchanged.

## Deterministic baseline strategy

`backtest.strategy.previous_close_momentum_targets` is a minimal one-bar close-momentum baseline used to validate the strategy-to-execution boundary. Its first completed bar maps to CASH. Each later completed bar maps to LONG only when its close is higher than the preceding close; a flat or falling close maps to CASH. Targets are created only after their bars complete and can execute only at the next bar open through the existing engine. Allocation remains binary fully invested long or cash.

`backtest.strategy.previous_close_momentum_target_weights` is the deterministic fractional counterpart. Its first completed bar maps to weight `0.0`; a later rising close maps to `0.75`, a flat close to `0.50`, and a falling close to `0.25`. It compares only the current and immediately preceding completed closes and returns precomputed immutable target weights. It does not inspect portfolio state, costs, tolerance, or minimum notional and does not execute orders. Execution remains the separate fractional engine's responsibility.

These fixed rules have no probability mapping, threshold selection, parameter fitting, or optimization. They are workflow baselines, not evidence of alpha, benchmark superiority, robust profitability, or an optimized allocation policy.

## Probability allocation policy

`backtest.allocation.probabilities_to_target_weights` converts already-generated probabilities into immutable `TargetWeight` values using a fixed three-region rule. Probabilities below the required lower threshold map to `0.0`, probabilities from the lower threshold through the upper threshold inclusive map to `0.5`, and probabilities above the upper threshold map to `1.0`. Input order is preserved, and the policy neither rounds nor clips values.

Threshold values are inputs to the policy and must be selected outside this module using training/validation data only. Final test data must not participate in threshold selection. The module does not fit or select thresholds and adds no model inference, model training, feature or probability generation, threshold optimization, or alpha claim.

Allocation thresholds convert probabilities into intended target weights. They are separate from rebalance tolerance, which decides whether realized weight is close enough to an intended target to avoid a trade, and minimum trade notional, which can suppress an already-sized order. The allocation module does not inspect bars, timestamps, portfolio state, costs, execution controls, orders, fills, or reporting. Synthetic targets produced by the policy pass directly to the existing fractional engine and reporting, where actual fills and realized snapshots remain the source of reported outcomes.

## Prediction timestamp alignment

`backtest.prediction_alignment.align_probabilities_to_bars` validates already-generated timestamped probabilities against completed chronological bars. Repository bar timestamps identify bar opens, and a prediction timestamp identifies that same decision bar: after UTC normalization, prediction `i` must exactly match `bars[i].timestamp`. Alignment is one-to-one and index-preserving. It does not sort, shift, fill, interpolate, drop, deduplicate, or nearest-match observations, and source timestamps are not mutated.

The prediction timestamp is not an execution timestamp. After alignment and separate allocation, the resulting target for bar `t` remains eligible for execution only at the next bar open through the existing engine. The alignment module does not inspect models, features, labels, thresholds, train/validation/test splits, portfolio state, costs, or execution. Synthetic integration validates timestamp and engine compatibility only; successful alignment does not prove feature-level absence of look-ahead or leakage and adds no model inference, threshold selection, or alpha claim.

## Minimal engine

`backtest.engine.run_backtest` consumes one precomputed CASH or LONG target for each completed chronological bar. A target created from bar `t` can execute only at bar `t+1` open; the final bar's target is retained as unexecuted because no later open exists. The engine delegates target conversion, fill calculation, and portfolio accounting to their existing modules, records real fills only, and stores one immutable end-of-bar portfolio snapshot per bar. Portfolio value is marked as cash plus long quantity times that bar's close and is only a state observation, not a performance conclusion. The loop does not perform prediction, threshold selection, strategy generation, performance analysis, or output generation.

`backtest.engine.run_target_weight_backtest` is a separate explicit workflow for precomputed continuous `TargetWeight` values, with exactly one target supplied per bar. It preserves the same close-decision to next-open timing: the first bar cannot execute its own target, target `t` may execute only at bar `t+1` open, and the final target remains unexecuted. Cost-aware sizing, rebalance tolerance, and minimum-notional filtering remain delegated to positioning. Only actual fills are recorded, and each snapshot is marked at the current bar close after any current-open execution. Binary engine behavior remains unchanged. No post-cost exact-target solver or multi-asset workflow is implemented.

## Portfolio accounting

`backtest.portfolio.apply_fill` returns a new immutable single-asset portfolio state after applying a fill to cash, long position quantity, and cumulative fees. It rejects insufficient cash and insufficient long position; leverage, margin, and short selling are not supported. Fill prices, fees, and slippage are accepted from the execution layer rather than recalculated.

## Deterministic reporting

`backtest.reporting` consumes an already completed immutable binary `BacktestResult` or continuous `TargetWeightBacktestResult`; it does not modify the result or add calculations to the engine. Both result types use the same immutable normalized trade-log records copied from actual fills and the same descriptive whole-period summary metrics: cumulative return, signed maximum drawdown, absolute gross fill-notional turnover, average end-of-bar long capital exposure, fees incurred during the run, and buy, sell, and total fill counts. Suppressed targets create no records or fees, and the final unexecuted target does not affect metrics.

Initial portfolio value marks initial cash and quantity using the first snapshot close; final value uses the last stored snapshot value. Turnover is normalized by the initial marked value and is not annualized or average daily turnover. Exposure is the arithmetic mean of actual close-marked portfolio holdings, not intended target weights or an intrabar measure; fractional exposures therefore arise naturally from realized positions. No Sharpe ratio or annualization is included, and no benchmark or alpha conclusion is made. These diagnostics are descriptive and are not evidence of robust profitability.

## Not implemented

Models, learned or optimized prediction-to-target policies, threshold-selection workflows, strategy frameworks, signals, pending-order queues, benchmark comparison, risk-adjusted or annualized performance metrics, configuration loading, CSV output, a command-line interface, notebooks, market-data ingestion, and production backtest behavior are not implemented. Production readiness and profitability assessment are outside the current scope.
