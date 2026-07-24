# Project instructions

## Scope and working practices

- Keep changes small, testable, and reviewable.
- Put core implementation logic in `src/backtest/`.
- Treat notebooks as exploratory material, never as the source of truth.
- Do not claim production readiness, robust profitability, or validated trading performance.
- For every task, report modified files, the actual diff, tests run with complete outputs, generated artifacts, and unresolved issues.

## Time-series and execution invariants

- Process time-series data in chronological order. Never randomly shuffle bars.
- A signal is generated only at a bar close and may execute only at the next bar open.
- Never execute a close-generated signal on that same bar's close.
- A signal generated on the final bar must not be executed.
- Never access future bars when producing a signal or decision.
- At each bar, process the pending prior-bar target at the current open before recording the current close snapshot and creating the current bar's next target.
- Do not silently forward-fill missing OHLC bars. Missing intervals must remain explicit or be handled by a documented, tested policy.

## Costs, fills, and accounting

- `TargetWeight` is constrained to `[0, 1]`; fractional weights are valid target representations but must not be silently coerced into the current binary execution path.
- Continuous target-weight rebalance sizing uses pre-trade portfolio value marked at the observable execution-bar open.
- BUY affordability must account for adverse slippage and proportional fees using formulas consistent with execution.
- Transaction costs may cause realized post-trade weight to differ from the intended target.
- Continuous-weight rebalance tolerance is an absolute weight deviation; a deviation less than or equal to tolerance produces no trade.
- When a continuous rebalance is triggered, sizing remains toward the exact target rather than the tolerance boundary.
- Rebalance tolerance is distinct from minimum trade notional.
- Minimum trade notional is applied to expected fill notional after final quantity sizing and after the rebalance-tolerance check.
- The minimum-notional threshold uses the adverse side-specific slipped price, excludes fees, suppresses values below the minimum, and allows equality.
- Skipped continuous-weight orders are not accumulated.
- Positioning returns a delta order only; execution and portfolio accounting remain separate.
- Keep fees and slippage separately traceable in records and results.
- Buy-side slippage increases the fill price; sell-side slippage decreases it.
- Track cash, position quantity, and portfolio value explicitly.
- Repeated target positions must not create unnecessary trades when the current position already matches the target.

## Future scenario assumptions

When configuration is introduced, begin with these configurable defaults:

```yaml
initial_cash: 10000.0
fee_rate: 0.001
slippage_rate: 0.0005
```

These values are scenario assumptions only. They are not an exact reconstruction of any particular exchange account, fee or VIP tier, liquidity condition, or market-impact model.
