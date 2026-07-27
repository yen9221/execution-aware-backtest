# Project instructions

## Scope and working practices

- Keep changes small, testable, and reviewable.
- Put core implementation logic in `src/backtest/`.
- Treat notebooks as exploratory material, never as the source of truth.
- Do not claim production readiness, robust profitability, or validated trading performance.
- For every task, report modified files, the actual diff, tests run with complete outputs, generated artifacts, and unresolved issues.

The planned project is a correctness-focused, execution-aware backtest prototype demonstrated through deterministic tests, predefined allocation-policy diagnostics, a fixed rule-based market-data case, and an optional frozen weak-model integration. This positioning describes planned work, not functionality that already exists.

## Time-series and execution invariants

- Process time-series data in chronological order. Never randomly shuffle bars.
- A signal is generated only at a bar close and may execute only at the next bar open.
- Never execute a close-generated signal on that same bar's close.
- A signal generated on the final bar must not be executed.
- Never access future bars when producing a signal or decision.
- At each bar, process the pending prior-bar target at the current open before recording the current close snapshot and creating the current bar's next target.
- Binary and continuous-weight engine entry points remain explicit and separate.
- A continuous target created after bar `t` may execute only at bar `t+1` open; the final continuous target remains unexecuted without a later bar.
- The continuous engine delegates sizing, rebalance tolerance, and minimum-notional filtering to positioning.
- Do not silently forward-fill missing OHLC bars. Missing intervals must remain explicit or be handled by a documented, tested policy.
- Deterministic strategies may produce target intents only from completed observable bars.
- Strategy decisions remain separate from execution costs, portfolio state, and no-trade filters.
- The deterministic fractional strategy is a workflow baseline, not a selected or optimized allocation policy.
- Allocation policies map already-available model outputs to target intents and remain separate from model fitting, execution, and reporting.
- Allocation thresholds must be selected outside the allocation module using training/validation data only; final test data must not participate.
- Allocation thresholds, rebalance tolerance, and minimum trade notional are separate controls with different responsibilities.
- Prediction timestamps identify completed decision bars, not execution bars.
- Prediction-to-bar alignment must be exact, one-to-one, chronological, and index-preserving after UTC normalization.
- Alignment must not sort, shift, fill, interpolate, drop, or nearest-match predictions.
- Successful timestamp alignment does not by itself prove feature-level absence of look-ahead or leakage.

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

## Reporting invariants

- Reporting remains target-type agnostic and operates from actual fills and portfolio snapshots.
- Intended target weights must not be substituted for realized portfolio exposure.
- Binary and continuous-weight results use the same reporting definitions.

## Evaluation roadmap

This ordered roadmap describes planned future work and must not be interpreted as already implemented functionality.

1. **Continuous mapping.** Add a transparent deterministic mapping from probabilities to continuous long-only target weights in `[0, 1]`; treat it as a predefined diagnostic assumption, not an optimized sizing rule.
2. **Continuous synthetic integration.** Validate partial rebalancing, next-bar execution, fills, cash, position quantity, close-marked snapshots, and final-target behavior with hand-checkable synthetic data.
3. **Synthetic three-policy comparison.** Compare binary, three-state, and continuous mapping mechanics on the same fixed synthetic probability sequence before using real prediction distributions. Examine target transitions, trade count, turnover, average exposure, and rebalance behavior; report every policy without selecting a winner.
4. **Cost sensitivity.** Compare predefined fee and slippage scenarios across the complete portfolio path. Zero-cost cumulative return minus cost-inclusive cumulative return is only a descriptive scenario-level return difference, not exact trade-cost attribution, because costs may change quantities, affordability, cash paths, future trade sizes, and portfolio paths.
5. **Fixed max-weight diagnostic.** Compare predefined exposure caps to determine whether lower drawdown or return is primarily explained by lower realized exposure; never optimize `max_weight` using final-test results.
6. **Rule-based real-market demonstration.** Demonstrate execution on real BTC or ETH bars with a fixed transparent rule capable of fractional target weights, entries, exits, partial rebalances, turnover, fees, slippage, drawdown, and exposure variation. Specify the formula and parameters before reviewing full evaluation results, perform no final-test parameter search, and treat the rule as a market-data demonstration rather than alpha evidence. Prefer a three-state rule when it is simpler and more defensible than a fitted continuous rule. This is the primary real-market execution demonstration and remains separate from optional weak-model integration.
7. **Prediction artifact schema and loader.** Create a reproducible cross-repository interface for frozen predictions with at least `timestamp`, `symbol`, `split`, `probability`, `model_name`, `feature_set`, and data-snapshot or artifact-version metadata. Preserve chronological order, exact UTC timestamp semantics, duplicate and missing timestamp rejection, exact prediction-to-bar alignment, explicit split metadata, and reproducible provenance. Do not fit models in this repository.
8. **Predefined policy specification and freeze.** Before final evaluation, freeze mapping formulas, thresholds, rule-based parameters, cost scenarios, rebalance tolerance, minimum trade notional, and max weight. Record parameter sources, predefined status, freeze timestamp or artifact version, confirmation that final-test results did not participate, and all diagnostic policies without selecting a winner. Do not call this policy selection.
9. **Optional frozen ML integration.** Integrate frozen BTC/ETH outputs only as an optional negative diagnostic and cross-project interface test after mapping and execution assumptions are frozen. State their weak out-of-sample discrimination; do not present them as validated signal evidence, the project's main empirical justification, or a case where friction destroyed an effective signal. Examine only probability distributions, target transitions, turnover, exposure, and cost profiles.
10. **Final descriptive evaluation.** Report rule-based and weak-model cases separately. For the rule-based case, include zero-cost and cost-inclusive scenarios, turnover, trade count, average exposure, fees, slippage assumptions, maximum drawdown, buy-and-hold comparison, and a no-trade baseline where applicable. For the weak-model case, include source-project discrimination metrics, probability distribution, target transition count, turnover, average exposure, scenario-level return difference, and the absence of a supported profitability conclusion. Never interpret classification metrics directly as tradable alpha.
11. **README and portfolio narrative.** Document the completed workflow, results, limitations, and conservative interview narrative only after the corresponding functionality and evaluations exist; never describe planned components as implemented.

## Evaluation design invariants

### Layer separation

- Maintain three distinct demonstration layers: Layer A is synthetic correctness, Layer B is the rule-based real-market execution demonstration, and Layer C is optional frozen weak-model integration.
- Layer A validates mathematics, timing, and accounting; Layer B demonstrates execution behavior on real price paths; Layer C validates cross-project integration and documents a weak-model negative case.
- Do not merge conclusions across these layers.

### Policy comparison

- Binary, three-state, and continuous mappings are predefined diagnostic assumptions.
- Use synthetic probabilities first to isolate mapping mechanics; the same policies may later be applied to real-market or frozen-model cases.
- Report all policies and select no winner from final-test performance.
- More trades are not automatically better; increased turnover is a diagnostic outcome, not an objective.

### Rule-based demonstration

- The rule must be transparent, fixed, reproducible, long-only, and based only on completed observable bars.
- It may generate fractional weights to exercise partial rebalancing, but must not be optimized for attractive historical performance.
- It is not evidence of robust profitability or alpha.

### Weak-model integration

- Frozen ML outputs are optional, and weak out-of-sample discrimination must be stated explicitly.
- Use this case to study how weak or noisy outputs interact with predefined mappings and execution assumptions.
- Do not describe the result as market friction degrading a validated signal. Use the more accurate causal description: weak frozen outputs -> predefined mapping -> target changes -> trading activity -> cost and exposure profile.

### Freeze discipline

- Before final evaluation, freeze policy formulas, thresholds, rule parameters, cost scenarios, rebalance tolerance, minimum trade notional, and max weight.
- Final-test results must not cause any frozen setting to be revised.

### Interpretation

- Distinguish model metrics, target behavior, execution diagnostics, and portfolio metrics.
- Lower drawdown under a cap may reflect lower exposure; higher turnover may reflect noisy target changes.
- Scenario-level return differences are descriptive, not exact cost attribution.
- Negative results are valid outcomes.
- Do not claim production readiness, robust profitability, or validated alpha.

## Future scenario assumptions

When configuration is introduced, begin with these configurable defaults:

```yaml
initial_cash: 10000.0
fee_rate: 0.001
slippage_rate: 0.0005
```

These values are scenario assumptions only. They are not an exact reconstruction of any particular exchange account, fee or VIP tier, liquidity condition, or market-impact model.
