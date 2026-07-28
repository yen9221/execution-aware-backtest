# ML Integration Compatibility Decision

## 1. Decision status

```text
decision_status=deferred_missing_compatible_hourly_prediction_artifact
execution_workflow_interval=1h
prediction_artifact_interval=1h
existing_signal_validation_interval=1d
direct_integration_compatible=False
performance_inspected_in_stage9=False
model_executed_in_stage9=False
backtest_executed_in_stage9=False
```

## 2. Context

The execution-aware project uses strict hourly OHLCV bars. `Bar.timestamp`
denotes the bar open time, while a prediction timestamp identifies the same
completed hourly decision bar and is assumed available only after that bar
closes.

Stage 7 requires immutable prediction artifacts with `bar_interval=1h` and
exact, one-to-one, index-preserving prediction-to-bar alignment. Stage 8 freezes
an hourly execution policy interface and requires a reconciled Stage 7 artifact
before Stage 9 eligibility can be established.

The existing BTC/ETH signal-validation project was built on daily observations.
That daily frequency is a valid design choice for its own methodology, but its
available outputs do not have the hourly row frequency required by the current
execution contract.

## 3. Compatibility assessment

| Assessment item | Existing ML workflow | Current execution workflow | Compatibility |
| --- | --- | --- | --- |
| Bar interval | Daily (`1d`) observations | Hourly (`1h`) OHLCV bars | No direct match |
| Prediction interval | One daily prediction interval | One hourly prediction interval | No direct match |
| Timestamp granularity | Completed daily decision bars | Completed hourly decision bars identified by bar open time | No direct match |
| One-to-one row alignment | Daily rows | Exactly one prediction row per hourly bar | Not satisfied |
| Execution timing | Decisions associated with completed daily observations | Hourly close decision, next hourly bar open execution | Different event path |
| Artifact metadata interval | Existing outputs are not a Stage 7 hourly artifact | Stage 7 requires `bar_interval=1h` | Not satisfied |

```text
Direct compatibility: No
```

## 4. Prohibited transformations

The following transformations must not be used to force compatibility:

- daily-to-hourly forward filling;
- duplicating one daily probability across multiple hourly bars;
- linear or nonlinear interpolation;
- nearest-timestamp matching;
- timestamp shifting;
- timestamp relabeling;
- implicit overlap selection;
- synthetic intraday probability generation;
- changing symbol labels to bypass instrument mismatch;
- weakening the Stage 7 exact-alignment contract.

These transformations would create artificial intraday information, change the
intended holding and rebalance path, alter turnover and transaction-cost
exposure, obscure decision-time semantics, and do not preserve the original
model information set.

## 5. Decision

Real ML execution integration is deferred. No daily-to-hourly compatibility
conversion is performed. The Stage 7 prediction-artifact interface and Stage 8
policy-freeze mechanism remain the valid future integration boundaries.

Stage 9 adds no model, mapping execution, threshold selection, backtest, or
performance result. The current repository remains a standalone
execution-aware prototype.

## 6. Future integration eligibility

Future integration requires all of the following:

- a genuine hourly market-data snapshot;
- explicit data snapshot identity and checksum;
- an hourly feature workflow;
- a chronological train/validation/test split;
- train-only preprocessing fit;
- a frozen feature specification;
- a frozen model;
- raw hourly probabilities;
- prediction timestamps representing hourly decision bars;
- a Stage 7-compatible prediction CSV and metadata;
- a complete Stage 8 policy specification;
- successful prediction-artifact reconciliation;
- successful exact alignment with the execution bars;
- no final-test use in feature, model, threshold, cap, or policy selection.

Future hourly training belongs in the separate ML repository. This execution
repository must receive only frozen prediction artifacts and must not train or
refit the model.

## 7. Methodological interpretation

```text
This decision is not model evaluation.
This decision is not signal rejection.
This decision does not show that daily predictions are unusable.
This decision only establishes that the available daily outputs do not satisfy
the current hourly execution contract.
```

## 8. Scope status

```text
execution_engine_complete=True
synthetic_diagnostics_complete=True
rule_based_real_market_demonstration_complete=True
prediction_artifact_interface_complete=True
policy_freeze_mechanism_complete=True
compatible_real_hourly_ml_artifact_available=False
real_ml_execution_integration_complete=False
```

These scope statements describe completed prototype components and the deferred
integration boundary. They do not claim production readiness.

## 9. Future extension path

```text
separate ML repository
-> hourly leakage-aware training workflow
-> frozen hourly raw probabilities
-> Stage 7 artifact validation
-> Stage 8 complete policy freeze
-> optional future frozen ML execution integration
```

This is a later extension. Any future ML execution integration must remain
separate from, and must not overwrite, the existing standalone synthetic and
rule-based real-market results.
