# Allocation Policy Specification Schema

## Stage 8 purpose

Stage 8 defines a strict machine-readable pre-registration mechanism for one
predefined ML execution policy. It freezes policy identity, prediction-artifact
provenance, exposure, costs, execution controls, baselines, output schemas, and
research-integrity restrictions before any Stage 9 result is inspected.

The specification loader runs no model, mapping, backtest, reporting, or
performance comparison. A policy freeze is not model or signal validation.

## Exact CSV schema

The CSV contains exactly one row and these columns in this order:

```text
policy_spec_id
freeze_status
created_at_utc
prediction_artifact_id
prediction_filename
prediction_sha256
prediction_metadata_filename
prediction_metadata_sha256
model_id
model_version
data_snapshot_id
feature_set_id
period_label
prediction_start_timestamp
prediction_end_timestamp
prediction_row_count
final_test_status
policy_id
mapping_type
mapping_formula
threshold_selection_used
lower_threshold
upper_threshold
maximum_target_weight
initial_cash
fee_rate
slippage_rate
rebalance_tolerance
minimum_trade_notional
position_domain
short_selling_allowed
leverage_allowed
execution_timing
final_target_execution
baseline_1
baseline_1_definition
baseline_2
baseline_2_definition
output_schema_version
post_result_parameter_changes_allowed
performance_based_policy_selection_allowed
notes
```

## Complete and incomplete status

`freeze_status` accepts only `complete` or
`incomplete_missing_prediction_artifact`. A complete specification requires a
non-placeholder immutable prediction identity, two exact lowercase SHA-256
values, explicit UTC bounds, and a positive row count. Only a complete valid
specification can pass Stage 9 eligibility.

An incomplete specification must leave every prediction identity field empty.
It documents the mechanism but is not eligible for execution. No real policy
CSV is included in this repository because no genuine immutable ML prediction
artifact has been supplied.

## Frozen continuous mapping

The one predefined policy identity is:

```text
policy_id=continuous_linear_long_only_v1
mapping_type=continuous_linear
mapping_formula=min(1.0,max(0.0,2.0*probability-1.0))
maximum_target_weight=1.0
```

The formula string identifies the existing
`backtest.allocation.probabilities_to_continuous_target_weights` implementation.
The policy-spec module neither duplicates nor calls that mapping. It does not
parse or evaluate formula text.

No probability threshold is used:

```text
threshold_selection_used=False
lower_threshold=
upper_threshold=
```

## Frozen exposure and execution settings

```text
initial_cash=1000.0
fee_rate=0.001
slippage_rate=0.0005
maximum_target_weight=1.0
rebalance_tolerance=0.0
minimum_trade_notional=0.0
position_domain=long_only
short_selling_allowed=False
leverage_allowed=False
execution_timing=decision_after_bar_close_execute_next_bar_open
final_target_execution=prohibited_without_next_bar
```

The full exposure cap means the existing continuous mapping receives no extra
exposure reduction. There is no max-weight optimization or comparison.

## Frozen baselines

Baseline order and definitions are exact:

```text
baseline_1=execution_aligned_buy_and_hold
baseline_1_definition=target[0]=0.0;target[1:]=1.0
baseline_2=zero_position
baseline_2_definition=all_target_weights=0.0
```

The buy-and-hold baseline cannot enter at the first bar open. Its `target[1]`
may execute at `bar[2]` open, aligning the first possible baseline entry with a
model prediction based on a prior completed decision bar.

## Output schema freeze

`output_schema_version=ml_execution_demo_v1` identifies the expected Stage 9
formal outputs: `summary.csv`, `trades.csv`, `portfolio_history.csv`,
`targets.csv`, and `metadata.csv`. Stage 8 generates none of these files.

## Prediction artifact reconciliation

A complete specification reconciles artifact ID, filenames, exact SHA-256
values, model and data identifiers, feature-set and period identifiers,
timestamp bounds, row count, and final-test status against one Stage 7
`PredictionArtifact`. The separate prediction metadata CSV filename and exact
file-byte SHA-256 are also checked. Inputs are never rewritten.

The Stage 8 schema deliberately contains no independent symbol column. Symbol
consistency remains enforced inside the Stage 7 artifact between its prediction
rows and metadata.

## Research integrity and Stage 9 eligibility

Both fields must remain false:

```text
post_result_parameter_changes_allowed=False
performance_based_policy_selection_allowed=False
```

After final-result inspection, changing any frozen setting requires a new
`policy_spec_id`, a new experiment designation, and explicit non-final
diagnostic status. A changed policy cannot replace the frozen final result.

Final-test data and outcomes must not participate in feature, model, parameter,
threshold, cap, mapping, or policy selection. Reconciled metadata is supplied
provenance only; the loader cannot certify that upstream leakage controls or a
`final_test_status` claim are truthful.

## Synthetic incomplete documentation example

The following abbreviated values describe an incomplete synthetic example; a
real CSV must still contain every schema column in order:

```text
policy_spec_id=synthetic-incomplete-v1
freeze_status=incomplete_missing_prediction_artifact
created_at_utc=2024-02-01T00:00:00Z
prediction_artifact_id=
prediction_filename=
prediction_sha256=
prediction_metadata_filename=
prediction_metadata_sha256=
policy_id=continuous_linear_long_only_v1
mapping_type=continuous_linear
mapping_formula=min(1.0,max(0.0,2.0*probability-1.0))
```

## Synthetic complete documentation example

The following values are valid-looking but synthetic documentation only. They
do not identify a real artifact or final-test experiment:

```text
policy_spec_id=synthetic-complete-v1
freeze_status=complete
prediction_artifact_id=synthetic-predictions-v1
prediction_filename=synthetic_predictions.csv
prediction_sha256=1111111111111111111111111111111111111111111111111111111111111111
prediction_metadata_filename=synthetic_prediction_metadata.csv
prediction_metadata_sha256=2222222222222222222222222222222222222222222222222222222222222222
model_id=synthetic-model-v1
model_version=1.0.0
data_snapshot_id=synthetic-snapshot-v1
feature_set_id=synthetic-features-v1
period_label=synthetic_final_test
prediction_start_timestamp=2024-01-01T00:00:00Z
prediction_end_timestamp=2024-01-01T01:00:00Z
prediction_row_count=2
final_test_status=synthetic_only_not_real_final_test
```

This example is not shipped as a loadable policy specification and must not be
treated as Stage 9 eligible in any real workflow.
