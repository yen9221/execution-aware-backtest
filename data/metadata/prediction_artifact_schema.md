# Prediction Artifact Schema

## Scope and responsibility boundary

Stage 7 defines a model-agnostic CSV interface for already-generated prediction
probabilities. An external model workflow creates and freezes the prediction CSV
and its separate metadata CSV. This repository can then load, verify, and align
those files without rerunning inference.

The loader does not train or evaluate models, generate features or labels,
select thresholds, map probabilities to allocations, execute a backtest, or
calculate performance. Probability loading and allocation mapping remain
separate responsibilities.

## Prediction row schema

The prediction CSV must use exactly these columns in exactly this order:

```text
prediction_timestamp
symbol
probability
model_id
period_label
```

- `prediction_timestamp` is an explicit UTC timestamp identifying a completed
  decision bar. It must exactly match the corresponding `Bar.timestamp`, which
  denotes bar open time in this project.
- `symbol` is a non-empty, trimmed, case-sensitive identifier. Aliases and
  symbol conversion are not performed.
- `probability` is a finite numeric value in the inclusive range `[0.0, 1.0]`.
- `model_id` is one non-empty opaque identifier shared by every row.
- `period_label` is one non-empty opaque label shared by every row. The loader
  does not infer methodology from it.

Rows must be unique and strictly chronological. The loader preserves file order
and does not sort, shift, round, clip, deduplicate, fill, interpolate, drop,
impute, or nearest-match values.

## Metadata schema

Prediction metadata remains in a separate CSV containing exactly one row and
exactly these columns in this order:

```text
artifact_id
prediction_filename
prediction_sha256
symbol
bar_interval
timestamp_semantics
start_timestamp
end_timestamp
row_count
model_id
model_version
data_snapshot_id
generation_timestamp_utc
period_label
feature_set_id
threshold_selection_used
allocation_mapping_used
final_test_status
notes
```

`artifact_id`, `model_version`, `data_snapshot_id`, `feature_set_id`, and
`final_test_status` are non-empty opaque identifiers or descriptions. `notes`
may be empty. `generation_timestamp_utc` is explicit UTC metadata and does not
control ordering or alignment. `threshold_selection_used` and
`allocation_mapping_used` accept only `True` or `False`; the loader records
these flags but performs neither operation.

For this project, `bar_interval` must reconcile to `1h` and
`timestamp_semantics` to `bar_open_time`. The artifact loader also requires:

- `prediction_filename` equals the loaded prediction filename;
- `prediction_sha256` equals the SHA-256 of the prediction file's exact bytes;
- `symbol`, `model_id`, and `period_label` equal every prediction row;
- `start_timestamp` and `end_timestamp` equal the first and last prediction
  timestamps;
- `row_count` equals the positive number of prediction rows.

Metadata is never repaired or rewritten when reconciliation fails.

## Decision timestamps and execution timing

`Bar.timestamp` denotes the bar's open time. A prediction timestamp equal to bar
`t` identifies that decision bar, but its data is assumed available only after
bar `t` closes. The loader does not shift that timestamp to an execution bar.
Any target later derived from the prediction may execute no earlier than bar
`t+1` open through the separate engine.

## Exact bar alignment

Full alignment is one-to-one and index-preserving:

```text
len(predictions) == len(bars)
predictions[i].timestamp == bars[i].timestamp
```

Missing, extra, shifted, duplicated, or reordered predictions are rejected.
Neither input is sorted or mutated. There is no implicit subset, overlap,
forward-fill, backward-fill, interpolation, default first prediction, or nearest
timestamp behavior.

Exact timestamp alignment establishes correspondence only. It does not prove
that upstream features, labels, fitting, or inference were free of look-ahead
or leakage.

## Synthetic documentation example

The following prediction CSV is synthetic documentation only:

```csv
prediction_timestamp,symbol,probability,model_id,period_label
2024-01-01T00:00:00Z,BTCUSDT,0.45,example_model_v1,demonstration
2024-01-01T01:00:00Z,BTCUSDT,0.62,example_model_v1,demonstration
```

A matching metadata example is:

```csv
artifact_id,prediction_filename,prediction_sha256,symbol,bar_interval,timestamp_semantics,start_timestamp,end_timestamp,row_count,model_id,model_version,data_snapshot_id,generation_timestamp_utc,period_label,feature_set_id,threshold_selection_used,allocation_mapping_used,final_test_status,notes
synthetic_example,predictions.csv,<sha256-of-prediction-file>,BTCUSDT,1h,bar_open_time,2024-01-01T00:00:00Z,2024-01-01T01:00:00Z,2,example_model_v1,1.0.0,synthetic_snapshot,2024-02-01T00:00:00Z,demonstration,example_features,False,False,synthetic_documentation_only,Not a real ML artifact
```

The placeholder checksum intentionally makes this documentation example
non-loadable until replaced with the exact prediction-file SHA-256. It is not a
real prediction artifact and must not be described as a valid final-test
artifact.

## Final-test contamination warning

Final-test predictions must be generated from an already frozen model and
feature specification and then remain immutable. Final-test data or outcomes
must not participate in feature, model, parameter, threshold, exposure-cap, or
allocation-mapping selection. A supplied `final_test_status` string is retained
as provenance only; the loader cannot certify that an upstream workflow was
genuinely untouched.
