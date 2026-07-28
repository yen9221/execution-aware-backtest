import ast
import csv
import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import backtest.policy_spec as policy_module
from backtest.policy_spec import (
    COMPLETE,
    INCOMPLETE,
    MAPPING_FORMULA,
    MAPPING_IMPLEMENTATION,
    MAPPING_TYPE,
    OUTPUT_SCHEMA_VERSION,
    POLICY_ID,
    POLICY_SPEC_COLUMNS,
    STAGE9_OUTPUT_FILES,
    FrozenPolicySpec,
    PolicySpecError,
    load_policy_spec_csv,
    validate_policy_spec_against_prediction_artifact,
    validate_stage9_eligibility,
)
from backtest.predictions import (
    Prediction,
    PredictionArtifact,
    PredictionArtifactMetadata,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
PREDICTION_SHA = "1" * 64


def metadata_file(tmp_path: Path) -> Path:
    path = tmp_path / "prediction_metadata.csv"
    path.write_bytes(b"synthetic prediction metadata bytes\n")
    return path


def complete_values(tmp_path: Path, **overrides: str) -> dict[str, str]:
    metadata_path = metadata_file(tmp_path)
    values = {
        "policy_spec_id": "synthetic-policy-freeze-v1",
        "freeze_status": COMPLETE,
        "created_at_utc": "2024-02-01T00:00:00Z",
        "prediction_artifact_id": "synthetic-predictions-v1",
        "prediction_filename": "predictions.csv",
        "prediction_sha256": PREDICTION_SHA,
        "prediction_metadata_filename": metadata_path.name,
        "prediction_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "model_id": "synthetic-model-v1",
        "model_version": "1.0.0",
        "data_snapshot_id": "synthetic-snapshot-v1",
        "feature_set_id": "synthetic-features-v1",
        "period_label": "synthetic_final_test",
        "prediction_start_timestamp": "2024-01-01T00:00:00Z",
        "prediction_end_timestamp": "2024-01-01T01:00:00Z",
        "prediction_row_count": "2",
        "final_test_status": "synthetic_only_not_real_final_test",
        "policy_id": POLICY_ID,
        "mapping_type": MAPPING_TYPE,
        "mapping_formula": MAPPING_FORMULA,
        "threshold_selection_used": "False",
        "lower_threshold": "",
        "upper_threshold": "",
        "maximum_target_weight": "1.0",
        "initial_cash": "1000.0",
        "fee_rate": "0.001",
        "slippage_rate": "0.0005",
        "rebalance_tolerance": "0.0",
        "minimum_trade_notional": "0.0",
        "position_domain": "long_only",
        "short_selling_allowed": "False",
        "leverage_allowed": "False",
        "execution_timing": "decision_after_bar_close_execute_next_bar_open",
        "final_target_execution": "prohibited_without_next_bar",
        "baseline_1": "execution_aligned_buy_and_hold",
        "baseline_1_definition": "target[0]=0.0;target[1:]=1.0",
        "baseline_2": "zero_position",
        "baseline_2_definition": "all_target_weights=0.0",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "post_result_parameter_changes_allowed": "False",
        "performance_based_policy_selection_allowed": "False",
        "notes": "",
    }
    values.update(overrides)
    return values


def incomplete_values(tmp_path: Path, **overrides: str) -> dict[str, str]:
    values = complete_values(tmp_path)
    values["freeze_status"] = INCOMPLETE
    for field in (
        "prediction_artifact_id",
        "prediction_filename",
        "prediction_sha256",
        "prediction_metadata_filename",
        "prediction_metadata_sha256",
        "model_id",
        "model_version",
        "data_snapshot_id",
        "feature_set_id",
        "period_label",
        "prediction_start_timestamp",
        "prediction_end_timestamp",
        "prediction_row_count",
        "final_test_status",
    ):
        values[field] = ""
    values.update(overrides)
    return values


def write_spec(
    tmp_path: Path,
    values: dict[str, str],
    *,
    header: tuple[str, ...] = POLICY_SPEC_COLUMNS,
    rows: list[tuple[str, ...]] | None = None,
) -> Path:
    path = tmp_path / "policy_spec.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(
            rows
            if rows is not None
            else [tuple(values.get(column, "") for column in header)]
        )
    return path


def load_complete(tmp_path: Path, **overrides: str) -> FrozenPolicySpec:
    values = complete_values(tmp_path, **overrides)
    return load_policy_spec_csv(write_spec(tmp_path, values))


def load_incomplete(tmp_path: Path, **overrides: str) -> FrozenPolicySpec:
    values = incomplete_values(tmp_path, **overrides)
    return load_policy_spec_csv(write_spec(tmp_path, values))


def synthetic_artifact() -> PredictionArtifact:
    predictions = (
        Prediction(
            START,
            "BTCUSDT",
            0.45,
            "synthetic-model-v1",
            "synthetic_final_test",
        ),
        Prediction(
            END,
            "BTCUSDT",
            0.62,
            "synthetic-model-v1",
            "synthetic_final_test",
        ),
    )
    metadata = PredictionArtifactMetadata(
        artifact_id="synthetic-predictions-v1",
        prediction_filename="predictions.csv",
        prediction_sha256=PREDICTION_SHA,
        symbol="BTCUSDT",
        bar_interval="1h",
        timestamp_semantics="bar_open_time",
        start_timestamp=START,
        end_timestamp=END,
        row_count=2,
        model_id="synthetic-model-v1",
        model_version="1.0.0",
        data_snapshot_id="synthetic-snapshot-v1",
        generation_timestamp_utc=datetime(2024, 2, 1, tzinfo=timezone.utc),
        period_label="synthetic_final_test",
        feature_set_id="synthetic-features-v1",
        threshold_selection_used=False,
        allocation_mapping_used=False,
        final_test_status="synthetic_only_not_real_final_test",
        notes="synthetic",
    )
    return PredictionArtifact(predictions=predictions, metadata=metadata)


def test_exact_schema_and_frozen_identities() -> None:
    assert len(POLICY_SPEC_COLUMNS) == 42
    assert POLICY_SPEC_COLUMNS == (
        "policy_spec_id", "freeze_status", "created_at_utc",
        "prediction_artifact_id", "prediction_filename", "prediction_sha256",
        "prediction_metadata_filename", "prediction_metadata_sha256", "model_id",
        "model_version", "data_snapshot_id", "feature_set_id", "period_label",
        "prediction_start_timestamp", "prediction_end_timestamp",
        "prediction_row_count", "final_test_status", "policy_id", "mapping_type",
        "mapping_formula", "threshold_selection_used", "lower_threshold",
        "upper_threshold", "maximum_target_weight", "initial_cash", "fee_rate",
        "slippage_rate", "rebalance_tolerance", "minimum_trade_notional",
        "position_domain", "short_selling_allowed", "leverage_allowed",
        "execution_timing", "final_target_execution", "baseline_1",
        "baseline_1_definition", "baseline_2", "baseline_2_definition",
        "output_schema_version", "post_result_parameter_changes_allowed",
        "performance_based_policy_selection_allowed", "notes",
    )
    assert MAPPING_IMPLEMENTATION.endswith(
        "probabilities_to_continuous_target_weights"
    )
    assert STAGE9_OUTPUT_FILES == (
        "summary.csv", "trades.csv", "portfolio_history.csv", "targets.csv",
        "metadata.csv",
    )


def test_valid_complete_spec_loads_with_exact_values(tmp_path: Path) -> None:
    spec = load_complete(tmp_path)
    assert type(spec) is FrozenPolicySpec
    assert spec.freeze_status == COMPLETE
    assert spec.policy_id == POLICY_ID
    assert spec.mapping_formula == MAPPING_FORMULA
    assert spec.maximum_target_weight == 1.0
    assert spec.initial_cash == 1000.0
    assert spec.fee_rate == 0.001
    assert spec.slippage_rate == 0.0005
    assert spec.created_at_utc.tzinfo is timezone.utc
    assert spec.prediction_start_timestamp == START
    assert spec.notes == ""


def test_valid_incomplete_spec_loads_but_preserves_missing_identity(tmp_path: Path) -> None:
    spec = load_incomplete(tmp_path)
    assert spec.freeze_status == INCOMPLETE
    assert spec.prediction_artifact_id == ""
    assert spec.prediction_start_timestamp is None
    assert spec.prediction_row_count is None


def test_dataclass_is_immutable_slotted_and_exactly_typed(tmp_path: Path) -> None:
    spec = load_complete(tmp_path)
    assert not hasattr(spec, "__dict__")
    assert [field.name for field in fields(spec)] == list(POLICY_SPEC_COLUMNS)
    with pytest.raises(FrozenInstanceError):
        spec.fee_rate = 0.0  # type: ignore[misc]


def test_missing_and_unreadable_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicySpecError, match="missing.csv: cannot read"):
        load_policy_spec_csv(tmp_path / "missing.csv")
    with pytest.raises(PolicySpecError, match="cannot read"):
        load_policy_spec_csv(tmp_path)


def test_empty_and_header_only_files_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(PolicySpecError, match="empty CSV"):
        load_policy_spec_csv(empty)
    with pytest.raises(PolicySpecError, match="header but no data rows"):
        load_policy_spec_csv(write_spec(tmp_path, complete_values(tmp_path), rows=[]))


def test_multiple_rows_are_rejected(tmp_path: Path) -> None:
    values = complete_values(tmp_path)
    row = tuple(values[column] for column in POLICY_SPEC_COLUMNS)
    with pytest.raises(PolicySpecError, match="exactly one data row"):
        load_policy_spec_csv(write_spec(tmp_path, values, rows=[row, row]))


@pytest.mark.parametrize(
    "header",
    [
        POLICY_SPEC_COLUMNS[:-1],
        (*POLICY_SPEC_COLUMNS, "extra"),
        (POLICY_SPEC_COLUMNS[1], POLICY_SPEC_COLUMNS[0], *POLICY_SPEC_COLUMNS[2:]),
    ],
    ids=["missing", "extra", "wrong-order"],
)
def test_csv_schema_is_exact(tmp_path: Path, header: tuple[str, ...]) -> None:
    with pytest.raises(PolicySpecError, match="invalid CSV schema"):
        load_policy_spec_csv(write_spec(tmp_path, complete_values(tmp_path), header=header))


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy_spec.csv"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(PolicySpecError, match="valid UTF-8"):
        load_policy_spec_csv(path)


@pytest.mark.parametrize("field", ["policy_spec_id", "policy_id", "position_domain"])
def test_blank_required_field_is_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(PolicySpecError, match=rf"field '{field}'.*empty"):
        load_complete(tmp_path, **{field: ""})


def test_surrounding_identifier_whitespace_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicySpecError, match="surrounding whitespace"):
        load_complete(tmp_path, model_id=" synthetic-model-v1")


@pytest.mark.parametrize("status", ["incomplete", "frozen", "Complete", ""])
def test_invalid_freeze_status_is_rejected(tmp_path: Path, status: str) -> None:
    with pytest.raises(PolicySpecError, match="freeze_status"):
        load_complete(tmp_path, freeze_status=status)


@pytest.mark.parametrize("field", ["prediction_sha256", "prediction_metadata_sha256"])
@pytest.mark.parametrize("value", ["abc", "A" * 64, "0" * 63, "0" * 65])
def test_invalid_sha256_is_rejected(tmp_path: Path, field: str, value: str) -> None:
    with pytest.raises(PolicySpecError, match=field):
        load_complete(tmp_path, **{field: value})


@pytest.mark.parametrize("value", ["", "0", "-1", "1.0", "NaN"])
def test_invalid_prediction_row_count_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(PolicySpecError, match="prediction_row_count"):
        load_complete(tmp_path, prediction_row_count=value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("created_at_utc", "not-time", "valid ISO"),
        ("created_at_utc", "2024-02-01T00:00:00", "timezone"),
        ("prediction_start_timestamp", "2024-01-01T00:00:00", "timezone"),
        ("prediction_start_timestamp", "2024-01-01T01:00:00+01:00", "must use UTC"),
        ("prediction_end_timestamp", "2024-01-01T01:00:00", "timezone"),
    ],
)
def test_invalid_timestamps_are_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    with pytest.raises(PolicySpecError, match=message):
        load_complete(tmp_path, **{field: value})


def test_prediction_start_after_end_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicySpecError, match="before or equal"):
        load_complete(tmp_path, prediction_start_timestamp="2024-01-01T02:00:00Z")


@pytest.mark.parametrize("placeholder", ["TBD", "TODO", "unknown", "placeholder"])
def test_placeholder_identity_in_complete_spec_is_rejected(
    tmp_path: Path, placeholder: str
) -> None:
    with pytest.raises(PolicySpecError, match="placeholder"):
        load_complete(tmp_path, prediction_artifact_id=placeholder)


def test_missing_complete_identity_and_partially_populated_incomplete_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(PolicySpecError, match="prediction_artifact_id.*empty"):
        load_complete(tmp_path, prediction_artifact_id="")
    with pytest.raises(PolicySpecError, match="must be empty"):
        load_incomplete(tmp_path, model_id="synthetic-model-v1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_id", "other_policy"),
        ("mapping_type", "thresholded"),
        ("mapping_formula", "2.0*probability-1.0"),
        ("threshold_selection_used", "True"),
        ("lower_threshold", "0.4"),
        ("upper_threshold", "0.7"),
        ("maximum_target_weight", "0.5"),
        ("initial_cash", "999.0"),
        ("fee_rate", "0.0"),
        ("slippage_rate", "0.0"),
        ("rebalance_tolerance", "0.01"),
        ("minimum_trade_notional", "1.0"),
        ("position_domain", "long_short"),
        ("short_selling_allowed", "True"),
        ("leverage_allowed", "True"),
        ("execution_timing", "same_bar_close"),
        ("final_target_execution", "force_at_close"),
        ("baseline_1", "buy_and_hold"),
        ("baseline_1_definition", "target[:]=1.0"),
        ("baseline_2", "cash"),
        ("baseline_2_definition", "targets=0"),
        ("output_schema_version", "v2"),
        ("post_result_parameter_changes_allowed", "True"),
        ("performance_based_policy_selection_allowed", "True"),
    ],
)
def test_wrong_frozen_policy_value_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(PolicySpecError, match=field):
        load_complete(tmp_path, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("maximum_target_weight", "-0.1", "within"),
        ("maximum_target_weight", "1.1", "within"),
        ("initial_cash", "-1", "non-negative"),
        ("fee_rate", "-0.001", "non-negative"),
        ("slippage_rate", "-0.001", "non-negative"),
        ("rebalance_tolerance", "-0.1", "non-negative"),
        ("minimum_trade_notional", "-1", "non-negative"),
        ("fee_rate", "NaN", "finite"),
        ("slippage_rate", "inf", "finite"),
    ],
)
def test_invalid_numeric_value_is_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    with pytest.raises(PolicySpecError, match=message):
        load_complete(tmp_path, **{field: value})


@pytest.mark.parametrize("value", ["true", "false", "1", "0", "yes", "no", ""])
def test_invalid_boolean_representations_are_rejected(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(PolicySpecError, match="exactly 'True' or 'False'"):
        load_complete(tmp_path, short_selling_allowed=value)


def test_complete_is_stage9_eligible_and_incomplete_is_not(tmp_path: Path) -> None:
    validate_stage9_eligibility(load_complete(tmp_path))
    with pytest.raises(PolicySpecError, match="identity is still missing"):
        validate_stage9_eligibility(load_incomplete(tmp_path))


def test_stage9_eligibility_revalidates_input_type_and_integrity(tmp_path: Path) -> None:
    with pytest.raises(PolicySpecError, match="exactly FrozenPolicySpec"):
        validate_stage9_eligibility(object())  # type: ignore[arg-type]
    spec = load_complete(tmp_path)
    object.__setattr__(spec, "prediction_sha256", "invalid")
    with pytest.raises(PolicySpecError, match="prediction_sha256"):
        validate_stage9_eligibility(spec)


def test_matching_synthetic_artifact_reconciles_without_mutation(tmp_path: Path) -> None:
    spec = load_complete(tmp_path)
    artifact = synthetic_artifact()
    original_spec = replace(spec)
    original_artifact = replace(artifact)
    validate_policy_spec_against_prediction_artifact(
        spec=spec,
        artifact=artifact,
        prediction_metadata_path=tmp_path / "prediction_metadata.csv",
    )
    assert spec == original_spec
    assert artifact == original_artifact


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prediction_artifact_id", "other-artifact"),
        ("prediction_filename", "other.csv"),
        ("prediction_sha256", "2" * 64),
        ("prediction_metadata_filename", "other-metadata.csv"),
        ("prediction_metadata_sha256", "3" * 64),
        ("model_id", "other-model"),
        ("model_version", "2.0.0"),
        ("data_snapshot_id", "other-snapshot"),
        ("feature_set_id", "other-features"),
        ("period_label", "other-period"),
        ("prediction_start_timestamp", "2023-12-31T23:00:00Z"),
        ("prediction_end_timestamp", "2024-01-01T02:00:00Z"),
        ("prediction_row_count", "3"),
        ("final_test_status", "other-status"),
    ],
)
def test_prediction_artifact_reconciliation_mismatch_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    spec = load_complete(tmp_path, **{field: value})
    with pytest.raises(PolicySpecError, match=field):
        validate_policy_spec_against_prediction_artifact(
            spec=spec,
            artifact=synthetic_artifact(),
            prediction_metadata_path=tmp_path / "prediction_metadata.csv",
        )


def test_prediction_artifact_internal_symbol_mismatch_is_rejected(tmp_path: Path) -> None:
    spec = load_complete(tmp_path)
    artifact = synthetic_artifact()
    changed = replace(artifact.predictions[1], symbol="ETHUSDT")
    inconsistent = replace(artifact, predictions=(artifact.predictions[0], changed))
    with pytest.raises(PolicySpecError, match="symbol mismatch"):
        validate_policy_spec_against_prediction_artifact(
            spec=spec,
            artifact=inconsistent,
            prediction_metadata_path=tmp_path / "prediction_metadata.csv",
        )


def test_metadata_checksum_uses_exact_file_bytes(tmp_path: Path) -> None:
    spec = load_complete(tmp_path)
    path = tmp_path / "prediction_metadata.csv"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(PolicySpecError, match="prediction_metadata_sha256"):
        validate_policy_spec_against_prediction_artifact(
            spec=spec,
            artifact=synthetic_artifact(),
            prediction_metadata_path=path,
        )


def test_production_import_and_execution_boundaries() -> None:
    tree = ast.parse(inspect.getsource(policy_module))
    internal_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("backtest")
    }
    assert internal_modules == {"backtest.predictions"}
    source = inspect.getsource(policy_module)
    for forbidden_call in (
        "probabilities_to_continuous_target_weights(",
        "run_target_weight_backtest(",
        "summarize_backtest(",
        "execute_market_order(",
        "apply_fill(",
    ):
        assert forbidden_call not in source


def test_no_result_or_real_artifact_files_are_generated(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    load_complete(tmp_path)
    after = tuple(tmp_path.iterdir())
    assert {path.name for path in after} - {path.name for path in before} == {
        "prediction_metadata.csv", "policy_spec.csv"
    }
    assert set(STAGE9_OUTPUT_FILES).isdisjoint(path.name for path in after)
