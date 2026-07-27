import ast
import csv
import hashlib
import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import backtest.predictions as predictions_module
from backtest.events import Bar
from backtest.predictions import (
    PREDICTION_COLUMNS,
    PREDICTION_METADATA_COLUMNS,
    Prediction,
    PredictionArtifact,
    PredictionArtifactError,
    PredictionArtifactMetadata,
    align_predictions_to_bars,
    load_prediction_artifact,
    load_prediction_metadata_csv,
    load_predictions_csv,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PREDICTION_ROWS = [
    ("2024-01-01T00:00:00Z", "BTCUSDT", "0.45", "model-v1", "demonstration"),
    ("2024-01-01T01:00:00Z", "BTCUSDT", "0.62", "model-v1", "demonstration"),
]


def write_csv(
    path: Path,
    *,
    header: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_predictions(
    tmp_path: Path,
    *,
    rows: list[tuple[str, ...]] | None = None,
    header: tuple[str, ...] = PREDICTION_COLUMNS,
    filename: str = "predictions.csv",
) -> Path:
    return write_csv(
        tmp_path / filename,
        header=header,
        rows=PREDICTION_ROWS if rows is None else rows,
    )


def metadata_row(prediction_path: Path, **overrides: str) -> tuple[str, ...]:
    values = {
        "artifact_id": "artifact-model-v1-demo",
        "prediction_filename": prediction_path.name,
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "symbol": "BTCUSDT",
        "bar_interval": "1h",
        "timestamp_semantics": "bar_open_time",
        "start_timestamp": "2024-01-01T00:00:00Z",
        "end_timestamp": "2024-01-01T01:00:00Z",
        "row_count": "2",
        "model_id": "model-v1",
        "model_version": "1.0.0",
        "data_snapshot_id": "synthetic-bars-v1",
        "generation_timestamp_utc": "2024-02-01T00:00:00Z",
        "period_label": "demonstration",
        "feature_set_id": "features-v1",
        "threshold_selection_used": "False",
        "allocation_mapping_used": "False",
        "final_test_status": "synthetic_documentation_only",
        "notes": "",
    }
    values.update(overrides)
    return tuple(values[column] for column in PREDICTION_METADATA_COLUMNS)


def write_metadata(
    tmp_path: Path,
    prediction_path: Path,
    *,
    rows: list[tuple[str, ...]] | None = None,
    header: tuple[str, ...] = PREDICTION_METADATA_COLUMNS,
) -> Path:
    return write_csv(
        tmp_path / "prediction_metadata.csv",
        header=header,
        rows=[metadata_row(prediction_path)] if rows is None else rows,
    )


def valid_artifact(tmp_path: Path) -> tuple[Path, Path]:
    prediction_path = write_predictions(tmp_path)
    metadata_path = write_metadata(tmp_path, prediction_path)
    return prediction_path, metadata_path


def bars(count: int = 2) -> tuple[Bar, ...]:
    return tuple(
        Bar(
            timestamp=START + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=10.0,
        )
        for index in range(count)
    )


def prediction_records() -> tuple[Prediction, ...]:
    return (
        Prediction(START, "BTCUSDT", 0.45, "model-v1", "demonstration"),
        Prediction(
            START + timedelta(hours=1),
            "BTCUSDT",
            0.62,
            "model-v1",
            "demonstration",
        ),
    )


def test_public_api_and_exact_schema_constants() -> None:
    assert issubclass(PredictionArtifactError, ValueError)
    assert PREDICTION_COLUMNS == (
        "prediction_timestamp",
        "symbol",
        "probability",
        "model_id",
        "period_label",
    )
    assert PREDICTION_METADATA_COLUMNS == (
        "artifact_id",
        "prediction_filename",
        "prediction_sha256",
        "symbol",
        "bar_interval",
        "timestamp_semantics",
        "start_timestamp",
        "end_timestamp",
        "row_count",
        "model_id",
        "model_version",
        "data_snapshot_id",
        "generation_timestamp_utc",
        "period_label",
        "feature_set_id",
        "threshold_selection_used",
        "allocation_mapping_used",
        "final_test_status",
        "notes",
    )
    assert all(
        callable(function)
        for function in (
            load_predictions_csv,
            load_prediction_metadata_csv,
            load_prediction_artifact,
            align_predictions_to_bars,
        )
    )


def test_minimal_valid_artifact_loads_and_preserves_values(tmp_path: Path) -> None:
    prediction_path, metadata_path = valid_artifact(tmp_path)
    artifact = load_prediction_artifact(
        predictions_path=prediction_path,
        metadata_path=metadata_path,
    )
    assert type(artifact) is PredictionArtifact
    assert tuple(item.probability for item in artifact.predictions) == (0.45, 0.62)
    assert artifact.predictions[0].timestamp == START
    assert artifact.predictions[0].timestamp.tzinfo is timezone.utc
    assert artifact.predictions[0].symbol == "BTCUSDT"
    assert artifact.predictions[0].model_id == "model-v1"
    assert artifact.predictions[0].period_label == "demonstration"
    assert artifact.metadata.notes == ""


def test_prediction_and_artifact_dataclasses_are_immutable_and_slotted(
    tmp_path: Path,
) -> None:
    prediction_path, metadata_path = valid_artifact(tmp_path)
    artifact = load_prediction_artifact(
        predictions_path=prediction_path,
        metadata_path=metadata_path,
    )
    assert not hasattr(artifact, "__dict__")
    assert not hasattr(artifact.predictions[0], "__dict__")
    assert not hasattr(artifact.metadata, "__dict__")
    with pytest.raises(FrozenInstanceError):
        artifact.predictions[0].probability = 0.1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        artifact.metadata.model_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        artifact.predictions = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "header",
    [
        PREDICTION_COLUMNS[:-1],
        (*PREDICTION_COLUMNS, "extra"),
        ("symbol", *PREDICTION_COLUMNS[1:]),
    ],
    ids=["missing", "extra", "wrong-order"],
)
def test_prediction_column_schema_is_exact(
    tmp_path: Path, header: tuple[str, ...]
) -> None:
    row = tuple(PREDICTION_ROWS[0][: len(header)])
    with pytest.raises(PredictionArtifactError, match="invalid CSV schema"):
        load_predictions_csv(write_predictions(tmp_path, rows=[row], header=header))


def test_missing_prediction_file_is_rejected_without_absolute_path() -> None:
    with pytest.raises(PredictionArtifactError, match=r"missing\.csv: cannot read") as error:
        load_predictions_csv(Path("missing.csv"))
    assert str(Path.cwd()) not in str(error.value)


def test_unreadable_prediction_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PredictionArtifactError, match="cannot read"):
        load_predictions_csv(tmp_path)


def test_empty_prediction_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(PredictionArtifactError, match="empty CSV file"):
        load_predictions_csv(path)


def test_header_only_prediction_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PredictionArtifactError, match="header but no data rows"):
        load_predictions_csv(write_predictions(tmp_path, rows=[]))


@pytest.mark.parametrize("field_index", range(5))
def test_blank_prediction_required_values_are_rejected(
    tmp_path: Path, field_index: int
) -> None:
    row = list(PREDICTION_ROWS[0])
    row[field_index] = "  "
    field = PREDICTION_COLUMNS[field_index]
    with pytest.raises(PredictionArtifactError, match=rf"field '{field}'.*empty"):
        load_predictions_csv(write_predictions(tmp_path, rows=[tuple(row)]))


def test_surrounding_text_whitespace_is_rejected_not_silently_changed(
    tmp_path: Path,
) -> None:
    row = list(PREDICTION_ROWS[0])
    row[1] = " BTCUSDT"
    with pytest.raises(PredictionArtifactError, match="surrounding whitespace"):
        load_predictions_csv(write_predictions(tmp_path, rows=[tuple(row)]))


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        ("not-a-time", "valid ISO 8601"),
        ("2024-01-01T00:00:00", "timezone information"),
        ("2024-01-01T01:00:00+01:00", "must use UTC"),
    ],
)
def test_invalid_prediction_timestamps_are_rejected(
    tmp_path: Path, timestamp: str, message: str
) -> None:
    row = (timestamp, *PREDICTION_ROWS[0][1:])
    with pytest.raises(PredictionArtifactError, match=message):
        load_predictions_csv(write_predictions(tmp_path, rows=[row]))


@pytest.mark.parametrize(
    ("probability", "message"),
    [
        ("buy", "not numeric"),
        ("NaN", "finite"),
        ("inf", "finite"),
        ("-inf", "finite"),
        ("-0.0001", "inclusive range"),
        ("1.0001", "inclusive range"),
    ],
)
def test_invalid_prediction_probabilities_are_rejected_without_coercion(
    tmp_path: Path, probability: str, message: str
) -> None:
    row = (*PREDICTION_ROWS[0][:2], probability, *PREDICTION_ROWS[0][3:])
    with pytest.raises(PredictionArtifactError, match=message):
        load_predictions_csv(write_predictions(tmp_path, rows=[row]))


def test_probability_endpoints_and_float_precision_are_preserved(tmp_path: Path) -> None:
    precise = "0.12345678901234566"
    rows = [
        (*PREDICTION_ROWS[0][:2], "0.0", *PREDICTION_ROWS[0][3:]),
        (*PREDICTION_ROWS[1][:2], precise, *PREDICTION_ROWS[1][3:]),
        (
            "2024-01-01T02:00:00Z",
            "BTCUSDT",
            "1.0",
            "model-v1",
            "demonstration",
        ),
    ]
    loaded = load_predictions_csv(write_predictions(tmp_path, rows=rows))
    assert tuple(item.probability for item in loaded) == (0.0, float(precise), 1.0)


def test_duplicate_prediction_timestamp_is_rejected(tmp_path: Path) -> None:
    rows = [PREDICTION_ROWS[0], (PREDICTION_ROWS[0][0], *PREDICTION_ROWS[1][1:])]
    with pytest.raises(PredictionArtifactError, match="duplicate"):
        load_predictions_csv(write_predictions(tmp_path, rows=rows))


def test_reversed_prediction_timestamps_are_rejected_not_sorted(tmp_path: Path) -> None:
    path = write_predictions(tmp_path, rows=list(reversed(PREDICTION_ROWS)))
    original = path.read_bytes()
    with pytest.raises(PredictionArtifactError, match="not strictly later"):
        load_predictions_csv(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("field_index", "changed", "message"),
    [
        (1, "ETHUSDT", "first-row symbol"),
        (3, "model-v2", "first-row model_id"),
        (4, "final_test", "first-row period_label"),
    ],
)
def test_inconsistent_prediction_identifiers_are_rejected(
    tmp_path: Path, field_index: int, changed: str, message: str
) -> None:
    second = list(PREDICTION_ROWS[1])
    second[field_index] = changed
    with pytest.raises(PredictionArtifactError, match=message):
        load_predictions_csv(
            write_predictions(tmp_path, rows=[PREDICTION_ROWS[0], tuple(second)])
        )


def test_valid_metadata_loads_with_exact_types(tmp_path: Path) -> None:
    prediction_path, metadata_path = valid_artifact(tmp_path)
    metadata = load_prediction_metadata_csv(metadata_path)
    assert type(metadata) is PredictionArtifactMetadata
    assert [field.name for field in fields(metadata)] == list(
        PREDICTION_METADATA_COLUMNS
    )
    assert metadata.prediction_filename == prediction_path.name
    assert metadata.row_count == 2
    assert type(metadata.row_count) is int
    assert metadata.threshold_selection_used is False
    assert metadata.allocation_mapping_used is False
    assert metadata.generation_timestamp_utc.tzinfo is timezone.utc


def test_metadata_must_contain_exactly_one_row(tmp_path: Path) -> None:
    prediction_path = write_predictions(tmp_path)
    row = metadata_row(prediction_path)
    path = write_metadata(tmp_path, prediction_path, rows=[row, row])
    with pytest.raises(PredictionArtifactError, match="exactly one data row"):
        load_prediction_metadata_csv(path)


@pytest.mark.parametrize(
    "header",
    [
        PREDICTION_METADATA_COLUMNS[:-1],
        (*PREDICTION_METADATA_COLUMNS, "extra"),
        (PREDICTION_METADATA_COLUMNS[1], PREDICTION_METADATA_COLUMNS[0],
         *PREDICTION_METADATA_COLUMNS[2:]),
    ],
    ids=["missing", "extra", "wrong-order"],
)
def test_metadata_column_schema_is_exact(
    tmp_path: Path, header: tuple[str, ...]
) -> None:
    prediction_path = write_predictions(tmp_path)
    row = metadata_row(prediction_path)[: len(header)]
    path = write_metadata(tmp_path, prediction_path, header=header, rows=[row])
    with pytest.raises(PredictionArtifactError, match="invalid CSV schema"):
        load_prediction_metadata_csv(path)


@pytest.mark.parametrize("value", ["true", "false", "1", "0", "yes", "no", ""])
def test_metadata_boolean_must_be_explicit(tmp_path: Path, value: str) -> None:
    prediction_path = write_predictions(tmp_path)
    row = metadata_row(prediction_path, threshold_selection_used=value)
    path = write_metadata(tmp_path, prediction_path, rows=[row])
    with pytest.raises(PredictionArtifactError, match="exactly 'True' or 'False'"):
        load_prediction_metadata_csv(path)


@pytest.mark.parametrize("value", ["0", "-1", "1.0", "two", ""])
def test_metadata_row_count_must_be_positive_integer(
    tmp_path: Path, value: str
) -> None:
    prediction_path = write_predictions(tmp_path)
    path = write_metadata(
        tmp_path,
        prediction_path,
        rows=[metadata_row(prediction_path, row_count=value)],
    )
    with pytest.raises(PredictionArtifactError, match="positive integer|is empty"):
        load_prediction_metadata_csv(path)


def test_generation_timestamp_must_be_explicit_utc(tmp_path: Path) -> None:
    prediction_path = write_predictions(tmp_path)
    for timestamp in ("2024-02-01T00:00:00", "2024-02-01T01:00:00+01:00"):
        path = write_metadata(
            tmp_path,
            prediction_path,
            rows=[metadata_row(prediction_path, generation_timestamp_utc=timestamp)],
        )
        with pytest.raises(PredictionArtifactError, match="timezone|must use UTC"):
            load_prediction_metadata_csv(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prediction_filename", "other.csv"),
        ("prediction_sha256", "0" * 64),
        ("symbol", "ETHUSDT"),
        ("model_id", "model-v2"),
        ("period_label", "final_test"),
        ("start_timestamp", "2023-12-31T23:00:00Z"),
        ("end_timestamp", "2024-01-01T02:00:00Z"),
        ("row_count", "3"),
        ("bar_interval", "5m"),
        ("timestamp_semantics", "bar_close_time"),
    ],
)
def test_artifact_metadata_reconciliation_mismatches_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    prediction_path = write_predictions(tmp_path)
    metadata_path = write_metadata(
        tmp_path,
        prediction_path,
        rows=[metadata_row(prediction_path, **{field: value})],
    )
    with pytest.raises(PredictionArtifactError, match=rf"{field!s}.*mismatch"):
        load_prediction_artifact(
            predictions_path=prediction_path,
            metadata_path=metadata_path,
        )


def test_checksum_is_over_exact_prediction_file_bytes(tmp_path: Path) -> None:
    prediction_path, metadata_path = valid_artifact(tmp_path)
    prediction_path.write_bytes(
        prediction_path.read_bytes().replace(b",0.45,", b",0.46,", 1)
    )
    with pytest.raises(PredictionArtifactError, match="prediction_sha256.*mismatch"):
        load_prediction_artifact(
            predictions_path=prediction_path,
            metadata_path=metadata_path,
        )


def test_metadata_flags_and_descriptive_values_are_preserved(tmp_path: Path) -> None:
    prediction_path = write_predictions(tmp_path)
    row = metadata_row(
        prediction_path,
        threshold_selection_used="True",
        allocation_mapping_used="True",
        final_test_status="opaque_status_supplied_externally",
        notes="free text is preserved",
    )
    metadata = load_prediction_metadata_csv(
        write_metadata(tmp_path, prediction_path, rows=[row])
    )
    assert metadata.threshold_selection_used is True
    assert metadata.allocation_mapping_used is True
    assert metadata.final_test_status == "opaque_status_supplied_externally"
    assert metadata.notes == "free text is preserved"


def test_exact_one_to_one_alignment_succeeds_in_bar_order() -> None:
    source_predictions = prediction_records()
    source_bars = bars()
    original_predictions = tuple(replace(item) for item in source_predictions)
    original_bars = tuple(replace(item) for item in source_bars)
    assert align_predictions_to_bars(
        predictions=source_predictions,
        bars=source_bars,
        expected_symbol="BTCUSDT",
    ) == (0.45, 0.62)
    assert source_predictions == original_predictions
    assert source_bars == original_bars


@pytest.mark.parametrize("prediction_count", [1, 3])
def test_missing_or_extra_prediction_is_rejected(prediction_count: int) -> None:
    source = tuple(
        Prediction(
            START + timedelta(hours=index),
            "BTCUSDT",
            0.5,
            "model-v1",
            "demonstration",
        )
        for index in range(prediction_count)
    )
    with pytest.raises(PredictionArtifactError, match="length must equal"):
        align_predictions_to_bars(predictions=source, bars=bars())


@pytest.mark.parametrize("shift", [timedelta(hours=1), timedelta(seconds=1)])
def test_shifted_prediction_timestamps_are_rejected(
    shift: timedelta,
) -> None:
    source = tuple(
        replace(item, timestamp=item.timestamp + shift)
        for item in prediction_records()
    )
    with pytest.raises(PredictionArtifactError, match="does not exactly match"):
        align_predictions_to_bars(predictions=source, bars=bars())


def test_reordered_predictions_are_rejected_without_sorting() -> None:
    source = tuple(reversed(prediction_records()))
    with pytest.raises(PredictionArtifactError, match="strictly later"):
        align_predictions_to_bars(predictions=source, bars=bars())


def test_expected_symbol_is_exact_and_case_sensitive() -> None:
    for symbol in ("ETHUSDT", "btcusdt"):
        with pytest.raises(PredictionArtifactError, match="expected_symbol"):
            align_predictions_to_bars(
                predictions=prediction_records(),
                bars=bars(),
                expected_symbol=symbol,
            )


def test_nearest_timestamp_is_not_used() -> None:
    source = list(prediction_records())
    source[1] = replace(source[1], timestamp=source[1].timestamp + timedelta(microseconds=1))
    with pytest.raises(PredictionArtifactError, match="does not exactly match"):
        align_predictions_to_bars(predictions=source, bars=bars())


def test_alignment_does_not_fill_a_missing_middle_prediction() -> None:
    source_bars = bars(3)
    source = (
        Prediction(START, "BTCUSDT", 0.45, "model-v1", "demonstration"),
        Prediction(
            START + timedelta(hours=2),
            "BTCUSDT",
            0.62,
            "model-v1",
            "demonstration",
        ),
    )
    with pytest.raises(PredictionArtifactError, match="length must equal"):
        align_predictions_to_bars(predictions=source, bars=source_bars)


def test_alignment_rejects_non_hourly_or_reordered_bars() -> None:
    irregular = list(bars())
    irregular[1] = replace(
        irregular[1], timestamp=irregular[0].timestamp + timedelta(minutes=30)
    )
    with pytest.raises(PredictionArtifactError, match="exactly one hour"):
        align_predictions_to_bars(
            predictions=prediction_records(), bars=irregular
        )
    with pytest.raises(PredictionArtifactError, match="strictly later"):
        align_predictions_to_bars(
            predictions=prediction_records(), bars=tuple(reversed(bars()))
        )


def test_production_module_has_only_allowed_internal_import_boundary() -> None:
    tree = ast.parse(inspect.getsource(predictions_module))
    internal_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("backtest")
        for alias in node.names
    }
    assert internal_imports == {"Bar"}
    assert predictions_module.__dict__.get("run_backtest") is None
    assert predictions_module.__dict__.get("probabilities_to_target_weights") is None


def test_loader_and_alignment_do_not_mutate_or_execute_external_inputs(
    tmp_path: Path,
) -> None:
    prediction_path, metadata_path = valid_artifact(tmp_path)
    prediction_bytes = prediction_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    artifact = load_prediction_artifact(
        predictions_path=prediction_path,
        metadata_path=metadata_path,
    )
    align_predictions_to_bars(predictions=artifact.predictions, bars=bars())
    assert prediction_path.read_bytes() == prediction_bytes
    assert metadata_path.read_bytes() == metadata_bytes


def test_module_defines_no_model_or_backtest_result_types() -> None:
    public_names = set(predictions_module.__dict__)
    assert public_names.isdisjoint(
        {
            "Model",
            "Feature",
            "Label",
            "TargetWeight",
            "MarketOrder",
            "Fill",
            "PortfolioState",
            "BacktestResult",
        }
    )
