"""Strict loading and exact bar alignment for frozen prediction artifacts."""

import csv
import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.events import Bar

PREDICTION_COLUMNS = (
    "prediction_timestamp",
    "symbol",
    "probability",
    "model_id",
    "period_label",
)
PREDICTION_METADATA_COLUMNS = (
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

_BAR_INTERVAL = timedelta(hours=1)
_REQUIRED_BAR_INTERVAL = "1h"
_REQUIRED_TIMESTAMP_SEMANTICS = "bar_open_time"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*")
_STRING_LIKE = (str, bytes, bytearray)


class PredictionArtifactError(ValueError):
    """Raised when a prediction artifact violates its strict contract."""


def _filename(path: Path) -> str:
    return path.name or str(path)


def _required_text(field: str, value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise PredictionArtifactError(
            f"{context}: field {field!r} must be text, got {value!r}"
        )
    if not value.strip():
        raise PredictionArtifactError(f"{context}: field {field!r} is empty")
    if value != value.strip():
        raise PredictionArtifactError(
            f"{context}: field {field!r} must not contain surrounding whitespace: "
            f"{value!r}"
        )
    return value


def _utc_timestamp(field: str, value: object, *, context: str) -> datetime:
    if not isinstance(value, datetime):
        raise PredictionArtifactError(
            f"{context}: field {field!r} must be a datetime, got {value!r}"
        )
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise PredictionArtifactError(
            f"{context}: field {field!r} has an invalid UTC offset: {value!r}"
        ) from error
    if value.tzinfo is None or offset is None:
        raise PredictionArtifactError(
            f"{context}: field {field!r} must include timezone information: {value!r}"
        )
    if offset != timedelta(0):
        raise PredictionArtifactError(
            f"{context}: field {field!r} must use UTC, got offset {offset}"
        )
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise PredictionArtifactError(
            f"{context}: field {field!r} cannot be normalized to UTC: {value!r}"
        ) from error


def _parse_utc_timestamp(field: str, value: str, *, context: str) -> datetime:
    text = _required_text(field, value, context=context)
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise PredictionArtifactError(
            f"{context}: field {field!r} is not a valid ISO 8601 datetime: {value!r}"
        ) from error
    return _utc_timestamp(field, timestamp, context=context)


def _probability_value(field: str, value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionArtifactError(
            f"{context}: field {field!r} must be numeric and not boolean, "
            f"got {value!r}"
        )
    probability = float(value)
    if not math.isfinite(probability):
        raise PredictionArtifactError(
            f"{context}: field {field!r} must be finite, got {value!r}"
        )
    if not 0.0 <= probability <= 1.0:
        raise PredictionArtifactError(
            f"{context}: field {field!r} must lie within the inclusive range "
            f"[0.0, 1.0], got {probability!r}"
        )
    return probability


def _parse_probability(value: str, *, context: str) -> float:
    text = _required_text("probability", value, context=context)
    try:
        probability = float(text)
    except ValueError as error:
        raise PredictionArtifactError(
            f"{context}: field 'probability' is not numeric: {value!r}"
        ) from error
    return _probability_value("probability", probability, context=context)


def _explicit_boolean(field: str, value: str, *, context: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise PredictionArtifactError(
        f"{context}: field {field!r} must be exactly 'True' or 'False', "
        f"got {value!r}"
    )


def _positive_integer(field: str, value: str, *, context: str) -> int:
    text = _required_text(field, value, context=context)
    if _POSITIVE_INTEGER_PATTERN.fullmatch(text) is None:
        raise PredictionArtifactError(
            f"{context}: field {field!r} must be a positive integer, got {value!r}"
        )
    return int(text)


def _prediction_sha256(value: str, *, context: str) -> str:
    text = _required_text("prediction_sha256", value, context=context)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise PredictionArtifactError(
            f"{context}: field 'prediction_sha256' must be 64 lowercase hexadecimal "
            f"characters, got {value!r}"
        )
    return text


def _read_csv_rows(path: str | Path, expected_columns: tuple[str, ...]) -> list[list[str]]:
    csv_path = Path(path)
    name = _filename(csv_path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise PredictionArtifactError(f"{name}: empty CSV file") from error
            if tuple(header) != expected_columns:
                expected = ",".join(expected_columns)
                actual = ",".join(header)
                raise PredictionArtifactError(
                    f"{name}: invalid CSV schema; expected {expected!r}, got {actual!r}"
                )

            rows: list[list[str]] = []
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(expected_columns):
                    raise PredictionArtifactError(
                        f"{name}: row {row_number}: expected {len(expected_columns)} "
                        f"values, got {len(row)}"
                    )
                rows.append(row)
    except PredictionArtifactError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise PredictionArtifactError(f"{name}: cannot read valid UTF-8 CSV: {error}") from error

    if not rows:
        raise PredictionArtifactError(
            f"{name}: CSV file contains a header but no data rows"
        )
    return rows


@dataclass(frozen=True, slots=True)
class Prediction:
    """One probability associated with one completed decision bar."""

    timestamp: datetime
    symbol: str
    probability: float
    model_id: str
    period_label: str

    def __post_init__(self) -> None:
        context = "Prediction"
        object.__setattr__(
            self,
            "timestamp",
            _utc_timestamp("timestamp", self.timestamp, context=context),
        )
        object.__setattr__(
            self, "symbol", _required_text("symbol", self.symbol, context=context)
        )
        object.__setattr__(
            self,
            "probability",
            _probability_value("probability", self.probability, context=context),
        )
        object.__setattr__(
            self,
            "model_id",
            _required_text("model_id", self.model_id, context=context),
        )
        object.__setattr__(
            self,
            "period_label",
            _required_text("period_label", self.period_label, context=context),
        )


@dataclass(frozen=True, slots=True)
class PredictionArtifactMetadata:
    """Opaque provenance and integrity metadata for one prediction CSV."""

    artifact_id: str
    prediction_filename: str
    prediction_sha256: str
    symbol: str
    bar_interval: str
    timestamp_semantics: str
    start_timestamp: datetime
    end_timestamp: datetime
    row_count: int
    model_id: str
    model_version: str
    data_snapshot_id: str
    generation_timestamp_utc: datetime
    period_label: str
    feature_set_id: str
    threshold_selection_used: bool
    allocation_mapping_used: bool
    final_test_status: str
    notes: str

    def __post_init__(self) -> None:
        context = "PredictionArtifactMetadata"
        for field in (
            "artifact_id",
            "prediction_filename",
            "prediction_sha256",
            "symbol",
            "bar_interval",
            "timestamp_semantics",
            "model_id",
            "model_version",
            "data_snapshot_id",
            "period_label",
            "feature_set_id",
            "final_test_status",
        ):
            object.__setattr__(
                self,
                field,
                _required_text(field, getattr(self, field), context=context),
            )
        _prediction_sha256(self.prediction_sha256, context=context)
        object.__setattr__(
            self,
            "start_timestamp",
            _utc_timestamp("start_timestamp", self.start_timestamp, context=context),
        )
        object.__setattr__(
            self,
            "end_timestamp",
            _utc_timestamp("end_timestamp", self.end_timestamp, context=context),
        )
        object.__setattr__(
            self,
            "generation_timestamp_utc",
            _utc_timestamp(
                "generation_timestamp_utc",
                self.generation_timestamp_utc,
                context=context,
            ),
        )
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int):
            raise PredictionArtifactError(
                f"{context}: field 'row_count' must be a positive integer, "
                f"got {self.row_count!r}"
            )
        if self.row_count <= 0:
            raise PredictionArtifactError(
                f"{context}: field 'row_count' must be positive, got {self.row_count!r}"
            )
        for field in ("threshold_selection_used", "allocation_mapping_used"):
            if type(getattr(self, field)) is not bool:
                raise PredictionArtifactError(
                    f"{context}: field {field!r} must be boolean, "
                    f"got {getattr(self, field)!r}"
                )
        if not isinstance(self.notes, str):
            raise PredictionArtifactError(
                f"{context}: field 'notes' must be text, got {self.notes!r}"
            )


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """One immutable set of prediction rows and reconciled metadata."""

    predictions: tuple[Prediction, ...]
    metadata: PredictionArtifactMetadata

    def __post_init__(self) -> None:
        if type(self.predictions) is not tuple or not self.predictions:
            raise PredictionArtifactError(
                "PredictionArtifact.predictions must be a non-empty tuple"
            )
        if any(type(item) is not Prediction for item in self.predictions):
            raise PredictionArtifactError(
                "PredictionArtifact.predictions must contain exactly Prediction objects"
            )
        if type(self.metadata) is not PredictionArtifactMetadata:
            raise PredictionArtifactError(
                "PredictionArtifact.metadata must be exactly PredictionArtifactMetadata"
            )


def load_predictions_csv(path: str | Path) -> tuple[Prediction, ...]:
    """Load prediction rows exactly as ordered in a strict CSV artifact."""

    csv_path = Path(path)
    rows = _read_csv_rows(csv_path, PREDICTION_COLUMNS)
    predictions: list[Prediction] = []
    previous_timestamp: datetime | None = None
    expected_symbol: str | None = None
    expected_model_id: str | None = None
    expected_period_label: str | None = None

    for row_number, row in enumerate(rows, start=2):
        context = f"{_filename(csv_path)}: row {row_number}"
        timestamp = _parse_utc_timestamp(
            "prediction_timestamp", row[0], context=context
        )
        symbol = _required_text("symbol", row[1], context=context)
        probability = _parse_probability(row[2], context=context)
        model_id = _required_text("model_id", row[3], context=context)
        period_label = _required_text("period_label", row[4], context=context)

        if previous_timestamp is not None:
            if timestamp == previous_timestamp:
                raise PredictionArtifactError(
                    f"{context}: duplicate prediction_timestamp {timestamp.isoformat()}"
                )
            if timestamp < previous_timestamp:
                raise PredictionArtifactError(
                    f"{context}: prediction_timestamp {timestamp.isoformat()} is not "
                    f"strictly later than {previous_timestamp.isoformat()}"
                )
        if expected_symbol is not None and symbol != expected_symbol:
            raise PredictionArtifactError(
                f"{context}: field 'symbol' {symbol!r} does not match first-row "
                f"symbol {expected_symbol!r}"
            )
        if expected_model_id is not None and model_id != expected_model_id:
            raise PredictionArtifactError(
                f"{context}: field 'model_id' {model_id!r} does not match first-row "
                f"model_id {expected_model_id!r}"
            )
        if expected_period_label is not None and period_label != expected_period_label:
            raise PredictionArtifactError(
                f"{context}: field 'period_label' {period_label!r} does not match "
                f"first-row period_label {expected_period_label!r}"
            )

        predictions.append(
            Prediction(timestamp, symbol, probability, model_id, period_label)
        )
        previous_timestamp = timestamp
        expected_symbol = expected_symbol or symbol
        expected_model_id = expected_model_id or model_id
        expected_period_label = expected_period_label or period_label

    return tuple(predictions)


def load_prediction_metadata_csv(
    path: str | Path,
) -> PredictionArtifactMetadata:
    """Load exactly one strictly typed prediction metadata row."""

    csv_path = Path(path)
    rows = _read_csv_rows(csv_path, PREDICTION_METADATA_COLUMNS)
    if len(rows) != 1:
        raise PredictionArtifactError(
            f"{_filename(csv_path)}: metadata CSV must contain exactly one data row, "
            f"got {len(rows)}"
        )
    row = rows[0]
    context = f"{_filename(csv_path)}: row 2"
    values = dict(zip(PREDICTION_METADATA_COLUMNS, row, strict=True))

    return PredictionArtifactMetadata(
        artifact_id=_required_text("artifact_id", values["artifact_id"], context=context),
        prediction_filename=_required_text(
            "prediction_filename", values["prediction_filename"], context=context
        ),
        prediction_sha256=_prediction_sha256(
            values["prediction_sha256"], context=context
        ),
        symbol=_required_text("symbol", values["symbol"], context=context),
        bar_interval=_required_text(
            "bar_interval", values["bar_interval"], context=context
        ),
        timestamp_semantics=_required_text(
            "timestamp_semantics", values["timestamp_semantics"], context=context
        ),
        start_timestamp=_parse_utc_timestamp(
            "start_timestamp", values["start_timestamp"], context=context
        ),
        end_timestamp=_parse_utc_timestamp(
            "end_timestamp", values["end_timestamp"], context=context
        ),
        row_count=_positive_integer("row_count", values["row_count"], context=context),
        model_id=_required_text("model_id", values["model_id"], context=context),
        model_version=_required_text(
            "model_version", values["model_version"], context=context
        ),
        data_snapshot_id=_required_text(
            "data_snapshot_id", values["data_snapshot_id"], context=context
        ),
        generation_timestamp_utc=_parse_utc_timestamp(
            "generation_timestamp_utc",
            values["generation_timestamp_utc"],
            context=context,
        ),
        period_label=_required_text(
            "period_label", values["period_label"], context=context
        ),
        feature_set_id=_required_text(
            "feature_set_id", values["feature_set_id"], context=context
        ),
        threshold_selection_used=_explicit_boolean(
            "threshold_selection_used",
            values["threshold_selection_used"],
            context=context,
        ),
        allocation_mapping_used=_explicit_boolean(
            "allocation_mapping_used",
            values["allocation_mapping_used"],
            context=context,
        ),
        final_test_status=_required_text(
            "final_test_status", values["final_test_status"], context=context
        ),
        notes=values["notes"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PredictionArtifactError(
            f"{_filename(path)}: cannot read prediction bytes for SHA-256: {error}"
        ) from error
    return digest.hexdigest()


def load_prediction_artifact(
    *,
    predictions_path: str | Path,
    metadata_path: str | Path,
) -> PredictionArtifact:
    """Load and reconcile prediction rows with immutable artifact metadata."""

    prediction_path = Path(predictions_path)
    predictions = load_predictions_csv(prediction_path)
    metadata = load_prediction_metadata_csv(metadata_path)
    first = predictions[0]
    last = predictions[-1]

    checks: tuple[tuple[str, object, object], ...] = (
        ("prediction_filename", metadata.prediction_filename, prediction_path.name),
        ("prediction_sha256", metadata.prediction_sha256, _sha256(prediction_path)),
        ("symbol", metadata.symbol, first.symbol),
        ("model_id", metadata.model_id, first.model_id),
        ("period_label", metadata.period_label, first.period_label),
        ("start_timestamp", metadata.start_timestamp, first.timestamp),
        ("end_timestamp", metadata.end_timestamp, last.timestamp),
        ("row_count", metadata.row_count, len(predictions)),
        ("bar_interval", metadata.bar_interval, _REQUIRED_BAR_INTERVAL),
        (
            "timestamp_semantics",
            metadata.timestamp_semantics,
            _REQUIRED_TIMESTAMP_SEMANTICS,
        ),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise PredictionArtifactError(
                f"metadata field {field!r} mismatch: got {actual!r}, "
                f"expected {expected!r}"
            )

    return PredictionArtifact(predictions=predictions, metadata=metadata)


def _validated_bar_timestamps(bars: object) -> tuple[datetime, ...]:
    if isinstance(bars, _STRING_LIKE) or not isinstance(bars, Sequence):
        raise PredictionArtifactError("bars must be a non-string sequence of Bar objects")
    if len(bars) == 0:
        raise PredictionArtifactError("bars must contain at least one Bar")

    timestamps: list[datetime] = []
    for index, bar in enumerate(bars):
        if type(bar) is not Bar:
            raise PredictionArtifactError(
                f"bars[{index}] must be exactly Bar, got {bar!r}"
            )
        timestamp = bar.timestamp
        if not isinstance(timestamp, datetime):
            raise PredictionArtifactError(
                f"bars[{index}].timestamp must be a datetime, got {timestamp!r}"
            )
        try:
            offset = timestamp.utcoffset()
        except (OverflowError, ValueError) as error:
            raise PredictionArtifactError(
                f"bars[{index}].timestamp has an invalid UTC offset: {timestamp!r}"
            ) from error
        if timestamp.tzinfo is None or offset is None:
            raise PredictionArtifactError(
                f"bars[{index}].timestamp must include timezone information"
            )
        normalized = timestamp.astimezone(timezone.utc)
        if timestamps:
            difference = normalized - timestamps[-1]
            if difference <= timedelta(0):
                raise PredictionArtifactError(
                    f"bars[{index}].timestamp must be strictly later than "
                    f"bars[{index - 1}].timestamp"
                )
            if difference != _BAR_INTERVAL:
                raise PredictionArtifactError(
                    f"bars[{index}].timestamp must be exactly one hour after "
                    f"bars[{index - 1}].timestamp"
                )
        timestamps.append(normalized)
    return tuple(timestamps)


def _validated_predictions(predictions: object) -> tuple[Prediction, ...]:
    if isinstance(predictions, _STRING_LIKE) or not isinstance(predictions, Sequence):
        raise PredictionArtifactError(
            "predictions must be a non-string sequence of Prediction objects"
        )
    if len(predictions) == 0:
        raise PredictionArtifactError("predictions must contain at least one Prediction")

    validated: list[Prediction] = []
    for index, prediction in enumerate(predictions):
        if type(prediction) is not Prediction:
            raise PredictionArtifactError(
                f"predictions[{index}] must be exactly Prediction, got {prediction!r}"
            )
        if validated:
            if prediction.timestamp == validated[-1].timestamp:
                raise PredictionArtifactError(
                    f"predictions[{index}].timestamp duplicates the preceding timestamp"
                )
            if prediction.timestamp < validated[-1].timestamp:
                raise PredictionArtifactError(
                    f"predictions[{index}].timestamp must be strictly later than "
                    f"predictions[{index - 1}].timestamp"
                )
            if prediction.symbol != validated[0].symbol:
                raise PredictionArtifactError(
                    f"predictions[{index}].symbol does not match predictions[0].symbol"
                )
            if prediction.model_id != validated[0].model_id:
                raise PredictionArtifactError(
                    f"predictions[{index}].model_id does not match predictions[0].model_id"
                )
            if prediction.period_label != validated[0].period_label:
                raise PredictionArtifactError(
                    f"predictions[{index}].period_label does not match "
                    "predictions[0].period_label"
                )
        validated.append(prediction)
    return tuple(validated)


def align_predictions_to_bars(
    *,
    predictions: Sequence[Prediction],
    bars: Sequence[Bar],
    expected_symbol: str | None = None,
) -> tuple[float, ...]:
    """Return probabilities only after exact one-to-one decision-bar alignment."""

    validated_predictions = _validated_predictions(predictions)
    bar_timestamps = _validated_bar_timestamps(bars)
    if len(validated_predictions) != len(bar_timestamps):
        raise PredictionArtifactError(
            "predictions length must equal bars length, "
            f"got predictions={len(validated_predictions)}, bars={len(bar_timestamps)}"
        )

    if expected_symbol is not None:
        symbol = _required_text(
            "expected_symbol", expected_symbol, context="alignment"
        )
        if validated_predictions[0].symbol != symbol:
            raise PredictionArtifactError(
                f"prediction symbol {validated_predictions[0].symbol!r} does not match "
                f"expected_symbol {symbol!r}"
            )

    probabilities: list[float] = []
    for index, (prediction, bar_timestamp) in enumerate(
        zip(validated_predictions, bar_timestamps, strict=True)
    ):
        if prediction.timestamp != bar_timestamp:
            raise PredictionArtifactError(
                f"predictions[{index}].timestamp {prediction.timestamp.isoformat()} "
                f"does not exactly match bars[{index}].timestamp "
                f"{bar_timestamp.isoformat()}"
            )
        probabilities.append(prediction.probability)
    return tuple(probabilities)
