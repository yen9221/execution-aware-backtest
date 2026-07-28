"""Strict loading and validation for predefined allocation-policy freezes."""

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.predictions import PredictionArtifact

POLICY_SPEC_COLUMNS = (
    "policy_spec_id",
    "freeze_status",
    "created_at_utc",
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
    "policy_id",
    "mapping_type",
    "mapping_formula",
    "threshold_selection_used",
    "lower_threshold",
    "upper_threshold",
    "maximum_target_weight",
    "initial_cash",
    "fee_rate",
    "slippage_rate",
    "rebalance_tolerance",
    "minimum_trade_notional",
    "position_domain",
    "short_selling_allowed",
    "leverage_allowed",
    "execution_timing",
    "final_target_execution",
    "baseline_1",
    "baseline_1_definition",
    "baseline_2",
    "baseline_2_definition",
    "output_schema_version",
    "post_result_parameter_changes_allowed",
    "performance_based_policy_selection_allowed",
    "notes",
)

COMPLETE = "complete"
INCOMPLETE = "incomplete_missing_prediction_artifact"
POLICY_ID = "continuous_linear_long_only_v1"
MAPPING_TYPE = "continuous_linear"
MAPPING_FORMULA = "min(1.0,max(0.0,2.0*probability-1.0))"
MAPPING_IMPLEMENTATION = (
    "backtest.allocation.probabilities_to_continuous_target_weights"
)
OUTPUT_SCHEMA_VERSION = "ml_execution_demo_v1"
STAGE9_OUTPUT_FILES = (
    "summary.csv",
    "trades.csv",
    "portfolio_history.csv",
    "targets.csv",
    "metadata.csv",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*")
_PLACEHOLDERS = {"tbd", "todo", "unknown", "<sha256>", "placeholder"}
_PREDICTION_IDENTITY_FIELDS = (
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
)


class PolicySpecError(ValueError):
    """Raised when an allocation-policy specification is invalid."""


def _filename(path: Path) -> str:
    return path.name or str(path)


def _required_text(field: str, value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise PolicySpecError(
            f"{context}: field {field!r} must be text, got {value!r}"
        )
    if not value.strip():
        raise PolicySpecError(f"{context}: field {field!r} is empty")
    if value != value.strip():
        raise PolicySpecError(
            f"{context}: field {field!r} must not contain surrounding whitespace: "
            f"{value!r}"
        )
    return value


def _optional_empty_text(field: str, value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise PolicySpecError(
            f"{context}: field {field!r} must be text, got {value!r}"
        )
    if value != "":
        raise PolicySpecError(
            f"{context}: incomplete specification field {field!r} must be empty, "
            f"got {value!r}"
        )
    return value


def _reject_placeholder(field: str, value: str, *, context: str) -> str:
    if value.lower() in _PLACEHOLDERS:
        raise PolicySpecError(
            f"{context}: complete specification field {field!r} must not use "
            f"placeholder value {value!r}"
        )
    return value


def _utc_timestamp(field: str, value: object, *, context: str) -> datetime:
    if not isinstance(value, datetime):
        raise PolicySpecError(
            f"{context}: field {field!r} must be a datetime, got {value!r}"
        )
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise PolicySpecError(
            f"{context}: field {field!r} has an invalid UTC offset: {value!r}"
        ) from error
    if value.tzinfo is None or offset is None:
        raise PolicySpecError(
            f"{context}: field {field!r} must include timezone information: {value!r}"
        )
    if offset != timedelta(0):
        raise PolicySpecError(
            f"{context}: field {field!r} must use UTC, got offset {offset}"
        )
    return value.astimezone(timezone.utc)


def _parse_utc_timestamp(field: str, value: str, *, context: str) -> datetime:
    text = _required_text(field, value, context=context)
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError as error:
        raise PolicySpecError(
            f"{context}: field {field!r} is not a valid ISO 8601 datetime: {value!r}"
        ) from error
    return _utc_timestamp(field, timestamp, context=context)


def _sha256(field: str, value: object, *, context: str) -> str:
    text = _required_text(field, value, context=context)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise PolicySpecError(
            f"{context}: field {field!r} must be exactly 64 lowercase hexadecimal "
            f"characters, got {value!r}"
        )
    return text


def _positive_integer(field: str, value: str, *, context: str) -> int:
    text = _required_text(field, value, context=context)
    if _POSITIVE_INTEGER_PATTERN.fullmatch(text) is None:
        raise PolicySpecError(
            f"{context}: field {field!r} must be a positive integer, got {value!r}"
        )
    return int(text)


def _finite_number(field: str, value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicySpecError(
            f"{context}: field {field!r} must be numeric and not boolean, "
            f"got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PolicySpecError(
            f"{context}: field {field!r} must be finite, got {value!r}"
        )
    return number


def _parse_number(field: str, value: str, *, context: str) -> float:
    text = _required_text(field, value, context=context)
    try:
        number = float(text)
    except ValueError as error:
        raise PolicySpecError(
            f"{context}: field {field!r} is not numeric: {value!r}"
        ) from error
    return _finite_number(field, number, context=context)


def _explicit_boolean(field: str, value: object, *, context: str) -> bool:
    if value == "True" or value is True:
        return True
    if value == "False" or value is False:
        return False
    raise PolicySpecError(
        f"{context}: field {field!r} must be exactly 'True' or 'False', "
        f"got {value!r}"
    )


def _exact(field: str, actual: object, expected: object, *, context: str) -> None:
    if actual != expected:
        raise PolicySpecError(
            f"{context}: field {field!r} must equal {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True, slots=True)
class FrozenPolicySpec:
    """One immutable predefined allocation-policy freeze specification."""

    policy_spec_id: str
    freeze_status: str
    created_at_utc: datetime
    prediction_artifact_id: str
    prediction_filename: str
    prediction_sha256: str
    prediction_metadata_filename: str
    prediction_metadata_sha256: str
    model_id: str
    model_version: str
    data_snapshot_id: str
    feature_set_id: str
    period_label: str
    prediction_start_timestamp: datetime | None
    prediction_end_timestamp: datetime | None
    prediction_row_count: int | None
    final_test_status: str
    policy_id: str
    mapping_type: str
    mapping_formula: str
    threshold_selection_used: bool
    lower_threshold: str
    upper_threshold: str
    maximum_target_weight: float
    initial_cash: float
    fee_rate: float
    slippage_rate: float
    rebalance_tolerance: float
    minimum_trade_notional: float
    position_domain: str
    short_selling_allowed: bool
    leverage_allowed: bool
    execution_timing: str
    final_target_execution: str
    baseline_1: str
    baseline_1_definition: str
    baseline_2: str
    baseline_2_definition: str
    output_schema_version: str
    post_result_parameter_changes_allowed: bool
    performance_based_policy_selection_allowed: bool
    notes: str

    def __post_init__(self) -> None:
        _validate_spec(self, context="FrozenPolicySpec")


def _validate_spec(spec: FrozenPolicySpec, *, context: str) -> None:
    if type(spec) is not FrozenPolicySpec:
        raise PolicySpecError(
            f"{context}: spec must be exactly FrozenPolicySpec, got {spec!r}"
        )
    _required_text("policy_spec_id", spec.policy_spec_id, context=context)
    if spec.freeze_status not in (COMPLETE, INCOMPLETE):
        raise PolicySpecError(
            f"{context}: field 'freeze_status' must be {COMPLETE!r} or "
            f"{INCOMPLETE!r}, got {spec.freeze_status!r}"
        )
    _utc_timestamp("created_at_utc", spec.created_at_utc, context=context)

    if spec.freeze_status == COMPLETE:
        for field in (
            "prediction_artifact_id",
            "prediction_filename",
            "prediction_metadata_filename",
            "model_id",
            "model_version",
            "data_snapshot_id",
            "feature_set_id",
            "period_label",
            "final_test_status",
        ):
            value = _required_text(field, getattr(spec, field), context=context)
            _reject_placeholder(field, value, context=context)
        _sha256("prediction_sha256", spec.prediction_sha256, context=context)
        _sha256(
            "prediction_metadata_sha256",
            spec.prediction_metadata_sha256,
            context=context,
        )
        start = _utc_timestamp(
            "prediction_start_timestamp",
            spec.prediction_start_timestamp,
            context=context,
        )
        end = _utc_timestamp(
            "prediction_end_timestamp",
            spec.prediction_end_timestamp,
            context=context,
        )
        if start > end:
            raise PolicySpecError(
                f"{context}: prediction_start_timestamp must be before or equal to "
                "prediction_end_timestamp"
            )
        if (
            isinstance(spec.prediction_row_count, bool)
            or not isinstance(spec.prediction_row_count, int)
            or spec.prediction_row_count <= 0
        ):
            raise PolicySpecError(
                f"{context}: field 'prediction_row_count' must be a positive integer, "
                f"got {spec.prediction_row_count!r}"
            )
    else:
        for field in _PREDICTION_IDENTITY_FIELDS:
            value = getattr(spec, field)
            if field in (
                "prediction_start_timestamp",
                "prediction_end_timestamp",
                "prediction_row_count",
            ):
                if value is not None:
                    raise PolicySpecError(
                        f"{context}: incomplete specification field {field!r} "
                        f"must be empty, got {value!r}"
                    )
            else:
                _optional_empty_text(field, value, context=context)

    for field in (
        "policy_id",
        "mapping_type",
        "mapping_formula",
        "position_domain",
        "execution_timing",
        "final_target_execution",
        "baseline_1",
        "baseline_1_definition",
        "baseline_2",
        "baseline_2_definition",
        "output_schema_version",
    ):
        _required_text(field, getattr(spec, field), context=context)
    if spec.lower_threshold != "" or spec.upper_threshold != "":
        raise PolicySpecError(
            f"{context}: continuous policy thresholds must be empty, got "
            f"lower_threshold={spec.lower_threshold!r}, "
            f"upper_threshold={spec.upper_threshold!r}"
        )
    for field in (
        "threshold_selection_used",
        "short_selling_allowed",
        "leverage_allowed",
        "post_result_parameter_changes_allowed",
        "performance_based_policy_selection_allowed",
    ):
        if type(getattr(spec, field)) is not bool:
            raise PolicySpecError(
                f"{context}: field {field!r} must be boolean, "
                f"got {getattr(spec, field)!r}"
            )

    numbers = {
        field: _finite_number(field, getattr(spec, field), context=context)
        for field in (
            "maximum_target_weight",
            "initial_cash",
            "fee_rate",
            "slippage_rate",
            "rebalance_tolerance",
            "minimum_trade_notional",
        )
    }
    if not 0.0 <= numbers["maximum_target_weight"] <= 1.0:
        raise PolicySpecError(
            f"{context}: field 'maximum_target_weight' must lie within [0.0, 1.0], "
            f"got {numbers['maximum_target_weight']!r}"
        )
    for field in (
        "initial_cash",
        "fee_rate",
        "slippage_rate",
        "rebalance_tolerance",
        "minimum_trade_notional",
    ):
        if numbers[field] < 0:
            raise PolicySpecError(
                f"{context}: field {field!r} must be non-negative, "
                f"got {numbers[field]!r}"
            )

    fixed = {
        "policy_id": POLICY_ID,
        "mapping_type": MAPPING_TYPE,
        "mapping_formula": MAPPING_FORMULA,
        "threshold_selection_used": False,
        "maximum_target_weight": 1.0,
        "initial_cash": 1000.0,
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "rebalance_tolerance": 0.0,
        "minimum_trade_notional": 0.0,
        "position_domain": "long_only",
        "short_selling_allowed": False,
        "leverage_allowed": False,
        "execution_timing": "decision_after_bar_close_execute_next_bar_open",
        "final_target_execution": "prohibited_without_next_bar",
        "baseline_1": "execution_aligned_buy_and_hold",
        "baseline_1_definition": "target[0]=0.0;target[1:]=1.0",
        "baseline_2": "zero_position",
        "baseline_2_definition": "all_target_weights=0.0",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "post_result_parameter_changes_allowed": False,
        "performance_based_policy_selection_allowed": False,
    }
    for field, expected in fixed.items():
        _exact(field, getattr(spec, field), expected, context=context)
    if not isinstance(spec.notes, str):
        raise PolicySpecError(
            f"{context}: field 'notes' must be text, got {spec.notes!r}"
        )


def _read_one_row(path: str | Path) -> dict[str, str]:
    csv_path = Path(path)
    name = _filename(csv_path)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise PolicySpecError(f"{name}: empty CSV file") from error
            if tuple(header) != POLICY_SPEC_COLUMNS:
                raise PolicySpecError(
                    f"{name}: invalid CSV schema; expected "
                    f"{','.join(POLICY_SPEC_COLUMNS)!r}, got {','.join(header)!r}"
                )
            rows = list(reader)
    except PolicySpecError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise PolicySpecError(
            f"{name}: cannot read valid UTF-8 CSV: {error}"
        ) from error
    if not rows:
        raise PolicySpecError(f"{name}: CSV file contains a header but no data rows")
    if len(rows) != 1:
        raise PolicySpecError(
            f"{name}: policy specification CSV must contain exactly one data row, "
            f"got {len(rows)}"
        )
    if len(rows[0]) != len(POLICY_SPEC_COLUMNS):
        raise PolicySpecError(
            f"{name}: row 2: expected {len(POLICY_SPEC_COLUMNS)} values, "
            f"got {len(rows[0])}"
        )
    return dict(zip(POLICY_SPEC_COLUMNS, rows[0], strict=True))


def load_policy_spec_csv(path: str | Path) -> FrozenPolicySpec:
    """Load exactly one predefined allocation-policy freeze specification."""

    values = _read_one_row(path)
    context = f"{_filename(Path(path))}: row 2"
    status = _required_text("freeze_status", values["freeze_status"], context=context)
    if status not in (COMPLETE, INCOMPLETE):
        raise PolicySpecError(
            f"{context}: field 'freeze_status' must be {COMPLETE!r} or "
            f"{INCOMPLETE!r}, got {status!r}"
        )

    if status == COMPLETE:
        identity_text = {
            field: _reject_placeholder(
                field,
                _required_text(field, values[field], context=context),
                context=context,
            )
            for field in (
                "prediction_artifact_id",
                "prediction_filename",
                "prediction_metadata_filename",
                "model_id",
                "model_version",
                "data_snapshot_id",
                "feature_set_id",
                "period_label",
                "final_test_status",
            )
        }
        prediction_sha256 = _sha256(
            "prediction_sha256", values["prediction_sha256"], context=context
        )
        prediction_metadata_sha256 = _sha256(
            "prediction_metadata_sha256",
            values["prediction_metadata_sha256"],
            context=context,
        )
        prediction_start = _parse_utc_timestamp(
            "prediction_start_timestamp",
            values["prediction_start_timestamp"],
            context=context,
        )
        prediction_end = _parse_utc_timestamp(
            "prediction_end_timestamp",
            values["prediction_end_timestamp"],
            context=context,
        )
        prediction_count = _positive_integer(
            "prediction_row_count",
            values["prediction_row_count"],
            context=context,
        )
    else:
        for field in _PREDICTION_IDENTITY_FIELDS:
            _optional_empty_text(field, values[field], context=context)
        identity_text = {
            field: ""
            for field in (
                "prediction_artifact_id",
                "prediction_filename",
                "prediction_metadata_filename",
                "model_id",
                "model_version",
                "data_snapshot_id",
                "feature_set_id",
                "period_label",
                "final_test_status",
            )
        }
        prediction_sha256 = ""
        prediction_metadata_sha256 = ""
        prediction_start = None
        prediction_end = None
        prediction_count = None

    return FrozenPolicySpec(
        policy_spec_id=_required_text(
            "policy_spec_id", values["policy_spec_id"], context=context
        ),
        freeze_status=status,
        created_at_utc=_parse_utc_timestamp(
            "created_at_utc", values["created_at_utc"], context=context
        ),
        prediction_artifact_id=identity_text["prediction_artifact_id"],
        prediction_filename=identity_text["prediction_filename"],
        prediction_sha256=prediction_sha256,
        prediction_metadata_filename=identity_text["prediction_metadata_filename"],
        prediction_metadata_sha256=prediction_metadata_sha256,
        model_id=identity_text["model_id"],
        model_version=identity_text["model_version"],
        data_snapshot_id=identity_text["data_snapshot_id"],
        feature_set_id=identity_text["feature_set_id"],
        period_label=identity_text["period_label"],
        prediction_start_timestamp=prediction_start,
        prediction_end_timestamp=prediction_end,
        prediction_row_count=prediction_count,
        final_test_status=identity_text["final_test_status"],
        policy_id=_required_text("policy_id", values["policy_id"], context=context),
        mapping_type=_required_text(
            "mapping_type", values["mapping_type"], context=context
        ),
        mapping_formula=_required_text(
            "mapping_formula", values["mapping_formula"], context=context
        ),
        threshold_selection_used=_explicit_boolean(
            "threshold_selection_used",
            values["threshold_selection_used"],
            context=context,
        ),
        lower_threshold=values["lower_threshold"],
        upper_threshold=values["upper_threshold"],
        maximum_target_weight=_parse_number(
            "maximum_target_weight", values["maximum_target_weight"], context=context
        ),
        initial_cash=_parse_number("initial_cash", values["initial_cash"], context=context),
        fee_rate=_parse_number("fee_rate", values["fee_rate"], context=context),
        slippage_rate=_parse_number(
            "slippage_rate", values["slippage_rate"], context=context
        ),
        rebalance_tolerance=_parse_number(
            "rebalance_tolerance", values["rebalance_tolerance"], context=context
        ),
        minimum_trade_notional=_parse_number(
            "minimum_trade_notional",
            values["minimum_trade_notional"],
            context=context,
        ),
        position_domain=_required_text(
            "position_domain", values["position_domain"], context=context
        ),
        short_selling_allowed=_explicit_boolean(
            "short_selling_allowed", values["short_selling_allowed"], context=context
        ),
        leverage_allowed=_explicit_boolean(
            "leverage_allowed", values["leverage_allowed"], context=context
        ),
        execution_timing=_required_text(
            "execution_timing", values["execution_timing"], context=context
        ),
        final_target_execution=_required_text(
            "final_target_execution", values["final_target_execution"], context=context
        ),
        baseline_1=_required_text("baseline_1", values["baseline_1"], context=context),
        baseline_1_definition=_required_text(
            "baseline_1_definition", values["baseline_1_definition"], context=context
        ),
        baseline_2=_required_text("baseline_2", values["baseline_2"], context=context),
        baseline_2_definition=_required_text(
            "baseline_2_definition", values["baseline_2_definition"], context=context
        ),
        output_schema_version=_required_text(
            "output_schema_version", values["output_schema_version"], context=context
        ),
        post_result_parameter_changes_allowed=_explicit_boolean(
            "post_result_parameter_changes_allowed",
            values["post_result_parameter_changes_allowed"],
            context=context,
        ),
        performance_based_policy_selection_allowed=_explicit_boolean(
            "performance_based_policy_selection_allowed",
            values["performance_based_policy_selection_allowed"],
            context=context,
        ),
        notes=values["notes"],
    )


def validate_stage9_eligibility(spec: FrozenPolicySpec) -> None:
    """Reject any policy specification not complete and internally frozen."""

    _validate_spec(spec, context="Stage 9 eligibility")
    if spec.freeze_status != COMPLETE:
        raise PolicySpecError(
            "Stage 9 eligibility requires freeze_status='complete'; the prediction "
            "artifact identity is still missing"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PolicySpecError(
            f"{_filename(path)}: cannot read metadata bytes for SHA-256: {error}"
        ) from error
    return digest.hexdigest()


def validate_policy_spec_against_prediction_artifact(
    *,
    spec: FrozenPolicySpec,
    artifact: PredictionArtifact,
    prediction_metadata_path: str | Path,
) -> None:
    """Reconcile one complete freeze with one immutable prediction artifact."""

    validate_stage9_eligibility(spec)
    if type(artifact) is not PredictionArtifact:
        raise PolicySpecError(
            f"artifact must be exactly PredictionArtifact, got {artifact!r}"
        )
    metadata_path = Path(prediction_metadata_path)
    metadata = artifact.metadata
    checks: tuple[tuple[str, object, object], ...] = (
        ("prediction_artifact_id", spec.prediction_artifact_id, metadata.artifact_id),
        ("prediction_filename", spec.prediction_filename, metadata.prediction_filename),
        ("prediction_sha256", spec.prediction_sha256, metadata.prediction_sha256),
        (
            "prediction_metadata_filename",
            spec.prediction_metadata_filename,
            metadata_path.name,
        ),
        (
            "prediction_metadata_sha256",
            spec.prediction_metadata_sha256,
            _file_sha256(metadata_path),
        ),
        ("model_id", spec.model_id, metadata.model_id),
        ("model_version", spec.model_version, metadata.model_version),
        ("data_snapshot_id", spec.data_snapshot_id, metadata.data_snapshot_id),
        ("feature_set_id", spec.feature_set_id, metadata.feature_set_id),
        ("period_label", spec.period_label, metadata.period_label),
        (
            "prediction_start_timestamp",
            spec.prediction_start_timestamp,
            metadata.start_timestamp,
        ),
        (
            "prediction_end_timestamp",
            spec.prediction_end_timestamp,
            metadata.end_timestamp,
        ),
        ("prediction_row_count", spec.prediction_row_count, metadata.row_count),
        ("final_test_status", spec.final_test_status, metadata.final_test_status),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise PolicySpecError(
                f"policy specification field {field!r} mismatch: got {actual!r}, "
                f"expected {expected!r}"
            )
    expected_symbol = metadata.symbol
    if any(prediction.symbol != expected_symbol for prediction in artifact.predictions):
        raise PolicySpecError(
            "prediction artifact symbol mismatch between prediction rows and metadata"
        )
