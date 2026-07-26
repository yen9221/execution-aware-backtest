"""Strict timestamp correspondence for precomputed probabilities and bars."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backtest.events import Bar

_BAR_INTERVAL = timedelta(hours=1)
_STRING_LIKE = (str, bytes, bytearray)


class PredictionAlignmentError(ValueError):
    """Raised when timestamped probabilities do not correspond to bars."""


@dataclass(frozen=True, slots=True)
class TimestampedProbability:
    """One precomputed probability associated with one completed bar."""

    timestamp: datetime
    probability: float


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise PredictionAlignmentError(f"{field} must be a datetime, got {value!r}")

    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise PredictionAlignmentError(
            f"{field} has an invalid UTC offset: {value!r}"
        ) from error

    if value.tzinfo is None or offset is None:
        raise PredictionAlignmentError(
            f"{field} must include timezone information, got {value!r}"
        )

    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise PredictionAlignmentError(
            f"{field} cannot be normalized to UTC: {value!r}"
        ) from error


def _probability_value(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionAlignmentError(
            f"{field} must be numeric and not boolean, got {value!r}"
        )

    number = float(value)
    if not math.isfinite(number):
        raise PredictionAlignmentError(f"{field} must be finite, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise PredictionAlignmentError(
            f"{field} must lie within the inclusive range [0.0, 1.0], "
            f"got {number!r}"
        )
    return number


def _bar_timestamps(bars: object) -> tuple[datetime, ...]:
    if isinstance(bars, _STRING_LIKE) or not isinstance(bars, Sequence):
        raise PredictionAlignmentError("bars must be a non-string sequence of Bar objects")
    if len(bars) == 0:
        raise PredictionAlignmentError("bars must contain at least one Bar")

    normalized: list[datetime] = []
    for index, bar in enumerate(bars):
        if type(bar) is not Bar:
            raise PredictionAlignmentError(f"bars[{index}] must be exactly Bar, got {bar!r}")
        timestamp = _utc_timestamp(f"bars[{index}].timestamp", bar.timestamp)
        if normalized:
            difference = timestamp - normalized[-1]
            if difference <= timedelta(0):
                raise PredictionAlignmentError(
                    f"bars[{index}].timestamp must be strictly later than "
                    f"bars[{index - 1}].timestamp"
                )
            if difference != _BAR_INTERVAL:
                raise PredictionAlignmentError(
                    f"bars[{index}].timestamp must be exactly one hour after "
                    f"bars[{index - 1}].timestamp"
                )
        normalized.append(timestamp)
    return tuple(normalized)


def _prediction_values(
    predictions: object,
) -> tuple[tuple[datetime, float], ...]:
    if isinstance(predictions, _STRING_LIKE) or not isinstance(predictions, Sequence):
        raise PredictionAlignmentError(
            "predictions must be a non-string sequence of TimestampedProbability objects"
        )
    if len(predictions) == 0:
        raise PredictionAlignmentError(
            "predictions must contain at least one TimestampedProbability"
        )

    validated: list[tuple[datetime, float]] = []
    for index, prediction in enumerate(predictions):
        if type(prediction) is not TimestampedProbability:
            raise PredictionAlignmentError(
                f"predictions[{index}] must be exactly TimestampedProbability, "
                f"got {prediction!r}"
            )
        timestamp = _utc_timestamp(
            f"predictions[{index}].timestamp", prediction.timestamp
        )
        probability = _probability_value(
            f"predictions[{index}].probability", prediction.probability
        )
        if validated:
            difference = timestamp - validated[-1][0]
            if difference <= timedelta(0):
                raise PredictionAlignmentError(
                    f"predictions[{index}].timestamp must be strictly later than "
                    f"predictions[{index - 1}].timestamp"
                )
            if difference != _BAR_INTERVAL:
                raise PredictionAlignmentError(
                    f"predictions[{index}].timestamp must be exactly one hour after "
                    f"predictions[{index - 1}].timestamp"
                )
        validated.append((timestamp, probability))
    return tuple(validated)


def align_probabilities_to_bars(
    bars: Sequence[Bar],
    predictions: Sequence[TimestampedProbability],
) -> tuple[float, ...]:
    """Validate exact index-by-index timestamp correspondence."""

    normalized_bars = _bar_timestamps(bars)
    validated_predictions = _prediction_values(predictions)
    if len(validated_predictions) != len(normalized_bars):
        raise PredictionAlignmentError(
            "predictions length must equal bars length, "
            f"got predictions={len(validated_predictions)}, bars={len(normalized_bars)}"
        )

    aligned: list[float] = []
    for index, (bar_timestamp, prediction) in enumerate(
        zip(normalized_bars, validated_predictions, strict=True)
    ):
        prediction_timestamp, probability = prediction
        if prediction_timestamp != bar_timestamp:
            raise PredictionAlignmentError(
                f"predictions[{index}].timestamp "
                f"{prediction_timestamp.isoformat()} does not match "
                f"bars[{index}].timestamp {bar_timestamp.isoformat()} after UTC normalization"
            )
        aligned.append(probability)
    return tuple(aligned)
