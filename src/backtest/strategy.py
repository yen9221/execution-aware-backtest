"""Deterministic target generation from completed chronological bars."""

import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from backtest.events import Bar
from backtest.positioning import TargetPosition, TargetWeight

_BAR_INTERVAL = timedelta(hours=1)
_STRING_LIKE = (str, bytes, bytearray)


class StrategyError(ValueError):
    """Raised when bars cannot produce an unambiguous target sequence."""


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise StrategyError(f"{field} must be a datetime, got {value!r}")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise StrategyError(f"{field} has an invalid UTC offset: {value!r}") from error
    if value.tzinfo is None or offset is None:
        raise StrategyError(f"{field} must include timezone information, got {value!r}")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise StrategyError(f"{field} cannot be normalized to UTC: {value!r}") from error


def _positive_close(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyError(f"{field} must be numeric and not boolean, got {value!r}")
    close = float(value)
    if not math.isfinite(close):
        raise StrategyError(f"{field} must be finite, got {value!r}")
    if close <= 0:
        raise StrategyError(f"{field} must be strictly positive, got {close!r}")
    return close


def _validated_closes(bars: object) -> tuple[float, ...]:
    if isinstance(bars, _STRING_LIKE) or not isinstance(bars, Sequence):
        raise StrategyError("bars must be a non-string sequence of Bar objects")
    if len(bars) == 0:
        raise StrategyError("bars must contain at least one Bar")

    closes: list[float] = []
    previous_timestamp: datetime | None = None
    for index, bar in enumerate(bars):
        if not isinstance(bar, Bar):
            raise StrategyError(f"bars[{index}] must be a Bar, got {bar!r}")

        timestamp = _utc_timestamp(f"bars[{index}].timestamp", bar.timestamp)
        if previous_timestamp is not None:
            if timestamp == previous_timestamp:
                raise StrategyError(
                    f"bars[{index}].timestamp duplicates the preceding UTC instant "
                    f"{timestamp.isoformat()}"
                )
            if timestamp < previous_timestamp:
                raise StrategyError(
                    f"bars[{index}].timestamp {timestamp.isoformat()} is not strictly "
                    f"later than {previous_timestamp.isoformat()}"
                )
            if timestamp - previous_timestamp != _BAR_INTERVAL:
                raise StrategyError(
                    f"bars[{index}].timestamp {timestamp.isoformat()} is not exactly one "
                    f"hour after {previous_timestamp.isoformat()}"
                )

        closes.append(_positive_close(f"bars[{index}].close", bar.close))
        previous_timestamp = timestamp

    return tuple(closes)


def previous_close_momentum_targets(
    bars: Sequence[Bar],
) -> tuple[TargetPosition, ...]:
    """Map completed bars to a one-bar close-momentum target sequence.

    The first completed bar maps to ``CASH``. For each later bar, only that
    bar's close and the immediately preceding completed close are compared:
    a rise maps to ``LONG`` and a flat or falling close maps to ``CASH``.
    Execution timing remains the engine's responsibility.
    """

    closes = _validated_closes(bars)

    targets = [TargetPosition.CASH]
    for index in range(1, len(closes)):
        current_close = closes[index]
        previous_close = closes[index - 1]
        target = (
            TargetPosition.LONG
            if current_close > previous_close
            else TargetPosition.CASH
        )
        targets.append(target)
    return tuple(targets)


def previous_close_momentum_target_weights(
    bars: Sequence[Bar],
) -> tuple[TargetWeight, ...]:
    """Map completed one-bar close momentum to fixed fractional weights.

    The first completed bar maps to ``0.0``. A later rising close maps to
    ``0.75``, a flat close to ``0.50``, and a falling close to ``0.25``.
    Execution timing and portfolio state remain outside the strategy layer.
    """

    closes = _validated_closes(bars)
    targets = [TargetWeight(0.0)]
    for index in range(1, len(closes)):
        current_close = closes[index]
        previous_close = closes[index - 1]
        if current_close > previous_close:
            weight = 0.75
        elif current_close == previous_close:
            weight = 0.50
        else:
            weight = 0.25
        targets.append(TargetWeight(weight))
    return tuple(targets)
