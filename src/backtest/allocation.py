"""Deterministic conversion of precomputed probabilities to target weights."""

import math
from collections.abc import Sequence

from backtest.positioning import TargetWeight

_STRING_LIKE = (str, bytes, bytearray)


class AllocationError(ValueError):
    """Raised when allocation-policy inputs are invalid."""


def _probability_value(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AllocationError(
            f"{field} must be numeric and not boolean, got {value!r}"
        )

    number = float(value)
    if not math.isfinite(number):
        raise AllocationError(f"{field} must be finite, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise AllocationError(
            f"{field} must lie within the inclusive range [0.0, 1.0], "
            f"got {number!r}"
        )
    return number


def probabilities_to_target_weights(
    probabilities: Sequence[float],
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> tuple[TargetWeight, ...]:
    """Map precomputed probabilities through a fixed three-region policy.

    Thresholds are already-fixed policy inputs. Values below the lower
    threshold map to 0.0, values within the inclusive threshold interval map
    to 0.5, and values above the upper threshold map to 1.0.
    """

    if isinstance(probabilities, _STRING_LIKE) or not isinstance(
        probabilities, Sequence
    ):
        raise AllocationError("probabilities must be a non-string sequence")
    if len(probabilities) == 0:
        raise AllocationError("probabilities must contain at least one value")

    lower = _probability_value("lower_threshold", lower_threshold)
    upper = _probability_value("upper_threshold", upper_threshold)
    if lower > upper:
        raise AllocationError(
            "lower_threshold must be less than or equal to upper_threshold, "
            f"got lower_threshold={lower!r}, upper_threshold={upper!r}"
        )

    weights: list[TargetWeight] = []
    for index, value in enumerate(probabilities):
        probability = _probability_value(f"probabilities[{index}]", value)
        if probability < lower:
            weight = 0.0
        elif probability <= upper:
            weight = 0.5
        else:
            weight = 1.0
        weights.append(TargetWeight(weight))

    return tuple(weights)
