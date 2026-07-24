"""Target-position intents and execution-time market-order conversion."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum

from backtest.orders import MarketOrder, Side
from backtest.portfolio import PortfolioState

_MAX_TOLERANCE = 1e-6


class PositioningError(ValueError):
    """Raised when a target intent cannot be converted safely."""


class TargetPosition(IntEnum):
    """Supported all-cash or long target positions."""

    CASH = 0
    LONG = 1


@dataclass(frozen=True, slots=True)
class PendingTarget:
    """A price-free target selected from one completed decision bar."""

    decision_bar_timestamp: datetime
    target: TargetPosition


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise PositioningError(f"{field} must be a datetime, got {value!r}")

    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise PositioningError(f"{field} has an invalid UTC offset: {value!r}") from error

    if value.tzinfo is None or offset is None:
        raise PositioningError(
            f"{field} must include timezone information, got {value!r}"
        )

    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise PositioningError(
            f"{field} cannot be normalized to UTC: {value!r}"
        ) from error


def _finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PositioningError(
            f"{field} must be numeric and not boolean, got {value!r}"
        )

    number = float(value)
    if not math.isfinite(number):
        raise PositioningError(f"{field} must be finite, got {value!r}")
    return number


def _nonnegative_number(field: str, value: object) -> float:
    number = _finite_number(field, value)
    if number < 0:
        raise PositioningError(f"{field} must be non-negative, got {number!r}")
    return number


def _validated_tolerance(value: object) -> float:
    tolerance = _nonnegative_number("tolerance", value)
    if tolerance > _MAX_TOLERANCE:
        raise PositioningError(
            f"tolerance must be no greater than {_MAX_TOLERANCE!r}, "
            f"got {tolerance!r}"
        )
    return tolerance


def _validated_target(field: str, value: object) -> TargetPosition:
    if not isinstance(value, TargetPosition):
        raise PositioningError(f"{field} must be a TargetPosition, got {value!r}")
    return value


def create_pending_target(
    *,
    decision_bar_timestamp: datetime,
    target: TargetPosition,
) -> PendingTarget:
    """Create a UTC-normalized, price-free intent from a completed bar."""

    timestamp = _utc_timestamp("decision_bar_timestamp", decision_bar_timestamp)
    target_value = _validated_target("target", target)
    return PendingTarget(decision_bar_timestamp=timestamp, target=target_value)


def market_order_for_target_at_open(
    *,
    state: PortfolioState,
    pending_target: PendingTarget,
    execution_bar_timestamp: datetime,
    reference_open: float,
    fee_rate: float,
    slippage_rate: float,
    tolerance: float = 1e-12,
) -> MarketOrder | None:
    """Convert one target into an order when the execution open is observable.

    Repeated CASH or LONG targets produce no order. For a new LONG target, the
    order uses nearly all available cash after accounting for adverse slippage
    and fees. One representable float step is removed from the theoretical
    maximum quantity to avoid overspending through floating-point rounding.
    This function neither executes the returned order nor mutates the inputs.
    """

    if not isinstance(state, PortfolioState):
        raise PositioningError(f"state must be a PortfolioState, got {state!r}")
    if not isinstance(pending_target, PendingTarget):
        raise PositioningError(
            f"pending_target must be a PendingTarget, got {pending_target!r}"
        )

    cash = _nonnegative_number("state.cash", state.cash)
    position = _nonnegative_number(
        "state.position_quantity", state.position_quantity
    )
    _nonnegative_number("state.cumulative_fees", state.cumulative_fees)
    tolerance_value = _validated_tolerance(tolerance)

    decision_timestamp = _utc_timestamp(
        "pending_target.decision_bar_timestamp",
        pending_target.decision_bar_timestamp,
    )
    target = _validated_target("pending_target.target", pending_target.target)
    execution_timestamp = _utc_timestamp(
        "execution_bar_timestamp", execution_bar_timestamp
    )
    if execution_timestamp <= decision_timestamp:
        relation = "equal to" if execution_timestamp == decision_timestamp else "before"
        raise PositioningError(
            "execution_bar_timestamp must be strictly later than "
            "pending_target.decision_bar_timestamp; execution instant "
            f"{execution_timestamp.isoformat()} is {relation} decision-bar instant "
            f"{decision_timestamp.isoformat()}"
        )

    reference = _finite_number("reference_open", reference_open)
    if reference <= 0:
        raise PositioningError(
            f"reference_open must be strictly positive, got {reference!r}"
        )

    fee_rate_value = _nonnegative_number("fee_rate", fee_rate)
    slippage_rate_value = _finite_number("slippage_rate", slippage_rate)
    if not 0 <= slippage_rate_value < 1:
        raise PositioningError(
            "slippage_rate must satisfy 0 <= slippage_rate < 1, "
            f"got {slippage_rate_value!r}"
        )

    currently_cash = position <= tolerance_value
    if target is TargetPosition.CASH:
        if currently_cash:
            return None
        return MarketOrder(
            created_at=pending_target.decision_bar_timestamp,
            side=Side.SELL,
            quantity=position,
        )

    if not currently_cash:
        return None
    if cash == 0:
        return None

    slipped_unit_price = reference * (1 + slippage_rate_value)
    effective_unit_cost = slipped_unit_price * (1 + fee_rate_value)
    if not math.isfinite(effective_unit_cost) or effective_unit_cost <= 0:
        return None

    raw_quantity = cash / effective_unit_cost
    quantity = math.nextafter(raw_quantity, 0.0)
    if not math.isfinite(quantity) or quantity <= 0:
        return None

    return MarketOrder(
        created_at=pending_target.decision_bar_timestamp,
        side=Side.BUY,
        quantity=quantity,
    )
