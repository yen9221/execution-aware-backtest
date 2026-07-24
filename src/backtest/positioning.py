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
class TargetWeight:
    """Immutable intended long allocation for one unlevered asset."""

    weight: float

    def __post_init__(self) -> None:
        value = self.weight
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PositioningError(
                f"weight must be numeric and not boolean, got {value!r}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise PositioningError(f"weight must be finite, got {value!r}")
        if not 0.0 <= number <= 1.0:
            raise PositioningError(
                f"weight must lie within the inclusive range [0.0, 1.0], got {number!r}"
            )
        object.__setattr__(self, "weight", number)


CASH_TARGET = TargetWeight(0.0)
LONG_TARGET = TargetWeight(1.0)


@dataclass(frozen=True, slots=True)
class PendingTarget:
    """A price-free target selected from one completed decision bar."""

    decision_bar_timestamp: datetime
    target: TargetPosition


@dataclass(frozen=True, slots=True)
class PendingTargetWeight:
    """A price-free continuous target selected from one completed bar."""

    decision_bar_timestamp: datetime
    target: TargetWeight


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


def _validated_target_weight(field: str, value: object) -> TargetWeight:
    if not isinstance(value, TargetWeight):
        raise PositioningError(f"{field} must be a TargetWeight, got {value!r}")
    # Revalidate the contained value in case an instance was constructed by
    # bypassing the frozen dataclass constructor.
    return TargetWeight(value.weight)


def _finite_calculation(field: str, value: float) -> float:
    if not math.isfinite(value):
        raise PositioningError(f"calculated {field} must be finite, got {value!r}")
    return value


def target_position_to_weight(target: TargetPosition) -> TargetWeight:
    """Convert one supported binary position to its canonical endpoint weight."""

    target_value = _validated_target("target", target)
    return CASH_TARGET if target_value is TargetPosition.CASH else LONG_TARGET


def target_weight_to_position(target: TargetWeight) -> TargetPosition:
    """Convert an exact endpoint weight for the binary execution path.

    Fractional weights are valid target representations and can be sized
    through ``market_order_for_target_weight_at_open``. They cannot be
    converted to the binary ``TargetPosition`` path without information loss,
    and therefore are rejected rather than silently coerced.
    """

    if not isinstance(target, TargetWeight):
        raise PositioningError(f"target must be a TargetWeight, got {target!r}")
    if target.weight == 0.0:
        return TargetPosition.CASH
    if target.weight == 1.0:
        return TargetPosition.LONG
    raise PositioningError(
        f"fractional target weight {target.weight!r} is not supported by the "
        "current binary execution path"
    )


def create_pending_target(
    *,
    decision_bar_timestamp: datetime,
    target: TargetPosition,
) -> PendingTarget:
    """Create a UTC-normalized, price-free intent from a completed bar."""

    timestamp = _utc_timestamp("decision_bar_timestamp", decision_bar_timestamp)
    target_value = _validated_target("target", target)
    return PendingTarget(decision_bar_timestamp=timestamp, target=target_value)


def create_pending_target_weight(
    *,
    decision_bar_timestamp: datetime,
    target: TargetWeight,
) -> PendingTargetWeight:
    """Create a UTC-normalized, price-free continuous target intent."""

    timestamp = _utc_timestamp("decision_bar_timestamp", decision_bar_timestamp)
    target_value = _validated_target_weight("target", target)
    return PendingTargetWeight(
        decision_bar_timestamp=timestamp,
        target=target_value,
    )


def market_order_for_target_weight_at_open(
    *,
    state: PortfolioState,
    pending_target: PendingTargetWeight,
    execution_bar_timestamp: datetime,
    reference_open: float,
    fee_rate: float,
    slippage_rate: float,
) -> MarketOrder | None:
    """Size a cost-aware delta order from an execution-open portfolio mark.

    The target is applied to pre-trade portfolio value at ``reference_open``.
    This function creates no fill and does not mutate or update the portfolio.
    Adverse slippage and proportional fees limit buy affordability but do not
    redefine the reference-open target exposure. Tolerances and minimum
    notionals are intentionally absent. A zero-value portfolio returns no order
    because it has no capital to allocate.
    """

    if not isinstance(state, PortfolioState):
        raise PositioningError(f"state must be a PortfolioState, got {state!r}")
    if not isinstance(pending_target, PendingTargetWeight):
        raise PositioningError(
            "pending_target must be a PendingTargetWeight, "
            f"got {pending_target!r}"
        )

    cash = _nonnegative_number("state.cash", state.cash)
    position = _nonnegative_number(
        "state.position_quantity", state.position_quantity
    )
    _nonnegative_number("state.cumulative_fees", state.cumulative_fees)

    decision_timestamp = _utc_timestamp(
        "pending_target.decision_bar_timestamp",
        pending_target.decision_bar_timestamp,
    )
    target = _validated_target_weight("pending_target.target", pending_target.target)
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

    current_asset_value = _finite_calculation(
        "current_asset_value", position * reference
    )
    pre_trade_portfolio_value = _finite_calculation(
        "pre_trade_portfolio_value", cash + current_asset_value
    )
    if pre_trade_portfolio_value < 0:
        raise PositioningError(
            "calculated pre_trade_portfolio_value must be non-negative, "
            f"got {pre_trade_portfolio_value!r}"
        )
    if pre_trade_portfolio_value == 0:
        return None

    desired_asset_value = _finite_calculation(
        "desired_asset_value", target.weight * pre_trade_portfolio_value
    )
    delta_asset_value = _finite_calculation(
        "delta_asset_value", desired_asset_value - current_asset_value
    )
    if delta_asset_value == 0:
        return None

    if delta_asset_value > 0:
        raw_quantity = _finite_calculation(
            "raw buy quantity", delta_asset_value / reference
        )
        expected_fill_price = reference * (1 + slippage_rate_value)
        effective_unit_cost = _finite_calculation(
            "effective buy unit cost",
            expected_fill_price * (1 + fee_rate_value),
        )
        if effective_unit_cost <= 0:
            raise PositioningError(
                "calculated effective buy unit cost must be strictly positive, "
                f"got {effective_unit_cost!r}"
            )
        maximum_quantity = _finite_calculation(
            "maximum affordable quantity", cash / effective_unit_cost
        )
        quantity = min(raw_quantity, maximum_quantity)

        expected_notional = _finite_calculation(
            "expected buy notional", quantity * expected_fill_price
        )
        expected_fee = _finite_calculation(
            "expected buy fee", expected_notional * fee_rate_value
        )
        expected_cash_outflow = _finite_calculation(
            "expected buy cash outflow", expected_notional + expected_fee
        )
        if expected_cash_outflow > cash:
            quantity = math.nextafter(quantity, 0.0)
            expected_notional = _finite_calculation(
                "adjusted expected buy notional", quantity * expected_fill_price
            )
            expected_fee = _finite_calculation(
                "adjusted expected buy fee", expected_notional * fee_rate_value
            )
            expected_cash_outflow = _finite_calculation(
                "adjusted expected buy cash outflow",
                expected_notional + expected_fee,
            )
        side = Side.BUY
    else:
        raw_quantity = _finite_calculation(
            "raw sell quantity", -delta_asset_value / reference
        )
        quantity = min(raw_quantity, position)
        side = Side.SELL

    if not math.isfinite(quantity) or quantity <= 0:
        raise PositioningError(
            "calculated order quantity must be finite and strictly positive, "
            f"got {quantity!r}"
        )
    if side is Side.BUY and expected_cash_outflow > cash:
        raise PositioningError(
            "calculated buy quantity exceeds cost-aware affordability after "
            "the conservative float adjustment"
        )
    if side is Side.SELL and quantity > position:
        raise PositioningError(
            "calculated sell quantity exceeds the existing position"
        )

    return MarketOrder(
        created_at=decision_timestamp,
        side=side,
        quantity=quantity,
    )


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
