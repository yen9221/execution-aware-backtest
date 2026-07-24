"""Immutable single-asset, long/cash portfolio accounting."""

import math
from dataclasses import dataclass

from backtest.execution import Fill
from backtest.orders import Side

_MAX_TOLERANCE = 1e-6


class PortfolioError(ValueError):
    """Raised when a fill cannot be validly applied to a portfolio state."""


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Cash, long asset quantity, and fees paid for one asset."""

    cash: float
    position_quantity: float = 0.0
    cumulative_fees: float = 0.0


def _finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioError(f"{field} must be numeric and not boolean, got {value!r}")

    number = float(value)
    if not math.isfinite(number):
        raise PortfolioError(f"{field} must be finite, got {value!r}")
    return number


def _nonnegative_number(field: str, value: object) -> float:
    number = _finite_number(field, value)
    if number < 0:
        raise PortfolioError(f"{field} must be non-negative, got {number!r}")
    return number


def _validated_tolerance(value: object) -> float:
    tolerance = _nonnegative_number("tolerance", value)
    if tolerance > _MAX_TOLERANCE:
        raise PortfolioError(
            f"tolerance must be no greater than {_MAX_TOLERANCE!r}, got {tolerance!r}"
        )
    return tolerance


def apply_fill(
    state: PortfolioState,
    fill: Fill,
    *,
    tolerance: float = 1e-12,
) -> PortfolioState:
    """Return a new state after applying one already-calculated fill.

    ``tolerance`` is an absolute allowance used only to normalize tiny negative
    cash or position residuals to zero. Deficits greater than the tolerance are
    rejected, and tolerance itself may not exceed ``1e-6``.
    """

    if not isinstance(state, PortfolioState):
        raise PortfolioError(f"state must be a PortfolioState, got {state!r}")
    if not isinstance(fill, Fill):
        raise PortfolioError(f"fill must be a Fill, got {fill!r}")

    cash = _nonnegative_number("state.cash", state.cash)
    position = _nonnegative_number(
        "state.position_quantity", state.position_quantity
    )
    cumulative_fees = _nonnegative_number(
        "state.cumulative_fees", state.cumulative_fees
    )
    tolerance_value = _validated_tolerance(tolerance)

    if not isinstance(fill.side, Side):
        raise PortfolioError(f"fill.side must be a Side, got {fill.side!r}")

    quantity = _finite_number("fill.quantity", fill.quantity)
    if quantity <= 0:
        raise PortfolioError(f"fill.quantity must be strictly positive, got {quantity!r}")

    fee = _nonnegative_number("fill.fee", fill.fee)
    cash_flow = _finite_number("fill.cash_flow", fill.cash_flow)
    if fill.side is Side.BUY and cash_flow >= 0:
        raise PortfolioError(
            f"buy fill.cash_flow must be negative, got {cash_flow!r}"
        )
    if fill.side is Side.SELL and cash_flow <= 0:
        raise PortfolioError(
            f"sell fill.cash_flow must be positive, got {cash_flow!r}"
        )

    new_cash = cash + cash_flow
    if fill.side is Side.BUY:
        required_cash = -cash_flow
        if required_cash > cash + tolerance_value:
            deficit = required_cash - cash
            raise PortfolioError(
                f"insufficient cash: required {required_cash!r}, available {cash!r}, "
                f"deficit {deficit!r} exceeds tolerance {tolerance_value!r}"
            )
        new_position = position + quantity
    else:
        if quantity > position + tolerance_value:
            deficit = quantity - position
            raise PortfolioError(
                f"insufficient position: sell quantity {quantity!r}, available {position!r}, "
                f"deficit {deficit!r} exceeds tolerance {tolerance_value!r}"
            )
        new_position = position - quantity

    new_cumulative_fees = cumulative_fees + fee
    for field, value in (
        ("new cash", new_cash),
        ("new position_quantity", new_position),
        ("new cumulative_fees", new_cumulative_fees),
    ):
        if not math.isfinite(value):
            raise PortfolioError(f"{field} must remain finite, got {value!r}")

    if -tolerance_value <= new_cash < 0:
        new_cash = 0.0
    if -tolerance_value <= new_position < 0:
        new_position = 0.0
    if new_cash < 0:
        raise PortfolioError(f"new cash must not be negative, got {new_cash!r}")
    if new_position < 0:
        raise PortfolioError(
            f"new position_quantity must not be negative, got {new_position!r}"
        )

    return PortfolioState(
        cash=new_cash,
        position_quantity=new_position,
        cumulative_fees=new_cumulative_fees,
    )
