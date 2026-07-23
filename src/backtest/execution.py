"""Deterministic market-order fill mathematics."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from backtest.orders import MarketOrder, Side


class ExecutionError(ValueError):
    """Raised when an order or execution input cannot produce a valid fill."""


@dataclass(frozen=True, slots=True)
class Fill:
    """An immutable, auditable market-order execution result.

    ``reference_price`` is the supplied next-bar open before slippage, while
    ``fill_price`` includes adverse directional slippage. ``notional`` is
    quantity times fill price. ``fee`` is charged once on that notional.
    ``cash_flow`` includes the fee and is negative for buys and positive for
    sells. Both rates are retained separately for auditability.
    """

    order_created_at: datetime
    executed_at: datetime
    side: Side
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    fee: float
    cash_flow: float
    fee_rate: float
    slippage_rate: float


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ExecutionError(f"{field} must be a datetime, got {value!r}")

    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise ExecutionError(f"{field} has an invalid UTC offset: {value!r}") from error

    if value.tzinfo is None or offset is None:
        raise ExecutionError(
            f"{field} must include timezone information, got {value!r}"
        )

    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ExecutionError(f"{field} cannot be normalized to UTC: {value!r}") from error


def _finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionError(f"{field} must be numeric and not boolean, got {value!r}")

    number = float(value)
    if not math.isfinite(number):
        raise ExecutionError(f"{field} must be finite, got {value!r}")
    return number


def _validate_calculated_fill(
    *, side: Side, fill_price: float, notional: float, fee: float, cash_flow: float
) -> None:
    if not math.isfinite(fill_price) or fill_price <= 0:
        raise ExecutionError(
            f"calculated fill_price must be finite and positive, got {fill_price!r}"
        )
    if not math.isfinite(notional) or notional <= 0:
        raise ExecutionError(
            f"calculated notional must be finite and positive, got {notional!r}"
        )
    if not math.isfinite(fee) or fee < 0:
        raise ExecutionError(
            f"calculated fee must be finite and non-negative, got {fee!r}"
        )
    if not math.isfinite(cash_flow):
        raise ExecutionError(f"calculated cash_flow must be finite, got {cash_flow!r}")
    if side is Side.BUY and cash_flow >= 0:
        raise ExecutionError(
            f"calculated cash_flow for a buy must be negative, got {cash_flow!r}"
        )
    if side is Side.SELL and cash_flow <= 0:
        raise ExecutionError(
            f"calculated cash_flow for a sell must be positive, got {cash_flow!r}"
        )


def execute_market_order(
    order: MarketOrder,
    *,
    executed_at: datetime,
    reference_price: float,
    fee_rate: float,
    slippage_rate: float,
) -> Fill:
    """Calculate one deterministic fill using a supplied next-bar open price.

    This function validates execution inputs and performs no portfolio checks.
    For both sides, slippage must satisfy ``0 <= slippage_rate < 1``.
    """

    if not isinstance(order, MarketOrder):
        raise ExecutionError(f"order must be a MarketOrder, got {order!r}")
    if not isinstance(order.side, Side):
        raise ExecutionError(f"order.side must be a Side, got {order.side!r}")

    order_created_at = _utc_timestamp("order.created_at", order.created_at)
    execution_timestamp = _utc_timestamp("executed_at", executed_at)
    if execution_timestamp <= order_created_at:
        relation = "equal to" if execution_timestamp == order_created_at else "before"
        raise ExecutionError(
            f"executed_at must be strictly later than order.created_at; execution instant "
            f"{execution_timestamp.isoformat()} is {relation} creation instant "
            f"{order_created_at.isoformat()}"
        )

    quantity = _finite_number("order.quantity", order.quantity)
    if quantity <= 0:
        raise ExecutionError(f"order.quantity must be strictly positive, got {quantity!r}")

    reference = _finite_number("reference_price", reference_price)
    if reference <= 0:
        raise ExecutionError(f"reference_price must be strictly positive, got {reference!r}")

    fee_rate_value = _finite_number("fee_rate", fee_rate)
    if fee_rate_value < 0:
        raise ExecutionError(f"fee_rate must be non-negative, got {fee_rate_value!r}")

    slippage_rate_value = _finite_number("slippage_rate", slippage_rate)
    if not 0 <= slippage_rate_value < 1:
        raise ExecutionError(
            f"slippage_rate must satisfy 0 <= slippage_rate < 1, got {slippage_rate_value!r}"
        )

    if order.side is Side.BUY:
        fill_price = reference * (1 + slippage_rate_value)
    else:
        fill_price = reference * (1 - slippage_rate_value)

    notional = quantity * fill_price
    fee = notional * fee_rate_value
    if order.side is Side.BUY:
        cash_flow = -(notional + fee)
    else:
        cash_flow = notional - fee

    _validate_calculated_fill(
        side=order.side,
        fill_price=fill_price,
        notional=notional,
        fee=fee,
        cash_flow=cash_flow,
    )

    return Fill(
        order_created_at=order_created_at,
        executed_at=execution_timestamp,
        side=order.side,
        quantity=quantity,
        reference_price=reference,
        fill_price=fill_price,
        notional=notional,
        fee=fee,
        cash_flow=cash_flow,
        fee_rate=fee_rate_value,
        slippage_rate=slippage_rate_value,
    )
