"""Minimal chronological bar loop for precomputed target positions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backtest.events import Bar
from backtest.execution import Fill, execute_market_order
from backtest.portfolio import PortfolioState, apply_fill
from backtest.positioning import (
    PendingTarget,
    TargetPosition,
    create_pending_target,
    market_order_for_target_at_open,
)

_BAR_INTERVAL = timedelta(hours=1)
_MAX_TOLERANCE = 1e-6
_STRING_LIKE = (str, bytes, bytearray)


class EngineError(ValueError):
    """Raised when engine inputs or orchestration invariants are invalid."""


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """End-of-bar mark-to-market state after that bar's open processing."""

    bar_timestamp: datetime
    cash: float
    position_quantity: float
    cumulative_fees: float
    close_price: float
    portfolio_value: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable execution, state, and snapshot history from one run."""

    initial_state: PortfolioState
    final_state: PortfolioState
    fills: tuple[Fill, ...]
    portfolio_history: tuple[PortfolioSnapshot, ...]
    unexecuted_final_target: PendingTarget | None


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise EngineError(f"{field} must be a datetime, got {value!r}")

    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise EngineError(f"{field} has an invalid UTC offset: {value!r}") from error

    if value.tzinfo is None or offset is None:
        raise EngineError(f"{field} must include timezone information, got {value!r}")

    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise EngineError(f"{field} cannot be normalized to UTC: {value!r}") from error


def _finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineError(f"{field} must be numeric and not boolean, got {value!r}")

    number = float(value)
    if not math.isfinite(number):
        raise EngineError(f"{field} must be finite, got {value!r}")
    return number


def _nonnegative_number(field: str, value: object) -> float:
    number = _finite_number(field, value)
    if number < 0:
        raise EngineError(f"{field} must be non-negative, got {number!r}")
    return number


def _validate_bars(bars: object) -> tuple[tuple[Bar, datetime, float, float], ...]:
    if isinstance(bars, _STRING_LIKE) or not isinstance(bars, Sequence):
        raise EngineError("bars must be a non-string sequence of Bar objects")
    if len(bars) == 0:
        raise EngineError("bars must contain at least one Bar")

    validated: list[tuple[Bar, datetime, float, float]] = []
    previous_timestamp: datetime | None = None
    for index, bar in enumerate(bars):
        if not isinstance(bar, Bar):
            raise EngineError(f"bars[{index}] must be a Bar, got {bar!r}")

        timestamp = _utc_timestamp(f"bars[{index}].timestamp", bar.timestamp)
        if previous_timestamp is not None:
            if timestamp == previous_timestamp:
                raise EngineError(
                    f"bars[{index}].timestamp duplicates the preceding UTC instant "
                    f"{timestamp.isoformat()}"
                )
            if timestamp < previous_timestamp:
                raise EngineError(
                    f"bars[{index}].timestamp {timestamp.isoformat()} is not strictly "
                    f"later than {previous_timestamp.isoformat()}"
                )
            if timestamp - previous_timestamp != _BAR_INTERVAL:
                raise EngineError(
                    f"bars[{index}].timestamp {timestamp.isoformat()} is not exactly one "
                    f"hour after {previous_timestamp.isoformat()}"
                )

        open_price = _finite_number(f"bars[{index}].open", bar.open)
        if open_price <= 0:
            raise EngineError(
                f"bars[{index}].open must be strictly positive, got {open_price!r}"
            )
        close_price = _finite_number(f"bars[{index}].close", bar.close)
        if close_price <= 0:
            raise EngineError(
                f"bars[{index}].close must be strictly positive, got {close_price!r}"
            )

        validated.append((bar, timestamp, open_price, close_price))
        previous_timestamp = timestamp

    return tuple(validated)


def _validate_targets(
    targets: object,
    *,
    expected_length: int,
) -> tuple[TargetPosition, ...]:
    if isinstance(targets, _STRING_LIKE) or not isinstance(targets, Sequence):
        raise EngineError("targets must be a non-string sequence of TargetPosition values")
    if len(targets) != expected_length:
        raise EngineError(
            f"targets length {len(targets)} must equal bars length {expected_length}"
        )

    validated: list[TargetPosition] = []
    for index, target in enumerate(targets):
        if not isinstance(target, TargetPosition):
            raise EngineError(
                f"targets[{index}] must be a TargetPosition, got {target!r}"
            )
        validated.append(target)
    return tuple(validated)


def _validate_initial_state(state: object) -> PortfolioState:
    if not isinstance(state, PortfolioState):
        raise EngineError(f"initial_state must be a PortfolioState, got {state!r}")

    _nonnegative_number("initial_state.cash", state.cash)
    _nonnegative_number("initial_state.position_quantity", state.position_quantity)
    _nonnegative_number("initial_state.cumulative_fees", state.cumulative_fees)
    return state


def _validate_rates(
    *,
    fee_rate: object,
    slippage_rate: object,
    tolerance: object,
) -> tuple[float, float, float]:
    fee = _nonnegative_number("fee_rate", fee_rate)
    slippage = _finite_number("slippage_rate", slippage_rate)
    if not 0 <= slippage < 1:
        raise EngineError(
            f"slippage_rate must satisfy 0 <= slippage_rate < 1, got {slippage!r}"
        )
    tolerance_value = _nonnegative_number("tolerance", tolerance)
    if tolerance_value > _MAX_TOLERANCE:
        raise EngineError(
            f"tolerance must be no greater than {_MAX_TOLERANCE!r}, "
            f"got {tolerance_value!r}"
        )
    return fee, slippage, tolerance_value


def run_backtest(
    *,
    bars: Sequence[Bar],
    targets: Sequence[TargetPosition],
    initial_state: PortfolioState,
    fee_rate: float,
    slippage_rate: float,
    tolerance: float = 1e-12,
) -> BacktestResult:
    """Run a deterministic close-decision to next-open execution loop.

    Engine input errors raise ``EngineError``. Domain errors from positioning,
    execution, or portfolio accounting propagate unchanged so their original
    validation context remains available.
    """

    validated_bars = _validate_bars(bars)
    validated_targets = _validate_targets(
        targets,
        expected_length=len(validated_bars),
    )
    state = _validate_initial_state(initial_state)
    fee, slippage, tolerance_value = _validate_rates(
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        tolerance=tolerance,
    )

    pending_target: PendingTarget | None = None
    fills: list[Fill] = []
    history: list[PortfolioSnapshot] = []

    for index, (bar, timestamp, open_price, close_price) in enumerate(validated_bars):
        if pending_target is not None:
            order = market_order_for_target_at_open(
                state=state,
                pending_target=pending_target,
                execution_bar_timestamp=timestamp,
                reference_open=open_price,
                fee_rate=fee,
                slippage_rate=slippage,
                tolerance=tolerance_value,
            )
            pending_target = None
            if order is not None:
                fill = execute_market_order(
                    order,
                    executed_at=timestamp,
                    reference_price=open_price,
                    fee_rate=fee,
                    slippage_rate=slippage,
                )
                state = apply_fill(state, fill, tolerance=tolerance_value)
                fills.append(fill)

        portfolio_value = state.cash + state.position_quantity * close_price
        if not math.isfinite(portfolio_value):
            raise EngineError(
                f"bars[{index}] portfolio_value must be finite, got "
                f"{portfolio_value!r} at {timestamp.isoformat()}"
            )
        history.append(
            PortfolioSnapshot(
                bar_timestamp=timestamp,
                cash=state.cash,
                position_quantity=state.position_quantity,
                cumulative_fees=state.cumulative_fees,
                close_price=close_price,
                portfolio_value=portfolio_value,
            )
        )

        pending_target = create_pending_target(
            decision_bar_timestamp=timestamp,
            target=validated_targets[index],
        )

    return BacktestResult(
        initial_state=initial_state,
        final_state=state,
        fills=tuple(fills),
        portfolio_history=tuple(history),
        unexecuted_final_target=pending_target,
    )
