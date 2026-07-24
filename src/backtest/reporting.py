"""Deterministic reporting views over an already completed backtest result.

The functions in this module do not rerun execution or mutate the supplied
``BacktestResult``.  They copy existing fill fields and calculate a deliberately
small set of descriptive, whole-period diagnostics.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from backtest.engine import (
    BacktestResult,
    PortfolioSnapshot,
    TargetWeightBacktestResult,
)
from backtest.execution import Fill
from backtest.orders import Side
from backtest.portfolio import PortfolioState

_TOLERANCE = 1e-12


class ReportingError(ValueError):
    """Raised when a result cannot be reported without ambiguity."""


class ReportingResult(Protocol):
    """Completed-result fields required by deterministic reporting."""

    initial_state: PortfolioState
    final_state: PortfolioState
    fills: tuple[Fill, ...]
    portfolio_history: tuple[PortfolioSnapshot, ...]


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Normalized reporting view copied from one existing execution fill."""

    decision_bar_timestamp: datetime
    execution_timestamp: datetime
    side: Side
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    fee: float
    cash_flow: float


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Descriptive whole-period metrics for one completed backtest."""

    initial_portfolio_value: float
    final_portfolio_value: float
    cumulative_return: float
    max_drawdown: float
    total_fees: float
    trade_count: int
    buy_count: int
    sell_count: int
    turnover: float
    average_exposure: float


def _require_result(result: object) -> ReportingResult:
    if not isinstance(result, (BacktestResult, TargetWeightBacktestResult)):
        raise ReportingError(
            "result must be a BacktestResult or TargetWeightBacktestResult, "
            f"got {result!r}"
        )
    return result


def _utc_timestamp(field: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ReportingError(f"{field} must be a datetime, got {value!r}")
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as error:
        raise ReportingError(f"{field} has an invalid UTC offset: {value!r}") from error
    if value.tzinfo is None or offset is None:
        raise ReportingError(f"{field} must include timezone information, got {value!r}")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as error:
        raise ReportingError(f"{field} cannot be normalized to UTC: {value!r}") from error


def _finite_number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingError(f"{field} must be numeric and not boolean, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ReportingError(f"{field} must be finite, got {value!r}")
    return number


def _nonnegative_number(field: str, value: object) -> float:
    number = _finite_number(field, value)
    if number < 0:
        raise ReportingError(f"{field} must be non-negative, got {number!r}")
    return number


def _positive_number(field: str, value: object) -> float:
    number = _finite_number(field, value)
    if number <= 0:
        raise ReportingError(f"{field} must be strictly positive, got {number!r}")
    return number


def _finite_sum(field: str, values: tuple[float, ...]) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ReportingError(f"{field} must remain finite") from error
    if not math.isfinite(total):
        raise ReportingError(f"{field} must remain finite, got {total!r}")
    return total


def _validated_fills(
    result: ReportingResult,
) -> tuple[tuple[Fill, datetime, datetime], ...]:
    if not isinstance(result.fills, tuple):
        raise ReportingError("result.fills must be a tuple")

    validated: list[tuple[Fill, datetime, datetime]] = []
    previous_execution: datetime | None = None
    for index, fill in enumerate(result.fills):
        prefix = f"result.fills[{index}]"
        if not isinstance(fill, Fill):
            raise ReportingError(f"{prefix} must be a Fill, got {fill!r}")
        if not isinstance(fill.side, Side):
            raise ReportingError(f"{prefix}.side must be a Side, got {fill.side!r}")

        decision_at = _utc_timestamp(f"{prefix}.order_created_at", fill.order_created_at)
        executed_at = _utc_timestamp(f"{prefix}.executed_at", fill.executed_at)
        if executed_at <= decision_at:
            raise ReportingError(f"{prefix}.executed_at must be strictly later than order_created_at")
        if previous_execution is not None and executed_at < previous_execution:
            raise ReportingError(f"{prefix}.executed_at is not in non-decreasing fill order")

        _positive_number(f"{prefix}.quantity", fill.quantity)
        _positive_number(f"{prefix}.reference_price", fill.reference_price)
        _positive_number(f"{prefix}.fill_price", fill.fill_price)
        _positive_number(f"{prefix}.notional", fill.notional)
        _nonnegative_number(f"{prefix}.fee", fill.fee)
        cash_flow = _finite_number(f"{prefix}.cash_flow", fill.cash_flow)
        if fill.side is Side.BUY and cash_flow >= 0:
            raise ReportingError(f"{prefix}.cash_flow for a buy must be negative")
        if fill.side is Side.SELL and cash_flow <= 0:
            raise ReportingError(f"{prefix}.cash_flow for a sell must be positive")

        validated.append((fill, decision_at, executed_at))
        previous_execution = executed_at
    return tuple(validated)


def _validated_history(
    result: ReportingResult,
) -> tuple[tuple[PortfolioSnapshot, datetime], ...]:
    if not isinstance(result.portfolio_history, tuple):
        raise ReportingError("result.portfolio_history must be a tuple")
    if not result.portfolio_history:
        raise ReportingError("result.portfolio_history must contain at least one snapshot")

    validated: list[tuple[PortfolioSnapshot, datetime]] = []
    previous_timestamp: datetime | None = None
    for index, snapshot in enumerate(result.portfolio_history):
        prefix = f"result.portfolio_history[{index}]"
        if not isinstance(snapshot, PortfolioSnapshot):
            raise ReportingError(f"{prefix} must be a PortfolioSnapshot, got {snapshot!r}")
        timestamp = _utc_timestamp(f"{prefix}.bar_timestamp", snapshot.bar_timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ReportingError(f"{prefix}.bar_timestamp must be strictly chronological")
        cash = _nonnegative_number(f"{prefix}.cash", snapshot.cash)
        position = _nonnegative_number(
            f"{prefix}.position_quantity", snapshot.position_quantity
        )
        _nonnegative_number(f"{prefix}.cumulative_fees", snapshot.cumulative_fees)
        close_price = _positive_number(f"{prefix}.close_price", snapshot.close_price)
        portfolio_value = _positive_number(
            f"{prefix}.portfolio_value", snapshot.portfolio_value
        )
        expected_portfolio_value = cash + position * close_price
        if not math.isfinite(expected_portfolio_value):
            raise ReportingError(
                f"{prefix} expected portfolio_value must be finite, "
                f"got {expected_portfolio_value!r}"
            )
        if not math.isclose(
            portfolio_value,
            expected_portfolio_value,
            rel_tol=_TOLERANCE,
            abs_tol=_TOLERANCE,
        ):
            raise ReportingError(
                f"{prefix}.portfolio_value actual {portfolio_value!r} does not "
                f"reconcile with expected {expected_portfolio_value!r}"
            )
        validated.append((snapshot, timestamp))
        previous_timestamp = timestamp
    return tuple(validated)


def _validated_state(field: str, value: object) -> tuple[float, float, float]:
    if not isinstance(value, PortfolioState):
        raise ReportingError(f"{field} must be a PortfolioState, got {value!r}")
    return (
        _nonnegative_number(f"{field}.cash", value.cash),
        _nonnegative_number(f"{field}.position_quantity", value.position_quantity),
        _nonnegative_number(f"{field}.cumulative_fees", value.cumulative_fees),
    )


def _reconcile_final_snapshot(
    snapshot: PortfolioSnapshot,
    *,
    final_cash: float,
    final_position: float,
    final_fees: float,
) -> None:
    for field, snapshot_value, state_value in (
        ("cash", snapshot.cash, final_cash),
        ("position_quantity", snapshot.position_quantity, final_position),
        ("cumulative_fees", snapshot.cumulative_fees, final_fees),
    ):
        if not math.isclose(
            snapshot_value,
            state_value,
            rel_tol=_TOLERANCE,
            abs_tol=_TOLERANCE,
        ):
            raise ReportingError(
                f"final snapshot {field} {snapshot_value!r} does not reconcile "
                f"with result.final_state.{field} {state_value!r}"
            )


def build_trade_log(result: ReportingResult) -> tuple[TradeRecord, ...]:
    """Return one immutable normalized record per fill, preserving fill order.

    Numeric fields are copied directly. Timestamps retain their instants and are
    normalized to UTC; no execution mathematics is repeated here.
    """

    actual_result = _require_result(result)
    return tuple(
        TradeRecord(
            decision_bar_timestamp=decision_at,
            execution_timestamp=executed_at,
            side=fill.side,
            quantity=fill.quantity,
            reference_price=fill.reference_price,
            fill_price=fill.fill_price,
            notional=fill.notional,
            fee=fill.fee,
            cash_flow=fill.cash_flow,
        )
        for fill, decision_at, executed_at in _validated_fills(actual_result)
    )


def summarize_backtest(result: ReportingResult) -> BacktestSummary:
    """Calculate deterministic descriptive metrics from an immutable result.

    Initial value marks the initial cash and position at the first snapshot's
    close. Final value is the last snapshot's stored portfolio value. Turnover
    is absolute gross fill notional divided by that initial marked value and is
    neither annualized nor an average-daily measure. Average exposure is the
    arithmetic mean of end-of-bar long capital exposure; it is not an intrabar
    or within-bar time-weighted measure.
    """

    actual_result = _require_result(result)
    fills = _validated_fills(actual_result)
    history = _validated_history(actual_result)
    initial_cash, initial_position, initial_fees = _validated_state(
        "result.initial_state", actual_result.initial_state
    )
    final_cash, final_position, final_fees = _validated_state(
        "result.final_state", actual_result.final_state
    )
    _reconcile_final_snapshot(
        history[-1][0],
        final_cash=final_cash,
        final_position=final_position,
        final_fees=final_fees,
    )

    first_snapshot = history[0][0]
    initial_value = initial_cash + initial_position * first_snapshot.close_price
    if not math.isfinite(initial_value) or initial_value <= 0:
        raise ReportingError(
            "initial marked portfolio value must be finite and strictly positive, "
            f"got {initial_value!r}"
        )
    final_value = float(history[-1][0].portfolio_value)
    cumulative_return = final_value / initial_value - 1.0
    if not math.isfinite(cumulative_return):
        raise ReportingError(f"cumulative_return must be finite, got {cumulative_return!r}")

    running_peak = 0.0
    maximum_drawdown = 0.0
    exposures: list[float] = []
    for snapshot, _ in history:
        value = float(snapshot.portfolio_value)
        running_peak = max(running_peak, value)
        drawdown = value / running_peak - 1.0
        maximum_drawdown = min(maximum_drawdown, drawdown)

        exposure = snapshot.position_quantity * snapshot.close_price / value
        if not math.isfinite(exposure):
            raise ReportingError(f"snapshot exposure must be finite, got {exposure!r}")
        if -_TOLERANCE <= exposure < 0:
            exposure = 0.0
        elif 1 < exposure <= 1 + _TOLERANCE:
            exposure = 1.0
        elif exposure < 0 or exposure > 1:
            raise ReportingError(f"snapshot exposure must lie within [0, 1], got {exposure!r}")
        exposures.append(exposure)

    run_fees = final_fees - initial_fees
    if -_TOLERANCE <= run_fees < 0:
        run_fees = 0.0
    elif run_fees < 0:
        raise ReportingError(f"run fees must be non-negative, got {run_fees!r}")
    fill_fees = _finite_sum("sum of fill fees", tuple(fill.fee for fill, _, _ in fills))
    if not math.isclose(fill_fees, run_fees, rel_tol=0.0, abs_tol=_TOLERANCE):
        raise ReportingError(
            f"fill fees {fill_fees!r} do not reconcile with cumulative fee change {run_fees!r}"
        )

    buy_count = sum(fill.side is Side.BUY for fill, _, _ in fills)
    sell_count = sum(fill.side is Side.SELL for fill, _, _ in fills)
    trade_count = len(fills)
    if buy_count + sell_count != trade_count:
        raise ReportingError("buy_count plus sell_count must equal trade_count")

    gross_notional = _finite_sum(
        "gross traded notional", tuple(fill.notional for fill, _, _ in fills)
    )
    turnover = gross_notional / initial_value
    average_exposure = math.fsum(exposures) / len(exposures)
    if not math.isfinite(turnover) or not math.isfinite(average_exposure):
        raise ReportingError("calculated turnover and average_exposure must be finite")

    return BacktestSummary(
        initial_portfolio_value=initial_value,
        final_portfolio_value=final_value,
        cumulative_return=cumulative_return,
        max_drawdown=maximum_drawdown,
        total_fees=run_fees,
        trade_count=trade_count,
        buy_count=buy_count,
        sell_count=sell_count,
        turnover=turnover,
        average_exposure=average_exposure,
    )
