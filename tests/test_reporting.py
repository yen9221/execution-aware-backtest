import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.reporting as reporting_module
from backtest.engine import (
    BacktestResult,
    PortfolioSnapshot,
    TargetWeightBacktestResult,
    run_backtest,
    run_target_weight_backtest,
)
from backtest.events import Bar
from backtest.execution import Fill
from backtest.orders import Side
from backtest.portfolio import PortfolioState
from backtest.positioning import TargetPosition, TargetWeight
from backtest.reporting import (
    BacktestSummary,
    ReportingError,
    TradeRecord,
    build_trade_log,
    summarize_backtest,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def fill(
    side: Side = Side.BUY,
    *,
    index: int = 1,
    quantity: float = 2.0,
    price: float = 100.0,
    fee: float = 0.2,
) -> Fill:
    notional = quantity * price
    cash_flow = -(notional + fee) if side is Side.BUY else notional - fee
    return Fill(
        order_created_at=START + timedelta(hours=index - 1),
        executed_at=START + timedelta(hours=index),
        side=side,
        quantity=quantity,
        reference_price=price,
        fill_price=price,
        notional=notional,
        fee=fee,
        cash_flow=cash_flow,
        fee_rate=0.001,
        slippage_rate=0.0,
    )


def snapshot(
    index: int,
    *,
    cash: float = 1_000.0,
    position: float = 0.0,
    close: float = 100.0,
    value: float | None = None,
    fees: float = 0.0,
) -> PortfolioSnapshot:
    portfolio_value = cash + position * close if value is None else value
    return PortfolioSnapshot(
        bar_timestamp=START + timedelta(hours=index),
        cash=cash,
        position_quantity=position,
        cumulative_fees=fees,
        close_price=close,
        portfolio_value=portfolio_value,
    )


def result(
    *,
    fills: object = (),
    history: object | None = None,
    initial: PortfolioState | None = None,
    final: PortfolioState | None = None,
) -> BacktestResult:
    initial_state = initial or PortfolioState(cash=1_000.0)
    fill_tuple = fills if isinstance(fills, tuple) else fills
    history_value = history
    if final is None:
        if (
            isinstance(history_value, tuple)
            and history_value
            and isinstance(history_value[-1], PortfolioSnapshot)
        ):
            last_snapshot = history_value[-1]
            final = PortfolioState(
                cash=last_snapshot.cash,
                position_quantity=last_snapshot.position_quantity,
                cumulative_fees=last_snapshot.cumulative_fees,
            )
        else:
            fee_total = (
                sum(item.fee for item in fill_tuple)
                if isinstance(fill_tuple, tuple)
                else 0.0
            )
            final = replace(
                initial_state,
                cumulative_fees=initial_state.cumulative_fees + fee_total,
            )
    if history_value is None:
        history_value = (
            snapshot(
                0,
                cash=final.cash,
                position=final.position_quantity,
                fees=final.cumulative_fees,
            ),
        )
    return BacktestResult(
        initial_state=initial_state,
        final_state=final,
        fills=fill_tuple,  # type: ignore[arg-type]
        portfolio_history=history_value,  # type: ignore[arg-type]
        unexecuted_final_target=None,
    )


def bars(prices: list[tuple[float, float]]) -> list[Bar]:
    return [
        Bar(
            timestamp=START + timedelta(hours=index),
            open=open_price,
            high=max(open_price, close_price),
            low=min(open_price, close_price),
            close=close_price,
            volume=1.0,
        )
        for index, (open_price, close_price) in enumerate(prices)
    ]


def test_reporting_records_are_immutable() -> None:
    trade = build_trade_log(result(fills=(fill(fee=0.0),)))[0]
    summary = summarize_backtest(result())
    with pytest.raises(FrozenInstanceError):
        trade.fee = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        summary.trade_count = 1  # type: ignore[misc]


def test_reporting_schemas_exclude_out_of_scope_fields() -> None:
    names = {item.name for schema in (TradeRecord, BacktestSummary) for item in fields(schema)}
    assert names.isdisjoint(
        {"model", "signal", "threshold", "strategy", "sharpe", "benchmark", "alpha"}
    )


@pytest.mark.parametrize("invalid", [None, object(), {}, "result", 1, True])
def test_invalid_result_type_is_rejected(invalid: object) -> None:
    with pytest.raises(ReportingError, match="BacktestResult"):
        build_trade_log(invalid)  # type: ignore[arg-type]
    with pytest.raises(ReportingError, match="BacktestResult"):
        summarize_backtest(invalid)  # type: ignore[arg-type]


def test_no_fills_returns_empty_immutable_tuple() -> None:
    trade_log = build_trade_log(result())
    assert trade_log == ()
    assert isinstance(trade_log, tuple)


def test_one_fill_maps_every_trade_record_field_without_recalculation() -> None:
    source = fill(quantity=3.0, price=123.0, fee=0.75)
    record = build_trade_log(result(fills=(source,)))[0]
    assert record == TradeRecord(
        decision_bar_timestamp=source.order_created_at,
        execution_timestamp=source.executed_at,
        side=source.side,
        quantity=source.quantity,
        reference_price=source.reference_price,
        fill_price=source.fill_price,
        notional=source.notional,
        fee=source.fee,
        cash_flow=source.cash_flow,
    )


def test_trade_timestamps_are_normalized_to_utc() -> None:
    source = replace(
        fill(fee=0.0),
        order_created_at=datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        executed_at=datetime(2023, 12, 31, 20, tzinfo=timezone(timedelta(hours=-5))),
    )
    record = build_trade_log(result(fills=(source,)))[0]
    assert record.decision_bar_timestamp == START
    assert record.execution_timestamp == START + timedelta(hours=1)
    assert record.decision_bar_timestamp.tzinfo is timezone.utc
    assert record.execution_timestamp.tzinfo is timezone.utc


def test_multiple_fills_preserve_order() -> None:
    first = fill(Side.BUY, index=1, fee=0.0)
    second = fill(Side.SELL, index=2, fee=0.0)
    assert [item.side for item in build_trade_log(result(fills=(first, second)))] == [
        Side.BUY,
        Side.SELL,
    ]


def test_trade_log_does_not_mutate_result_or_fills() -> None:
    source = result(fills=(fill(fee=0.0),))
    original = replace(source)
    original_fill = replace(source.fills[0])
    assert build_trade_log(source) == build_trade_log(source)
    assert source == original
    assert source.fills[0] == original_fill


def test_invalid_fill_and_fill_container_are_rejected() -> None:
    with pytest.raises(ReportingError, match="fills must be a tuple"):
        build_trade_log(result(fills=[]))
    with pytest.raises(ReportingError, match="must be a Fill"):
        build_trade_log(
            result(fills=(object(),), final=PortfolioState(cash=1_000.0))
        )


def test_invalid_fill_side_is_rejected() -> None:
    with pytest.raises(ReportingError, match="side must be a Side"):
        build_trade_log(result(fills=(replace(fill(), side="buy"),)))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("quantity", 0.0, "strictly positive"),
        ("quantity", -1.0, "strictly positive"),
        ("quantity", float("nan"), "finite"),
        ("quantity", True, "not boolean"),
        ("reference_price", 0.0, "strictly positive"),
        ("fill_price", float("inf"), "finite"),
        ("notional", -1.0, "strictly positive"),
        ("fee", -1.0, "non-negative"),
        ("fee", float("nan"), "finite"),
        ("cash_flow", float("inf"), "finite"),
    ],
)
def test_invalid_fill_numeric_fields_are_rejected(
    field: str, invalid: object, message: str
) -> None:
    with pytest.raises(ReportingError, match=message):
        build_trade_log(result(fills=(replace(fill(), **{field: invalid}),)))


@pytest.mark.parametrize(
    "invalid_fill",
    [replace(fill(Side.BUY), cash_flow=0.0), replace(fill(Side.SELL), cash_flow=-1.0)],
)
def test_invalid_fill_cash_flow_sign_is_rejected(invalid_fill: Fill) -> None:
    with pytest.raises(ReportingError, match="cash_flow"):
        build_trade_log(result(fills=(invalid_fill,)))


@pytest.mark.parametrize("field", ["order_created_at", "executed_at"])
def test_naive_fill_timestamp_is_rejected(field: str) -> None:
    with pytest.raises(ReportingError, match="timezone"):
        build_trade_log(
            result(fills=(replace(fill(), **{field: datetime(2024, 1, 1)}),))
        )


def test_execution_must_follow_decision_and_fill_order_must_not_descend() -> None:
    with pytest.raises(ReportingError, match="strictly later"):
        build_trade_log(result(fills=(replace(fill(), executed_at=START),)))
    first = fill(index=2, fee=0.0)
    second = fill(index=1, fee=0.0)
    with pytest.raises(ReportingError, match="non-decreasing"):
        build_trade_log(result(fills=(first, second)))


def test_empty_and_invalid_history_containers_are_rejected() -> None:
    with pytest.raises(ReportingError, match="at least one"):
        summarize_backtest(result(history=()))
    with pytest.raises(ReportingError, match="history must be a tuple"):
        summarize_backtest(result(history=[]))


def test_invalid_snapshot_type_is_rejected() -> None:
    with pytest.raises(ReportingError, match="PortfolioSnapshot"):
        summarize_backtest(result(history=(object(),)))


def test_naive_and_non_chronological_snapshot_timestamps_are_rejected() -> None:
    naive = replace(snapshot(0), bar_timestamp=datetime(2024, 1, 1))
    with pytest.raises(ReportingError, match="timezone"):
        summarize_backtest(result(history=(naive,)))
    with pytest.raises(ReportingError, match="strictly chronological"):
        summarize_backtest(result(history=(snapshot(1), snapshot(0))))
    with pytest.raises(ReportingError, match="strictly chronological"):
        summarize_backtest(result(history=(snapshot(0), snapshot(0))))


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("cash", -1.0, "non-negative"),
        ("cash", float("nan"), "finite"),
        ("cash", True, "not boolean"),
        ("position_quantity", -1.0, "non-negative"),
        ("position_quantity", float("inf"), "finite"),
        ("cumulative_fees", -1.0, "non-negative"),
        ("close_price", 0.0, "strictly positive"),
        ("close_price", float("nan"), "finite"),
        ("portfolio_value", 0.0, "strictly positive"),
        ("portfolio_value", -1.0, "strictly positive"),
        ("portfolio_value", float("inf"), "finite"),
    ],
)
def test_invalid_snapshot_fields_are_rejected(
    field: str, invalid: object, message: str
) -> None:
    with pytest.raises(ReportingError, match=message):
        summarize_backtest(result(history=(replace(snapshot(0), **{field: invalid}),)))


def test_materially_inconsistent_snapshot_portfolio_value_is_rejected() -> None:
    inconsistent = snapshot(
        0,
        cash=500.0,
        position=5.0,
        close=100.0,
        value=900.0,
    )
    with pytest.raises(
        ReportingError,
        match=r"portfolio_history\[0\].*actual 900\.0.*expected 1000\.0",
    ):
        summarize_backtest(result(history=(inconsistent,)))


def test_tiny_snapshot_portfolio_value_residual_is_accepted() -> None:
    tiny_residual = snapshot(
        0,
        cash=500.0,
        position=5.0,
        close=100.0,
        value=1_000.0 + 5e-10,
    )
    summary = summarize_backtest(result(history=(tiny_residual,)))
    assert summary.final_portfolio_value == pytest.approx(1_000.0)


def test_non_finite_expected_snapshot_portfolio_value_is_rejected() -> None:
    overflow = snapshot(
        0,
        cash=1e308,
        position=1e308,
        close=1e308,
        value=1e308,
    )
    with pytest.raises(ReportingError, match=r"portfolio_history\[0\].*expected.*finite"):
        summarize_backtest(result(history=(overflow,)))


@pytest.mark.parametrize(
    ("field", "final_state"),
    [
        ("cash", PortfolioState(cash=999.0)),
        ("position_quantity", PortfolioState(cash=1_000.0, position_quantity=1.0)),
        ("cumulative_fees", PortfolioState(cash=1_000.0, cumulative_fees=1.0)),
    ],
)
def test_material_final_snapshot_state_mismatch_is_rejected(
    field: str,
    final_state: PortfolioState,
) -> None:
    with pytest.raises(ReportingError, match=rf"final snapshot {field}.*final_state\.{field}"):
        summarize_backtest(result(final=final_state, history=(snapshot(0),)))


def test_tiny_final_snapshot_state_residuals_are_accepted() -> None:
    final_state = PortfolioState(
        cash=1_000.0 + 5e-13,
        position_quantity=5e-13,
        cumulative_fees=5e-13,
    )
    summary = summarize_backtest(result(final=final_state, history=(snapshot(0),)))
    assert summary.final_portfolio_value == 1_000.0


@pytest.mark.parametrize("cash", [0.0, -1.0])
def test_non_positive_initial_marked_value_is_rejected(cash: float) -> None:
    initial = PortfolioState(cash=cash)
    with pytest.raises(ReportingError, match="initial_state.cash|initial marked"):
        summarize_backtest(
            result(
                initial=initial,
                final=PortfolioState(cash=1_000.0),
                history=(snapshot(0),),
            )
        )


@pytest.mark.parametrize(
    ("values", "expected_return"),
    [([1_000.0], 0.0), ([1_000.0, 1_100.0], 0.1), ([1_000.0, 900.0], -0.1)],
)
def test_cumulative_return_uses_first_mark_and_last_stored_value(
    values: list[float], expected_return: float
) -> None:
    history = tuple(snapshot(index, cash=value) for index, value in enumerate(values))
    summary = summarize_backtest(result(history=history))
    assert summary.initial_portfolio_value == 1_000.0
    assert summary.final_portfolio_value == values[-1]
    assert summary.cumulative_return == pytest.approx(expected_return)


def test_initial_position_is_marked_at_first_snapshot_close() -> None:
    initial = PortfolioState(cash=200.0, position_quantity=2.0)
    history = (snapshot(0, cash=200.0, position=2.0, close=150.0),)
    summary = summarize_backtest(result(initial=initial, final=initial, history=history))
    assert summary.initial_portfolio_value == 500.0
    assert summary.final_portfolio_value == 500.0
    assert summary.cumulative_return == 0.0


def test_return_is_whole_period_not_annualized() -> None:
    history = (snapshot(0, cash=1_000.0), snapshot(1, cash=1_210.0))
    assert summarize_backtest(result(history=history)).cumulative_return == pytest.approx(0.21)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([100.0], 0.0),
        ([100.0, 100.0, 100.0], 0.0),
        ([100.0, 110.0, 120.0], 0.0),
        ([100.0, 80.0], -0.2),
        ([100.0, 80.0, 100.0], -0.2),
        ([100.0, 120.0, 90.0, 110.0, 88.0], 88.0 / 120.0 - 1.0),
    ],
)
def test_signed_maximum_drawdown(values: list[float], expected: float) -> None:
    history = tuple(snapshot(index, cash=value) for index, value in enumerate(values))
    drawdown = summarize_backtest(result(history=history)).max_drawdown
    assert drawdown == pytest.approx(expected)
    assert drawdown <= 0


def test_no_fills_gives_zero_fees_counts_and_turnover() -> None:
    summary = summarize_backtest(result())
    assert (summary.total_fees, summary.trade_count, summary.buy_count, summary.sell_count) == (
        0.0,
        0,
        0,
        0,
    )
    assert summary.turnover == 0.0


def test_fees_counts_and_gross_turnover_include_each_fill_once() -> None:
    buy = fill(Side.BUY, index=1, quantity=2.0, price=100.0, fee=0.2)
    sell = fill(Side.SELL, index=2, quantity=2.0, price=110.0, fee=0.22)
    initial = PortfolioState(cash=1_000.0, cumulative_fees=5.0)
    final = PortfolioState(cash=1_019.58, cumulative_fees=5.42)
    summary = summarize_backtest(result(fills=(buy, sell), initial=initial, final=final))
    assert summary.total_fees == pytest.approx(0.42)
    assert summary.trade_count == 2
    assert summary.buy_count == 1
    assert summary.sell_count == 1
    assert summary.turnover == pytest.approx((200.0 + 220.0) / 1_000.0)


def test_initial_marked_value_is_turnover_denominator() -> None:
    source_fill = fill(fee=0.0, quantity=1.0, price=100.0)
    initial = PortfolioState(cash=500.0, position_quantity=5.0)
    history = (snapshot(0, cash=500.0, position=5.0, close=100.0),)
    summary = summarize_backtest(
        result(fills=(source_fill,), initial=initial, final=initial, history=history)
    )
    assert summary.turnover == pytest.approx(100.0 / 1_000.0)


def test_material_fee_mismatch_and_negative_run_fees_are_rejected() -> None:
    with pytest.raises(ReportingError, match="do not reconcile"):
        summarize_backtest(
            result(
                fills=(fill(fee=1.0),),
                final=PortfolioState(cash=1_000.0, cumulative_fees=2.0),
            )
        )
    with pytest.raises(ReportingError, match="run fees must be non-negative"):
        summarize_backtest(
            result(
                initial=PortfolioState(cash=1_000.0, cumulative_fees=1.0),
                final=PortfolioState(cash=1_000.0, cumulative_fees=0.9),
            )
        )


def test_multi_fill_machine_precision_fee_residual_is_accepted() -> None:
    fills = (
        fill(index=1, fee=0.1),
        fill(index=2, fee=0.2),
        fill(index=3, fee=0.3),
    )
    cumulative_fee_change = math.fsum(item.fee for item in fills) + 3.6e-12
    summary = summarize_backtest(
        result(
            fills=fills,
            final=PortfolioState(
                cash=1_000.0,
                cumulative_fees=cumulative_fee_change,
            ),
        )
    )
    assert summary.total_fees == cumulative_fee_change
    assert abs(
        math.fsum(item.fee for item in fills) - summary.total_fees
    ) == pytest.approx(3.6e-12)


def test_meaningful_fee_reconciliation_mismatch_is_rejected() -> None:
    fills = (fill(index=1, fee=0.25), fill(index=2, fee=0.25))
    with pytest.raises(ReportingError, match="do not reconcile"):
        summarize_backtest(
            result(
                fills=fills,
                final=PortfolioState(cash=1_000.0, cumulative_fees=0.500001),
            )
        )


def test_exact_multi_fill_fee_reconciliation_still_passes() -> None:
    fills = (fill(index=1, fee=0.25), fill(index=2, fee=0.25))
    summary = summarize_backtest(
        result(
            fills=fills,
            final=PortfolioState(cash=1_000.0, cumulative_fees=0.5),
        )
    )
    assert summary.total_fees == 0.5


def test_non_finite_fee_remains_rejected_by_summary_validation() -> None:
    with pytest.raises(ReportingError, match=r"result\.fills\[0\]\.fee must be finite"):
        summarize_backtest(
            result(
                fills=(replace(fill(), fee=math.nan),),
                final=PortfolioState(cash=1_000.0),
            )
        )


def test_overflowing_fill_fee_sum_is_reported_as_reporting_error() -> None:
    first = replace(fill(index=1), fee=1e308)
    second = replace(fill(index=2), fee=1e308)
    with pytest.raises(ReportingError, match="sum of fill fees must remain finite"):
        summarize_backtest(
            result(
                fills=(first, second),
                final=PortfolioState(cash=1_000.0, cumulative_fees=1e308),
            )
        )


def test_overflowing_gross_notional_is_reported_as_reporting_error() -> None:
    first = replace(fill(index=1, fee=0.0), notional=1e308)
    second = replace(fill(index=2, fee=0.0), notional=1e308)
    with pytest.raises(ReportingError, match="gross traded notional must remain finite"):
        summarize_backtest(result(fills=(first, second)))


def test_tiny_negative_run_fee_residual_normalizes_to_zero() -> None:
    summary = summarize_backtest(
        result(
            initial=PortfolioState(cash=1_000.0, cumulative_fees=1.0),
            final=PortfolioState(cash=1_000.0, cumulative_fees=1.0 - 5e-13),
        )
    )
    assert summary.total_fees == 0.0


def test_all_cash_and_fully_long_exposure() -> None:
    cash_summary = summarize_backtest(result(history=(snapshot(0), snapshot(1))))
    long_history = (
        snapshot(0, cash=0.0, position=10.0, close=100.0),
        snapshot(1, cash=0.0, position=5.0, close=200.0),
    )
    long_summary = summarize_backtest(result(history=long_history))
    assert cash_summary.average_exposure == 0.0
    assert long_summary.average_exposure == pytest.approx(1.0)


def test_mixed_end_of_bar_exposure_uses_close_and_arithmetic_mean() -> None:
    history = (
        snapshot(0, cash=1_000.0, position=0.0, close=50.0),
        snapshot(1, cash=500.0, position=5.0, close=100.0),
        snapshot(2, cash=0.0, position=4.0, close=250.0),
    )
    summary = summarize_backtest(result(history=history))
    assert summary.average_exposure == pytest.approx((0.0 + 0.5 + 1.0) / 3.0)


def test_material_exposure_above_one_is_rejected_by_snapshot_identity() -> None:
    invalid = snapshot(0, cash=0.0, position=11.0, close=100.0, value=1_000.0)
    with pytest.raises(ReportingError, match="portfolio_value actual.*expected"):
        summarize_backtest(result(history=(invalid,)))


def test_tiny_exposure_boundary_residual_is_normalized() -> None:
    tiny_above = snapshot(
        0,
        cash=0.0,
        position=1.0 + 5e-13,
        close=1_000.0,
        value=1_000.0,
    )
    assert summarize_backtest(result(history=(tiny_above,))).average_exposure == 1.0


def test_zero_cost_engine_integration_is_hand_checkable_and_deterministic() -> None:
    engine_result = run_backtest(
        bars=bars([(100.0, 100.0)] * 3),
        targets=[TargetPosition.LONG, TargetPosition.CASH, TargetPosition.CASH],
        initial_state=PortfolioState(cash=1_000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    original = replace(engine_result)
    trade_log = build_trade_log(engine_result)
    summary = summarize_backtest(engine_result)

    assert len(trade_log) == len(engine_result.fills) == 2
    assert [item.side for item in trade_log] == [Side.BUY, Side.SELL]
    assert summary.initial_portfolio_value == 1_000.0
    assert summary.final_portfolio_value == pytest.approx(1_000.0)
    assert summary.cumulative_return == pytest.approx(0.0)
    assert summary.max_drawdown == pytest.approx(0.0)
    assert summary.total_fees == 0.0
    assert summary.turnover == pytest.approx(2.0)
    assert summary.average_exposure == pytest.approx(1.0 / 3.0)
    assert engine_result == original
    assert build_trade_log(engine_result) == trade_log
    assert summarize_backtest(engine_result) == summary


def test_falling_price_engine_scenario_has_negative_return_and_drawdown() -> None:
    engine_result = run_backtest(
        bars=bars([(100.0, 100.0), (100.0, 90.0), (90.0, 80.0)]),
        targets=[TargetPosition.LONG, TargetPosition.LONG, TargetPosition.LONG],
        initial_state=PortfolioState(cash=1_000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    summary = summarize_backtest(engine_result)
    assert summary.final_portfolio_value == pytest.approx(800.0)
    assert summary.cumulative_return == pytest.approx(-0.2)
    assert summary.max_drawdown == pytest.approx(-0.2)


def fractional_result_zero_cost() -> TargetWeightBacktestResult:
    source_bars = bars(
        [(100.0, 101.0), (102.0, 103.0), (104.0, 105.0), (106.0, 107.0), (108.0, 109.0)]
    )
    return run_target_weight_backtest(
        bars=source_bars,
        targets=[TargetWeight(value) for value in (0.0, 0.5, 0.8, 0.3, 0.0)],
        initial_state=PortfolioState(cash=1_000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )


def fractional_result_cost_aware() -> TargetWeightBacktestResult:
    source_bars = bars([(100.0, 101.0)] * 6)
    return run_target_weight_backtest(
        bars=source_bars,
        targets=[
            TargetWeight(value)
            for value in (0.4, 0.405, 0.42, 0.7, 0.25, 0.0)
        ],
        initial_state=PortfolioState(cash=600.0, position_quantity=4.0),
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=21.0,
    )


def independently_calculated_summary(result: TargetWeightBacktestResult) -> dict[str, float | int]:
    first_close = result.portfolio_history[0].close_price
    initial_value = (
        result.initial_state.cash
        + result.initial_state.position_quantity * first_close
    )
    final_value = result.portfolio_history[-1].portfolio_value
    peak = 0.0
    maximum_drawdown = 0.0
    exposures: list[float] = []
    for item in result.portfolio_history:
        peak = max(peak, item.portfolio_value)
        maximum_drawdown = min(
            maximum_drawdown, item.portfolio_value / peak - 1.0
        )
        exposures.append(
            item.position_quantity * item.close_price / item.portfolio_value
        )
    return {
        "initial": initial_value,
        "final": final_value,
        "return": final_value / initial_value - 1.0,
        "drawdown": maximum_drawdown,
        "fees": sum(item.fee for item in result.fills),
        "trades": len(result.fills),
        "buys": sum(item.side is Side.BUY for item in result.fills),
        "sells": sum(item.side is Side.SELL for item in result.fills),
        "turnover": sum(item.notional for item in result.fills) / initial_value,
        "exposure": math.fsum(exposures) / len(exposures),
    }


def test_reporting_public_functions_accept_both_result_types() -> None:
    binary = run_backtest(
        bars=bars([(100.0, 100.0)]),
        targets=[TargetPosition.CASH],
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    fractional = fractional_result_zero_cost()
    assert isinstance(build_trade_log(binary), tuple)
    assert isinstance(build_trade_log(fractional), tuple)
    assert type(summarize_backtest(binary)) is BacktestSummary
    assert type(summarize_backtest(fractional)) is BacktestSummary


def test_fractional_trade_log_matches_actual_fills_exactly() -> None:
    engine_result = fractional_result_zero_cost()
    original_result = replace(engine_result)
    original_fills = tuple(replace(item) for item in engine_result.fills)
    trade_log = build_trade_log(engine_result)
    assert len(trade_log) == len(engine_result.fills) == 3
    assert [record.side for record in trade_log] == [
        Side.BUY,
        Side.BUY,
        Side.SELL,
    ]
    for record, source in zip(trade_log, engine_result.fills, strict=True):
        assert record == TradeRecord(
            decision_bar_timestamp=source.order_created_at,
            execution_timestamp=source.executed_at,
            side=source.side,
            quantity=source.quantity,
            reference_price=source.reference_price,
            fill_price=source.fill_price,
            notional=source.notional,
            fee=source.fee,
            cash_flow=source.cash_flow,
        )
    assert engine_result == original_result
    assert engine_result.fills == original_fills


def test_fractional_zero_cost_summary_matches_independent_calculation() -> None:
    engine_result = fractional_result_zero_cost()
    expected = independently_calculated_summary(engine_result)
    summary = summarize_backtest(engine_result)
    assert summary.initial_portfolio_value == pytest.approx(expected["initial"])
    assert summary.final_portfolio_value == pytest.approx(expected["final"])
    assert summary.cumulative_return == pytest.approx(expected["return"])
    assert summary.max_drawdown == pytest.approx(expected["drawdown"])
    assert summary.total_fees == pytest.approx(expected["fees"])
    assert summary.trade_count == expected["trades"]
    assert summary.buy_count == expected["buys"]
    assert summary.sell_count == expected["sells"]
    assert summary.turnover == pytest.approx(expected["turnover"])
    assert summary.average_exposure == pytest.approx(expected["exposure"])


def test_fractional_cost_aware_summary_uses_actual_fills_and_snapshots() -> None:
    engine_result = fractional_result_cost_aware()
    expected = independently_calculated_summary(engine_result)
    trade_log = build_trade_log(engine_result)
    summary = summarize_backtest(engine_result)
    assert len(engine_result.fills) == len(trade_log) == summary.trade_count == 2
    assert [record.side for record in trade_log] == [Side.BUY, Side.SELL]
    assert summary.total_fees == pytest.approx(
        sum(item.fee for item in engine_result.fills)
    )
    assert summary.total_fees == pytest.approx(expected["fees"])
    assert summary.turnover == pytest.approx(expected["turnover"])
    assert summary.final_portfolio_value == pytest.approx(expected["final"])
    assert summary.cumulative_return == pytest.approx(expected["return"])
    assert summary.max_drawdown == pytest.approx(expected["drawdown"])
    assert summary.average_exposure == pytest.approx(expected["exposure"])


def test_fractional_exposure_uses_realized_holdings_not_target_weight() -> None:
    engine_result = fractional_result_cost_aware()
    snapshot_after_buy = engine_result.portfolio_history[4]
    realized = (
        snapshot_after_buy.position_quantity * snapshot_after_buy.close_price
        / snapshot_after_buy.portfolio_value
    )
    intended = 0.7
    assert realized != pytest.approx(intended)
    all_exposures = [
        item.position_quantity * item.close_price / item.portfolio_value
        for item in engine_result.portfolio_history
    ]
    assert summarize_backtest(engine_result).average_exposure == pytest.approx(
        math.fsum(all_exposures) / len(all_exposures)
    )


def test_suppressed_and_final_targets_create_no_reporting_records() -> None:
    engine_result = fractional_result_cost_aware()
    assert len(engine_result.fills) == 2
    assert len(build_trade_log(engine_result)) == 2
    assert engine_result.unexecuted_final_target is not None
    assert all(
        item.order_created_at
        != engine_result.unexecuted_final_target.decision_bar_timestamp
        for item in engine_result.fills
    )
    summary = summarize_backtest(engine_result)
    assert summary.trade_count == 2
    assert summary.total_fees == pytest.approx(
        engine_result.fills[0].fee + engine_result.fills[1].fee
    )


def test_unexecuted_fractional_target_contents_do_not_affect_reporting() -> None:
    engine_result = fractional_result_zero_cost()
    assert engine_result.unexecuted_final_target is not None
    changed_pending = replace(
        engine_result.unexecuted_final_target,
        target=TargetWeight(1.0),
    )
    changed_result = replace(
        engine_result,
        unexecuted_final_target=changed_pending,
    )
    assert build_trade_log(changed_result) == build_trade_log(engine_result)
    assert summarize_backtest(changed_result) == summarize_backtest(engine_result)


def test_fractional_initial_fees_are_excluded_from_run_fees() -> None:
    source_bars = bars([(100.0, 100.0), (100.0, 100.0)])
    engine_result = run_target_weight_backtest(
        bars=source_bars,
        targets=[TargetWeight(0.5), TargetWeight(0.0)],
        initial_state=PortfolioState(cash=1000.0, cumulative_fees=10.0),
        fee_rate=0.01,
        slippage_rate=0.0,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    summary = summarize_backtest(engine_result)
    assert summary.total_fees == pytest.approx(engine_result.fills[0].fee)
    assert summary.total_fees == pytest.approx(
        engine_result.final_state.cumulative_fees
        - engine_result.initial_state.cumulative_fees
    )


def test_reporting_uses_one_shared_typed_implementation_without_target_routing() -> None:
    source = inspect.getsource(reporting_module)
    assert "class ReportingResult(Protocol)" in source
    assert "TargetWeightBacktestResult" in source
    assert "Any" not in source
    assert "cast(" not in source
    assert source.count("def build_trade_log(") == 1
    assert source.count("def summarize_backtest(") == 1
    for forbidden in (
        "unexecuted_final_target.target",
        "TargetPosition",
        "from backtest.positioning",
        "execute_market_order",
        "apply_fill",
        "run_backtest",
        "run_target_weight_backtest",
        "sharpe",
        "benchmark",
    ):
        assert forbidden not in source
