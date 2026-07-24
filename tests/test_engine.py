import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.engine as engine_module
from backtest.engine import BacktestResult, EngineError, PortfolioSnapshot, run_backtest
from backtest.events import Bar
from backtest.execution import Fill
from backtest.orders import MarketOrder, Side
from backtest.portfolio import PortfolioState
from backtest.positioning import PositioningError, TargetPosition

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005


def bar(
    index: int = 0,
    *,
    timestamp: datetime | None = None,
    open_price: float = 100.0,
    close_price: float = 101.0,
) -> Bar:
    return Bar(
        timestamp=timestamp or START + timedelta(hours=index),
        open=open_price,
        high=max(open_price, close_price) + 1.0,
        low=min(open_price, close_price) - 1.0,
        close=close_price,
        volume=10.0,
    )


def bar_sequence(count: int) -> list[Bar]:
    return [
        bar(index, open_price=100.0 + index, close_price=100.5 + index)
        for index in range(count)
    ]


def run(
    bars: object,
    targets: object,
    *,
    initial_state: object = PortfolioState(cash=10_000.0),
    fee_rate: object = FEE_RATE,
    slippage_rate: object = SLIPPAGE_RATE,
    tolerance: object = 1e-12,
) -> BacktestResult:
    return run_backtest(
        bars=bars,  # type: ignore[arg-type]
        targets=targets,  # type: ignore[arg-type]
        initial_state=initial_state,  # type: ignore[arg-type]
        fee_rate=fee_rate,  # type: ignore[arg-type]
        slippage_rate=slippage_rate,  # type: ignore[arg-type]
        tolerance=tolerance,  # type: ignore[arg-type]
    )


def test_portfolio_snapshot_is_immutable() -> None:
    snapshot = run([bar()], [TargetPosition.CASH]).portfolio_history[0]

    with pytest.raises(FrozenInstanceError):
        snapshot.cash = 0.0  # type: ignore[misc]


def test_backtest_result_is_immutable() -> None:
    result = run([bar()], [TargetPosition.CASH])

    with pytest.raises(FrozenInstanceError):
        result.final_state = PortfolioState(cash=0.0)  # type: ignore[misc]


def test_empty_bars_are_rejected() -> None:
    with pytest.raises(EngineError, match="at least one"):
        run([], [])


@pytest.mark.parametrize("invalid", [None, 1, object(), {"bar": bar()}])
def test_non_sequence_bars_are_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="bars must be a non-string sequence"):
        run(invalid, [])


@pytest.mark.parametrize("invalid", ["bar", b"bar", bytearray(b"bar")])
def test_string_like_bars_are_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="bars must be a non-string sequence"):
        run(invalid, [])


def test_invalid_bar_element_is_rejected() -> None:
    with pytest.raises(EngineError, match=r"bars\[0\] must be a Bar"):
        run([object()], [TargetPosition.CASH])


def test_naive_bar_timestamp_is_rejected() -> None:
    with pytest.raises(EngineError, match=r"bars\[0\].timestamp.*timezone"):
        run(
            [bar(timestamp=datetime(2024, 1, 1))],
            [TargetPosition.CASH],
        )


def test_duplicate_bar_timestamp_is_rejected() -> None:
    with pytest.raises(EngineError, match="duplicates"):
        run(
            [bar(timestamp=START), bar(timestamp=START)],
            [TargetPosition.CASH, TargetPosition.CASH],
        )


def test_descending_bar_timestamps_are_rejected() -> None:
    with pytest.raises(EngineError, match="not strictly later"):
        run(
            [bar(timestamp=START), bar(timestamp=START - timedelta(hours=1))],
            [TargetPosition.CASH, TargetPosition.CASH],
        )


def test_missing_hour_is_rejected() -> None:
    with pytest.raises(EngineError, match="not exactly one hour"):
        run(
            [bar(timestamp=START), bar(timestamp=START + timedelta(hours=2))],
            [TargetPosition.CASH, TargetPosition.CASH],
        )


def test_irregular_spacing_is_rejected() -> None:
    with pytest.raises(EngineError, match="not exactly one hour"):
        run(
            [bar(timestamp=START), bar(timestamp=START + timedelta(minutes=30))],
            [TargetPosition.CASH, TargetPosition.CASH],
        )


def test_equivalent_timezone_timestamps_are_compared_by_utc_instant() -> None:
    first = datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    second = datetime(2023, 12, 31, 20, tzinfo=timezone(timedelta(hours=-5)))

    result = run(
        [bar(timestamp=first), bar(timestamp=second)],
        [TargetPosition.CASH, TargetPosition.CASH],
    )

    assert [snapshot.bar_timestamp for snapshot in result.portfolio_history] == [
        START,
        START + timedelta(hours=1),
    ]
    assert all(
        snapshot.bar_timestamp.tzinfo is timezone.utc
        for snapshot in result.portfolio_history
    )


def test_equivalent_timezone_duplicate_is_rejected() -> None:
    same_instant = datetime(
        2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))
    )

    with pytest.raises(EngineError, match="duplicates"):
        run(
            [bar(timestamp=START), bar(timestamp=same_instant)],
            [TargetPosition.CASH, TargetPosition.CASH],
        )


@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "100"],
)
def test_invalid_bar_open_is_rejected(invalid: object) -> None:
    invalid_bar = replace(bar(), open=invalid)  # type: ignore[arg-type]

    with pytest.raises(EngineError, match=r"bars\[0\].open"):
        run([invalid_bar], [TargetPosition.CASH])


@pytest.mark.parametrize(
    "invalid",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "101"],
)
def test_invalid_bar_close_is_rejected(invalid: object) -> None:
    invalid_bar = replace(bar(), close=invalid)  # type: ignore[arg-type]

    with pytest.raises(EngineError, match=r"bars\[0\].close"):
        run([invalid_bar], [TargetPosition.CASH])


@pytest.mark.parametrize("invalid", [None, 1, object(), {"target": "cash"}])
def test_non_sequence_targets_are_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="targets must be a non-string sequence"):
        run([bar()], invalid)


@pytest.mark.parametrize("invalid", ["cash", b"cash", bytearray(b"cash")])
def test_string_like_targets_are_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="targets must be a non-string sequence"):
        run([bar()], invalid)


@pytest.mark.parametrize("targets", [[], [TargetPosition.CASH, TargetPosition.LONG]])
def test_target_length_mismatch_is_rejected(targets: list[TargetPosition]) -> None:
    with pytest.raises(EngineError, match="targets length.*bars length"):
        run([bar()], targets)


@pytest.mark.parametrize("target", [0, 1, True, False, "cash", "long"])
def test_non_enum_target_is_rejected(target: object) -> None:
    with pytest.raises(EngineError, match=r"targets\[0\] must be a TargetPosition"):
        run([bar()], [target])


def test_invalid_initial_state_type_is_rejected() -> None:
    with pytest.raises(EngineError, match="initial_state must be a PortfolioState"):
        run([bar()], [TargetPosition.CASH], initial_state={"cash": 10_000.0})


@pytest.mark.parametrize(
    ("field", "state"),
    [
        ("cash", PortfolioState(cash=-1.0)),
        ("position_quantity", PortfolioState(cash=1.0, position_quantity=-1.0)),
        ("cumulative_fees", PortfolioState(cash=1.0, cumulative_fees=-1.0)),
    ],
)
def test_negative_initial_state_field_is_rejected(
    field: str,
    state: PortfolioState,
) -> None:
    with pytest.raises(EngineError, match=f"initial_state.{field}.*non-negative"):
        run([bar()], [TargetPosition.CASH], initial_state=state)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_non_finite_initial_state_field_is_rejected(
    field: str,
    value: float,
) -> None:
    state = replace(PortfolioState(cash=1.0), **{field: value})

    with pytest.raises(EngineError, match=f"initial_state.{field}.*finite"):
        run([bar()], [TargetPosition.CASH], initial_state=state)


@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_boolean_initial_state_field_is_rejected(field: str) -> None:
    state = replace(PortfolioState(cash=1.0), **{field: True})

    with pytest.raises(EngineError, match=f"initial_state.{field}.*not boolean"):
        run([bar()], [TargetPosition.CASH], initial_state=state)


@pytest.mark.parametrize(
    "invalid",
    [-0.001, float("nan"), float("inf"), float("-inf"), True, "0.001"],
)
def test_invalid_fee_rate_is_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="fee_rate"):
        run([bar()], [TargetPosition.CASH], fee_rate=invalid)


@pytest.mark.parametrize(
    "invalid",
    [-0.001, 1.0, 2.0, float("nan"), float("inf"), float("-inf"), True, "0.001"],
)
def test_invalid_slippage_rate_is_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="slippage_rate"):
        run([bar()], [TargetPosition.CASH], slippage_rate=invalid)


@pytest.mark.parametrize(
    "invalid",
    [-1.0, 1.1e-6, float("nan"), float("inf"), float("-inf"), True, "1e-12"],
)
def test_invalid_tolerance_is_rejected(invalid: object) -> None:
    with pytest.raises(EngineError, match="tolerance"):
        run([bar()], [TargetPosition.CASH], tolerance=invalid)


@pytest.mark.parametrize("target", list(TargetPosition))
def test_single_final_bar_target_never_executes(target: TargetPosition) -> None:
    result = run([bar()], [target])

    assert result.fills == ()
    assert result.final_state == PortfolioState(cash=10_000.0)
    assert len(result.portfolio_history) == 1
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is target


def test_single_final_cash_target_does_not_liquidate_initial_long() -> None:
    initial = PortfolioState(cash=1_000.0, position_quantity=2.0)

    result = run([bar()], [TargetPosition.CASH], initial_state=initial)

    assert result.fills == ()
    assert result.final_state == initial
    assert result.final_state.position_quantity == 2.0


def test_no_pending_target_exists_at_first_bar_open(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[datetime] = []
    real_converter = engine_module.market_order_for_target_at_open

    def spy_converter(**kwargs: object) -> MarketOrder | None:
        calls.append(kwargs["execution_bar_timestamp"])  # type: ignore[arg-type]
        return real_converter(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(engine_module, "market_order_for_target_at_open", spy_converter)

    run(bar_sequence(2), [TargetPosition.CASH, TargetPosition.CASH])

    assert calls == [START + timedelta(hours=1)]


def test_target_from_bar_t_never_executes_on_bar_t() -> None:
    result = run(bar_sequence(2), [TargetPosition.LONG, TargetPosition.LONG])

    assert len(result.fills) == 1
    assert result.fills[0].executed_at == START + timedelta(hours=1)
    assert result.fills[0].order_created_at == START
    assert result.portfolio_history[0].position_quantity == 0.0


def test_target_executes_at_next_bar_open_not_prior_close_or_next_close() -> None:
    bars = [
        bar(0, open_price=90.0, close_price=150.0),
        bar(1, open_price=100.0, close_price=250.0),
    ]

    result = run(bars, [TargetPosition.LONG, TargetPosition.LONG])

    fill = result.fills[0]
    assert fill.reference_price == 100.0
    assert fill.reference_price != 150.0
    assert fill.reference_price != 250.0
    assert fill.executed_at == bars[1].timestamp
    assert fill.order_created_at == bars[0].timestamp


def test_repeated_cash_targets_produce_no_fills() -> None:
    result = run(bar_sequence(4), [TargetPosition.CASH] * 4)

    assert result.fills == ()


def test_repeated_long_target_produces_only_initial_entry() -> None:
    result = run(bar_sequence(4), [TargetPosition.LONG] * 4)

    assert len(result.fills) == 1
    assert result.fills[0].side is Side.BUY
    assert result.fills[0].executed_at == START + timedelta(hours=1)


def test_cash_to_long_creates_exactly_one_buy() -> None:
    result = run(
        bar_sequence(3),
        [TargetPosition.CASH, TargetPosition.LONG, TargetPosition.LONG],
    )

    assert [fill.side for fill in result.fills] == [Side.BUY]
    assert result.fills[0].executed_at == START + timedelta(hours=2)


def test_long_to_cash_creates_exactly_one_full_liquidation_sell() -> None:
    initial = PortfolioState(cash=1_000.0, position_quantity=2.5)
    result = run(
        bar_sequence(2),
        [TargetPosition.CASH, TargetPosition.CASH],
        initial_state=initial,
    )

    assert len(result.fills) == 1
    assert result.fills[0].side is Side.SELL
    assert result.fills[0].quantity == initial.position_quantity
    assert result.fills[0].executed_at == START + timedelta(hours=1)
    assert result.fills[0].order_created_at == START
    assert result.final_state.position_quantity == 0.0


def test_final_target_that_would_reverse_position_remains_unexecuted() -> None:
    bars = bar_sequence(2)

    result = run(bars, [TargetPosition.LONG, TargetPosition.CASH])

    assert len(result.fills) == 1
    assert result.fills[0].side is Side.BUY
    assert result.final_state.position_quantity > 0
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is TargetPosition.CASH
    assert result.unexecuted_final_target.decision_bar_timestamp == bars[1].timestamp


def test_no_trade_conversion_does_not_create_fake_fill() -> None:
    result = run(
        bar_sequence(2),
        [TargetPosition.CASH, TargetPosition.LONG],
    )

    assert result.fills == ()
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is TargetPosition.LONG


def test_one_snapshot_per_bar_with_chronological_timestamps() -> None:
    bars = bar_sequence(5)

    result = run(bars, [TargetPosition.CASH] * 5)

    assert len(result.portfolio_history) == len(bars)
    assert [snapshot.bar_timestamp for snapshot in result.portfolio_history] == [
        item.timestamp for item in bars
    ]


def test_snapshot_reflects_entry_at_same_bar_open_and_marks_at_close() -> None:
    bars = [
        bar(0, open_price=100.0, close_price=110.0),
        bar(1, open_price=120.0, close_price=130.0),
    ]

    result = run(bars, [TargetPosition.LONG, TargetPosition.LONG])
    before_entry, entry_bar = result.portfolio_history

    assert before_entry.cash == 10_000.0
    assert before_entry.position_quantity == 0.0
    assert before_entry.portfolio_value == 10_000.0
    assert entry_bar.position_quantity > 0
    assert entry_bar.close_price == 130.0
    assert entry_bar.portfolio_value == pytest.approx(
        entry_bar.cash + entry_bar.position_quantity * 130.0
    )


def test_exit_bar_snapshot_reflects_cash_after_open_liquidation() -> None:
    result = run(
        bar_sequence(3),
        [TargetPosition.LONG, TargetPosition.CASH, TargetPosition.CASH],
    )

    assert result.portfolio_history[1].position_quantity > 0
    assert result.portfolio_history[2].position_quantity == 0.0
    assert result.portfolio_history[2].cash == result.final_state.cash
    assert result.portfolio_history[2].portfolio_value == result.final_state.cash


def test_initial_state_and_inputs_remain_unchanged() -> None:
    bars = bar_sequence(3)
    targets = [TargetPosition.LONG, TargetPosition.CASH, TargetPosition.LONG]
    original_bars = list(bars)
    original_targets = list(targets)
    initial = PortfolioState(cash=10_000.0)
    original_initial = replace(initial)

    result = run(bars, targets, initial_state=initial)

    assert bars == original_bars
    assert targets == original_targets
    assert initial == original_initial
    assert result.initial_state is initial
    assert result.initial_state == original_initial


def test_result_uses_immutable_tuples_and_final_applied_state() -> None:
    result = run(bar_sequence(2), [TargetPosition.LONG, TargetPosition.LONG])

    assert isinstance(result.fills, tuple)
    assert isinstance(result.portfolio_history, tuple)
    assert result.final_state.cash == result.portfolio_history[-1].cash
    assert (
        result.final_state.position_quantity
        == result.portfolio_history[-1].position_quantity
    )
    assert result.final_state.cumulative_fees == result.portfolio_history[-1].cumulative_fees


def test_repeated_identical_runs_are_deterministic() -> None:
    bars = bar_sequence(5)
    targets = [
        TargetPosition.CASH,
        TargetPosition.LONG,
        TargetPosition.LONG,
        TargetPosition.CASH,
        TargetPosition.LONG,
    ]

    first = run(bars, targets)
    second = run(bars, targets)

    assert first == second
    assert first is not second


def test_zero_fee_slippage_round_trip_is_hand_checkable() -> None:
    bars = [
        bar(0, open_price=100.0, close_price=100.0),
        bar(1, open_price=100.0, close_price=100.0),
        bar(2, open_price=100.0, close_price=100.0),
    ]

    result = run(
        bars,
        [TargetPosition.LONG, TargetPosition.CASH, TargetPosition.CASH],
        initial_state=PortfolioState(cash=1_000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]
    assert result.fills[0].fill_price == 100.0
    assert result.fills[1].fill_price == 100.0
    assert result.final_state.cash == pytest.approx(1_000.0)
    assert result.final_state.position_quantity == 0.0
    assert result.final_state.cumulative_fees == 0.0


def test_fees_and_slippage_flow_through_existing_execution_layer() -> None:
    result = run(
        bar_sequence(3),
        [TargetPosition.LONG, TargetPosition.CASH, TargetPosition.CASH],
        fee_rate=FEE_RATE,
        slippage_rate=0.01,
    )

    buy, sell = result.fills
    assert buy.fill_price > buy.reference_price
    assert sell.fill_price < sell.reference_price
    assert result.final_state.cumulative_fees == pytest.approx(buy.fee + sell.fee)


def test_engine_delegates_positioning_execution_and_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"positioning": 0, "execution": 0, "accounting": 0}
    real_positioning = engine_module.market_order_for_target_at_open
    real_execution = engine_module.execute_market_order
    real_accounting = engine_module.apply_fill

    def positioning_spy(**kwargs: object) -> MarketOrder | None:
        calls["positioning"] += 1
        return real_positioning(**kwargs)  # type: ignore[arg-type]

    def execution_spy(order: MarketOrder, **kwargs: object) -> Fill:
        calls["execution"] += 1
        return real_execution(order, **kwargs)  # type: ignore[arg-type]

    def accounting_spy(
        state: PortfolioState,
        fill: Fill,
        **kwargs: object,
    ) -> PortfolioState:
        calls["accounting"] += 1
        return real_accounting(state, fill, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(engine_module, "market_order_for_target_at_open", positioning_spy)
    monkeypatch.setattr(engine_module, "execute_market_order", execution_spy)
    monkeypatch.setattr(engine_module, "apply_fill", accounting_spy)

    run(bar_sequence(2), [TargetPosition.LONG, TargetPosition.LONG])

    assert calls == {"positioning": 1, "execution": 1, "accounting": 1}


def test_domain_errors_propagate_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    error = PositioningError("positioning failed")

    def fail_positioning(**kwargs: object) -> None:
        raise error

    monkeypatch.setattr(engine_module, "market_order_for_target_at_open", fail_positioning)

    with pytest.raises(PositioningError, match="positioning failed") as caught:
        run(bar_sequence(2), [TargetPosition.CASH, TargetPosition.CASH])

    assert caught.value is error


def test_portfolio_value_overflow_is_rejected() -> None:
    initial = PortfolioState(cash=0.0, position_quantity=1e308)

    with pytest.raises(EngineError, match="portfolio_value must be finite"):
        run(
            [bar(close_price=1e308)],
            [TargetPosition.LONG],
            initial_state=initial,
        )


def test_full_five_bar_scenario_matches_expected_event_order() -> None:
    bars = [
        bar(0, open_price=100.0, close_price=101.0),
        bar(1, open_price=102.0, close_price=103.0),
        bar(2, open_price=104.0, close_price=105.0),
        bar(3, open_price=106.0, close_price=107.0),
        bar(4, open_price=108.0, close_price=109.0),
    ]
    targets = [
        TargetPosition.CASH,
        TargetPosition.LONG,
        TargetPosition.LONG,
        TargetPosition.CASH,
        TargetPosition.LONG,
    ]

    result = run(bars, targets)

    assert len(result.fills) == 2
    buy, sell = result.fills
    assert [buy.side, sell.side] == [Side.BUY, Side.SELL]
    assert [buy.executed_at, sell.executed_at] == [
        bars[2].timestamp,
        bars[4].timestamp,
    ]
    assert [buy.order_created_at, sell.order_created_at] == [
        bars[1].timestamp,
        bars[3].timestamp,
    ]
    assert buy.reference_price == bars[2].open
    assert sell.reference_price == bars[4].open
    assert sell.quantity == buy.quantity
    assert result.final_state.position_quantity == 0.0
    assert result.portfolio_history[0].position_quantity == 0.0
    assert result.portfolio_history[1].position_quantity == 0.0
    assert result.portfolio_history[2].position_quantity > 0.0
    assert result.portfolio_history[3].position_quantity > 0.0
    assert result.portfolio_history[4].position_quantity == 0.0
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is TargetPosition.LONG
    assert result.unexecuted_final_target.decision_bar_timestamp == bars[4].timestamp


def test_engine_schemas_contain_no_model_or_performance_fields() -> None:
    schema_fields = {
        field.name for schema in (PortfolioSnapshot, BacktestResult) for field in fields(schema)
    }

    assert schema_fields.isdisjoint(
        {
            "model",
            "prediction",
            "probability",
            "threshold",
            "feature",
            "signal",
            "strategy",
            "pnl",
            "return",
            "sharpe",
            "drawdown",
            "turnover",
            "exposure",
        }
    )


def test_engine_source_uses_explicit_loop_without_queue_or_formula_helpers() -> None:
    source = inspect.getsource(engine_module.run_backtest)

    assert "for index" in source
    assert "market_order_for_target_at_open(" in source
    assert "execute_market_order(" in source
    assert "apply_fill(" in source
    assert "deque" not in source
    assert "queue" not in source.lower()
    assert "fill_price" not in source
    assert "cash_flow" not in source
    assert "raw_quantity" not in source
