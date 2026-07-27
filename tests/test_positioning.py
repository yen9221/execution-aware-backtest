import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

import backtest.positioning as positioning_module
from backtest.execution import Fill, execute_market_order
from backtest.orders import MarketOrder, Side
from backtest.portfolio import PortfolioState, apply_fill
from backtest.positioning import (
    CASH_TARGET,
    LONG_TARGET,
    PendingTarget,
    PendingTargetWeight,
    PositioningError,
    TargetPosition,
    TargetWeight,
    create_pending_target,
    create_pending_target_weight,
    market_order_for_target_at_open,
    market_order_for_target_weight_at_open,
    target_position_to_weight,
    target_weight_to_position,
)

DECISION_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
EXECUTION_AT = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
REFERENCE_OPEN = 100.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_valid_target_weight_values_are_preserved(weight: float) -> None:
    target = TargetWeight(weight)
    assert target.weight == weight
    assert type(target.weight) is float


@pytest.mark.parametrize("weight", [-0.01, -5e-324, 1.01, 1.0000000000000002])
def test_out_of_range_target_weight_is_rejected_without_clipping(weight: float) -> None:
    with pytest.raises(PositioningError, match=r"inclusive range \[0\.0, 1\.0\]"):
        TargetWeight(weight)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_target_weight_is_rejected(weight: float) -> None:
    with pytest.raises(PositioningError, match="finite"):
        TargetWeight(weight)


@pytest.mark.parametrize("weight", [True, False])
def test_boolean_target_weight_is_rejected(weight: bool) -> None:
    with pytest.raises(PositioningError, match="not boolean"):
        TargetWeight(weight)


@pytest.mark.parametrize("weight", ["0.5", None, object()])
def test_nonnumeric_target_weight_is_rejected(weight: object) -> None:
    with pytest.raises(PositioningError, match="numeric"):
        TargetWeight(weight)  # type: ignore[arg-type]


def test_target_weight_is_immutable_and_has_value_equality() -> None:
    first = TargetWeight(0.5)
    second = TargetWeight(0.5)
    assert first == second
    assert first is not second
    with pytest.raises(FrozenInstanceError):
        first.weight = 0.75  # type: ignore[misc]


def test_target_weight_does_not_round_valid_values() -> None:
    weight = 0.12345678901234566
    assert TargetWeight(weight).weight == weight


def test_canonical_endpoint_constants_are_exact_and_immutable() -> None:
    assert CASH_TARGET == TargetWeight(0.0)
    assert LONG_TARGET == TargetWeight(1.0)
    assert CASH_TARGET is not LONG_TARGET
    with pytest.raises(FrozenInstanceError):
        CASH_TARGET.weight = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        LONG_TARGET.weight = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (TargetPosition.CASH, CASH_TARGET),
        (TargetPosition.LONG, LONG_TARGET),
    ],
)
def test_binary_position_converts_to_exact_endpoint_weight(
    position: TargetPosition,
    expected: TargetWeight,
) -> None:
    original = position
    first = target_position_to_weight(position)
    second = target_position_to_weight(position)
    assert type(first) is TargetWeight
    assert first == second == expected
    assert position is original


@pytest.mark.parametrize("invalid", [0, 1, True, "cash", None, TargetWeight(0.0)])
def test_invalid_raw_target_position_conversion_is_rejected(invalid: object) -> None:
    with pytest.raises(PositioningError, match="TargetPosition"):
        target_position_to_weight(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("weight", "expected"),
    [(CASH_TARGET, TargetPosition.CASH), (LONG_TARGET, TargetPosition.LONG)],
)
def test_endpoint_weight_converts_back_to_binary_position(
    weight: TargetWeight,
    expected: TargetPosition,
) -> None:
    assert target_weight_to_position(weight) is expected


@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75])
def test_fractional_weight_is_rejected_by_binary_adapter(weight: float) -> None:
    with pytest.raises(PositioningError, match="fractional.*binary execution"):
        target_weight_to_position(TargetWeight(weight))


@pytest.mark.parametrize("invalid", [0.0, 1.0, None, "0.5", TargetPosition.CASH])
def test_invalid_raw_weight_conversion_is_rejected(invalid: object) -> None:
    with pytest.raises(PositioningError, match="TargetWeight"):
        target_weight_to_position(invalid)  # type: ignore[arg-type]


def test_fractional_weight_cannot_enter_pending_binary_target() -> None:
    with pytest.raises(PositioningError, match="TargetPosition"):
        create_pending_target(
            decision_bar_timestamp=DECISION_AT,
            target=TargetWeight(0.5),  # type: ignore[arg-type]
        )


def pending(target: TargetPosition = TargetPosition.LONG) -> PendingTarget:
    return create_pending_target(
        decision_bar_timestamp=DECISION_AT,
        target=target,
    )


def convert(
    state: PortfolioState,
    target: TargetPosition = TargetPosition.LONG,
    *,
    pending_target: PendingTarget | None = None,
    execution_bar_timestamp: datetime = EXECUTION_AT,
    reference_open: float = REFERENCE_OPEN,
    fee_rate: float = FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
    tolerance: float = 1e-12,
) -> MarketOrder | None:
    return market_order_for_target_at_open(
        state=state,
        pending_target=pending_target or pending(target),
        execution_bar_timestamp=execution_bar_timestamp,
        reference_open=reference_open,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        tolerance=tolerance,
    )


def test_target_position_contains_only_cash_and_long() -> None:
    assert list(TargetPosition) == [TargetPosition.CASH, TargetPosition.LONG]
    assert TargetPosition.CASH.value == 0
    assert TargetPosition.LONG.value == 1


@pytest.mark.parametrize("target", list(TargetPosition))
def test_valid_pending_target_creation(target: TargetPosition) -> None:
    result = pending(target)

    assert result == PendingTarget(
        decision_bar_timestamp=DECISION_AT,
        target=target,
    )


def test_pending_target_timestamp_is_normalized_to_utc() -> None:
    local_timestamp = datetime(
        2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))
    )

    result = create_pending_target(
        decision_bar_timestamp=local_timestamp,
        target=TargetPosition.CASH,
    )

    assert result.decision_bar_timestamp == DECISION_AT
    assert result.decision_bar_timestamp.tzinfo is timezone.utc


def test_naive_decision_timestamp_is_rejected() -> None:
    with pytest.raises(PositioningError, match="decision_bar_timestamp.*timezone"):
        create_pending_target(
            decision_bar_timestamp=datetime(2024, 1, 1),
            target=TargetPosition.CASH,
        )


def test_non_datetime_decision_timestamp_is_rejected() -> None:
    with pytest.raises(PositioningError, match="decision_bar_timestamp must be a datetime"):
        create_pending_target(
            decision_bar_timestamp="2024-01-01T00:00:00Z",  # type: ignore[arg-type]
            target=TargetPosition.CASH,
        )


@pytest.mark.parametrize("target", ["cash", "long", 0, 1, True, False])
def test_non_enum_target_is_rejected(target: object) -> None:
    with pytest.raises(PositioningError, match="target must be a TargetPosition"):
        create_pending_target(
            decision_bar_timestamp=DECISION_AT,
            target=target,  # type: ignore[arg-type]
        )


def test_pending_target_is_immutable() -> None:
    intent = pending()

    with pytest.raises(FrozenInstanceError):
        intent.target = TargetPosition.CASH  # type: ignore[misc]


def test_pending_target_schema_is_price_free() -> None:
    assert [field.name for field in fields(PendingTarget)] == [
        "decision_bar_timestamp",
        "target",
    ]


def test_pending_target_has_no_model_or_threshold_fields() -> None:
    field_names = {field.name for field in fields(PendingTarget)}

    assert field_names.isdisjoint(
        {"price", "reference_open", "quantity", "prediction", "probability", "threshold"}
    )


def test_invalid_state_type_is_rejected() -> None:
    with pytest.raises(PositioningError, match="state must be a PortfolioState"):
        market_order_for_target_at_open(
            state={"cash": 10_000.0},  # type: ignore[arg-type]
            pending_target=pending(),
            execution_bar_timestamp=EXECUTION_AT,
            reference_open=REFERENCE_OPEN,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
        )


def test_invalid_pending_target_type_is_rejected() -> None:
    with pytest.raises(PositioningError, match="pending_target must be a PendingTarget"):
        market_order_for_target_at_open(
            state=PortfolioState(cash=10_000.0),
            pending_target=TargetPosition.LONG,  # type: ignore[arg-type]
            execution_bar_timestamp=EXECUTION_AT,
            reference_open=REFERENCE_OPEN,
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
        )


@pytest.mark.parametrize(
    ("field", "state"),
    [
        ("state.cash", PortfolioState(cash=-1.0)),
        (
            "state.position_quantity",
            PortfolioState(cash=10_000.0, position_quantity=-1.0),
        ),
        (
            "state.cumulative_fees",
            PortfolioState(cash=10_000.0, cumulative_fees=-1.0),
        ),
    ],
)
def test_negative_state_field_is_rejected(field: str, state: PortfolioState) -> None:
    with pytest.raises(PositioningError, match=field):
        convert(state)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_non_finite_state_field_is_rejected(field: str, value: float) -> None:
    state = replace(PortfolioState(cash=10_000.0), **{field: value})

    with pytest.raises(PositioningError, match=f"state.{field}.*finite"):
        convert(state)


@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_boolean_state_field_is_rejected(field: str) -> None:
    state = replace(PortfolioState(cash=10_000.0), **{field: True})

    with pytest.raises(PositioningError, match=f"state.{field}.*not boolean"):
        convert(state)


def test_invalid_target_inside_pending_target_is_rejected() -> None:
    invalid = replace(pending(), target=1)  # type: ignore[arg-type]

    with pytest.raises(PositioningError, match="pending_target.target"):
        convert(PortfolioState(cash=10_000.0), pending_target=invalid)


def test_naive_execution_timestamp_is_rejected() -> None:
    with pytest.raises(PositioningError, match="execution_bar_timestamp.*timezone"):
        convert(
            PortfolioState(cash=10_000.0),
            execution_bar_timestamp=datetime(2024, 1, 1, 1),
        )


def test_non_datetime_execution_timestamp_is_rejected() -> None:
    with pytest.raises(PositioningError, match="execution_bar_timestamp must be a datetime"):
        convert(
            PortfolioState(cash=10_000.0),
            execution_bar_timestamp="2024-01-01T01:00:00Z",  # type: ignore[arg-type]
        )


def test_execution_at_same_instant_is_rejected() -> None:
    with pytest.raises(PositioningError, match="strictly later.*equal"):
        convert(
            PortfolioState(cash=10_000.0),
            execution_bar_timestamp=DECISION_AT,
        )


def test_execution_before_decision_is_rejected() -> None:
    with pytest.raises(PositioningError, match="strictly later.*before"):
        convert(
            PortfolioState(cash=10_000.0),
            execution_bar_timestamp=DECISION_AT - timedelta(seconds=1),
        )


def test_timestamps_are_compared_as_actual_utc_instants() -> None:
    decision_at = datetime(
        2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))
    )
    same_instant = datetime(
        2023, 12, 31, 19, tzinfo=timezone(timedelta(hours=-5))
    )
    intent = create_pending_target(
        decision_bar_timestamp=decision_at,
        target=TargetPosition.LONG,
    )

    with pytest.raises(PositioningError, match="strictly later.*equal"):
        convert(
            PortfolioState(cash=10_000.0),
            pending_target=intent,
            execution_bar_timestamp=same_instant,
        )


def test_execution_timestamp_need_not_be_exactly_one_hour_later() -> None:
    result = convert(
        PortfolioState(cash=10_000.0),
        execution_bar_timestamp=DECISION_AT + timedelta(seconds=1),
    )

    assert result is not None
    assert result.side is Side.BUY


@pytest.mark.parametrize(
    "reference_open",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "100"],
)
def test_invalid_reference_open_is_rejected(reference_open: object) -> None:
    with pytest.raises(PositioningError, match="reference_open"):
        convert(
            PortfolioState(cash=10_000.0),
            reference_open=reference_open,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "fee_rate",
    [-0.001, float("nan"), float("inf"), float("-inf"), True, "0.001"],
)
def test_invalid_fee_rate_is_rejected(fee_rate: object) -> None:
    with pytest.raises(PositioningError, match="fee_rate"):
        convert(
            PortfolioState(cash=10_000.0),
            fee_rate=fee_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "slippage_rate",
    [-0.001, 1.0, 2.0, float("nan"), float("inf"), float("-inf"), True, "0.0005"],
)
def test_invalid_slippage_rate_is_rejected(slippage_rate: object) -> None:
    with pytest.raises(PositioningError, match="slippage_rate"):
        convert(
            PortfolioState(cash=10_000.0),
            slippage_rate=slippage_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, 1.1e-6, float("nan"), float("inf"), float("-inf"), True, "1e-12"],
)
def test_invalid_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(PositioningError, match="tolerance"):
        convert(
            PortfolioState(cash=10_000.0),
            tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_cash_state_plus_cash_target_returns_none() -> None:
    assert convert(PortfolioState(cash=10_000.0), TargetPosition.CASH) is None


def test_long_state_plus_long_target_returns_none_without_rebalancing() -> None:
    state = PortfolioState(cash=1_000.0, position_quantity=2.0)

    assert convert(state, TargetPosition.LONG) is None


def test_long_state_plus_cash_target_sells_entire_position() -> None:
    state = PortfolioState(cash=1_000.0, position_quantity=2.5)

    order = convert(state, TargetPosition.CASH)

    assert order == MarketOrder(
        created_at=DECISION_AT,
        side=Side.SELL,
        quantity=2.5,
    )


def test_sell_order_preserves_decision_bar_timestamp() -> None:
    intent = pending(TargetPosition.CASH)

    order = convert(
        PortfolioState(cash=1_000.0, position_quantity=2.0),
        pending_target=intent,
    )

    assert order is not None
    assert order.created_at == intent.decision_bar_timestamp
    assert order.created_at != EXECUTION_AT


def test_cash_state_plus_long_target_creates_buy_order() -> None:
    order = convert(PortfolioState(cash=10_000.0))

    assert order is not None
    assert order.side is Side.BUY
    assert order.quantity > 0


def test_buy_order_preserves_decision_bar_timestamp() -> None:
    intent = pending(TargetPosition.LONG)

    order = convert(PortfolioState(cash=10_000.0), pending_target=intent)

    assert order is not None
    assert order.created_at == intent.decision_bar_timestamp
    assert order.created_at != EXECUTION_AT


def test_buy_quantity_uses_observable_execution_open() -> None:
    state = PortfolioState(cash=10_000.0)

    low_open_order = convert(state, reference_open=100.0)
    high_open_order = convert(state, reference_open=200.0)

    assert low_open_order is not None
    assert high_open_order is not None
    assert low_open_order.quantity > high_open_order.quantity


def test_buy_quantity_includes_adverse_slippage_in_affordability() -> None:
    state = PortfolioState(cash=10_000.0)

    no_slippage = convert(state, slippage_rate=0.0)
    with_slippage = convert(state, slippage_rate=0.01)

    assert no_slippage is not None
    assert with_slippage is not None
    assert with_slippage.quantity < no_slippage.quantity


def test_buy_quantity_includes_fee_in_affordability() -> None:
    state = PortfolioState(cash=10_000.0)

    no_fee = convert(state, fee_rate=0.0)
    with_fee = convert(state, fee_rate=0.01)

    assert no_fee is not None
    assert with_fee is not None
    assert with_fee.quantity < no_fee.quantity


def test_zero_cost_quantity_is_conservatively_below_theoretical_maximum() -> None:
    state = PortfolioState(cash=10_000.0)

    order = convert(state, reference_open=100.0, fee_rate=0.0, slippage_rate=0.0)

    assert order is not None
    assert order.quantity == math.nextafter(100.0, 0.0)
    assert order.quantity <= state.cash / 100.0
    assert order.quantity == pytest.approx(state.cash / 100.0)


def test_zero_cash_plus_long_target_returns_none() -> None:
    assert convert(PortfolioState(cash=0.0)) is None


def test_tiny_cash_that_cannot_produce_quantity_returns_none() -> None:
    assert convert(PortfolioState(cash=5e-324), reference_open=1.0) is None


def test_non_finite_effective_unit_cost_returns_none() -> None:
    assert (
        convert(
            PortfolioState(cash=10_000.0),
            reference_open=1e308,
            fee_rate=1e308,
        )
        is None
    )


def test_tiny_position_within_tolerance_plus_cash_returns_none() -> None:
    state = PortfolioState(cash=10_000.0, position_quantity=5e-13)

    assert convert(state, TargetPosition.CASH, tolerance=1e-12) is None


def test_tiny_position_within_tolerance_plus_long_is_treated_as_cash() -> None:
    state = PortfolioState(cash=10_000.0, position_quantity=5e-13)

    order = convert(state, TargetPosition.LONG, tolerance=1e-12)

    assert order is not None
    assert order.side is Side.BUY


def test_position_above_tolerance_plus_cash_sells_full_quantity() -> None:
    state = PortfolioState(cash=10_000.0, position_quantity=1.1e-12)

    order = convert(state, TargetPosition.CASH, tolerance=1e-12)

    assert order is not None
    assert order.quantity == state.position_quantity


def test_conversion_does_not_execute_or_update_portfolio() -> None:
    state = PortfolioState(cash=10_000.0)
    original = replace(state)

    result = convert(state)

    assert isinstance(result, MarketOrder)
    assert not isinstance(result, Fill)
    assert state == original


def test_original_pending_target_remains_unchanged() -> None:
    intent = pending()
    original = replace(intent)

    convert(PortfolioState(cash=10_000.0), pending_target=intent)

    assert intent == original


def test_repeated_identical_calls_are_deterministic() -> None:
    state = PortfolioState(cash=10_000.0)
    intent = pending()

    first = convert(state, pending_target=intent)
    second = convert(state, pending_target=intent)

    assert first == second
    assert first is not second


def test_generated_buy_executes_and_applies_without_negative_cash() -> None:
    initial = PortfolioState(cash=10_000.0)
    original = replace(initial)
    intent = pending(TargetPosition.LONG)
    order = convert(initial, pending_target=intent)

    assert order is not None
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    final = apply_fill(initial, fill)

    assert math.isfinite(final.cash)
    assert final.cash >= 0
    assert final.position_quantity > 0
    assert initial == original


def test_generated_sell_executes_and_applies_without_short_position() -> None:
    initial = PortfolioState(cash=1_000.0, position_quantity=2.0)
    order = convert(initial, TargetPosition.CASH)

    assert order is not None
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    final = apply_fill(initial, fill)

    assert final.cash > initial.cash
    assert final.position_quantity == 0.0


def test_different_execution_opens_produce_exactly_scaled_quantities() -> None:
    state = PortfolioState(cash=10_000.0)

    first = convert(state, reference_open=100.0)
    second = convert(state, reference_open=200.0)

    assert first is not None
    assert second is not None
    assert first.quantity == pytest.approx(second.quantity * 2)


class InvalidOffset(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta:
        return timedelta(hours=24)


def test_invalid_decision_timestamp_offset_is_rejected() -> None:
    invalid = datetime(2024, 1, 1, tzinfo=InvalidOffset())

    with pytest.raises(PositioningError, match="invalid UTC offset"):
        create_pending_target(
            decision_bar_timestamp=invalid,
            target=TargetPosition.CASH,
        )


def pending_weight(weight: float = 0.5) -> PendingTargetWeight:
    return create_pending_target_weight(
        decision_bar_timestamp=DECISION_AT,
        target=TargetWeight(weight),
    )


def convert_weight(
    state: PortfolioState,
    weight: float = 0.5,
    *,
    pending_target: PendingTargetWeight | None = None,
    execution_bar_timestamp: datetime = EXECUTION_AT,
    reference_open: float = REFERENCE_OPEN,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    rebalance_tolerance: float = 0.0,
    minimum_trade_notional: float = 0.0,
) -> MarketOrder | None:
    return market_order_for_target_weight_at_open(
        state=state,
        pending_target=pending_target or pending_weight(weight),
        execution_bar_timestamp=execution_bar_timestamp,
        reference_open=reference_open,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        rebalance_tolerance=rebalance_tolerance,
        minimum_trade_notional=minimum_trade_notional,
    )


def apply_zero_cost_weight_order(
    state: PortfolioState,
    weight: float,
    *,
    reference_open: float = REFERENCE_OPEN,
) -> tuple[MarketOrder | None, PortfolioState]:
    order = convert_weight(state, weight, reference_open=reference_open)
    if order is None:
        return None, state
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=reference_open,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    return order, apply_fill(state, fill)


def test_valid_pending_target_weight_creation_and_utc_normalization() -> None:
    local_timestamp = datetime(
        2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))
    )
    result = create_pending_target_weight(
        decision_bar_timestamp=local_timestamp,
        target=TargetWeight(0.7),
    )

    assert result == PendingTargetWeight(DECISION_AT, TargetWeight(0.7))
    assert result.decision_bar_timestamp.tzinfo is timezone.utc


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        (datetime(2024, 1, 1), "timezone"),
        ("2024-01-01T00:00:00Z", "must be a datetime"),
    ],
)
def test_invalid_pending_weight_decision_timestamp_is_rejected(
    timestamp: object, message: str
) -> None:
    with pytest.raises(PositioningError, match=message):
        create_pending_target_weight(
            decision_bar_timestamp=timestamp,  # type: ignore[arg-type]
            target=TargetWeight(0.5),
        )


@pytest.mark.parametrize("target", [0.5, TargetPosition.LONG, None, True])
def test_invalid_raw_pending_weight_target_is_rejected(target: object) -> None:
    with pytest.raises(PositioningError, match="TargetWeight"):
        create_pending_target_weight(
            decision_bar_timestamp=DECISION_AT,
            target=target,  # type: ignore[arg-type]
        )


def test_pending_target_weight_is_immutable_and_has_minimal_schema() -> None:
    intent = pending_weight(0.5)
    assert [field.name for field in fields(PendingTargetWeight)] == [
        "decision_bar_timestamp",
        "target",
    ]
    assert {field.name for field in fields(PendingTargetWeight)}.isdisjoint(
        {
            "price",
            "reference_open",
            "quantity",
            "model",
            "prediction",
            "probability",
            "threshold",
        }
    )
    with pytest.raises(FrozenInstanceError):
        intent.target = TargetWeight(0.75)  # type: ignore[misc]


def test_weight_converter_rejects_invalid_state_and_pending_types() -> None:
    with pytest.raises(PositioningError, match="state must be a PortfolioState"):
        market_order_for_target_weight_at_open(
            state={"cash": 1000.0},  # type: ignore[arg-type]
            pending_target=pending_weight(),
            execution_bar_timestamp=EXECUTION_AT,
            reference_open=REFERENCE_OPEN,
            fee_rate=0.0,
            slippage_rate=0.0,
            rebalance_tolerance=0.0,
            minimum_trade_notional=0.0,
        )
    with pytest.raises(PositioningError, match="PendingTargetWeight"):
        market_order_for_target_weight_at_open(
            state=PortfolioState(cash=1000.0),
            pending_target=pending(),  # type: ignore[arg-type]
            execution_bar_timestamp=EXECUTION_AT,
            reference_open=REFERENCE_OPEN,
            fee_rate=0.0,
            slippage_rate=0.0,
            rebalance_tolerance=0.0,
            minimum_trade_notional=0.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cash", -1.0, "non-negative"),
        ("position_quantity", -1.0, "non-negative"),
        ("cumulative_fees", -1.0, "non-negative"),
        ("cash", float("nan"), "finite"),
        ("position_quantity", float("inf"), "finite"),
        ("cumulative_fees", float("-inf"), "finite"),
        ("cash", True, "not boolean"),
        ("position_quantity", False, "not boolean"),
        ("cumulative_fees", True, "not boolean"),
    ],
)
def test_weight_converter_rejects_invalid_state_fields(
    field: str, value: object, message: str
) -> None:
    state = replace(PortfolioState(cash=1000.0), **{field: value})
    with pytest.raises(PositioningError, match=message):
        convert_weight(state)


def test_weight_converter_rejects_invalid_target_inside_pending_object() -> None:
    invalid = replace(pending_weight(), target=0.5)  # type: ignore[arg-type]
    with pytest.raises(PositioningError, match="pending_target.target.*TargetWeight"):
        convert_weight(PortfolioState(cash=1000.0), pending_target=invalid)


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        (datetime(2024, 1, 1, 1), "timezone"),
        ("2024-01-01T01:00:00Z", "must be a datetime"),
        (DECISION_AT, "equal"),
        (DECISION_AT - timedelta(seconds=1), "before"),
    ],
)
def test_weight_converter_rejects_invalid_execution_timestamp(
    timestamp: object, message: str
) -> None:
    with pytest.raises(PositioningError, match=message):
        convert_weight(
            PortfolioState(cash=1000.0),
            execution_bar_timestamp=timestamp,  # type: ignore[arg-type]
        )


def test_weight_timestamps_compare_equivalent_utc_instants() -> None:
    same_instant = datetime(
        2023, 12, 31, 19, tzinfo=timezone(timedelta(hours=-5))
    )
    with pytest.raises(PositioningError, match="equal"):
        convert_weight(
            PortfolioState(cash=1000.0),
            execution_bar_timestamp=same_instant,
        )


@pytest.mark.parametrize(
    "reference_open",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "100"],
)
def test_weight_converter_rejects_invalid_reference_open(
    reference_open: object,
) -> None:
    with pytest.raises(PositioningError, match="reference_open"):
        convert_weight(
            PortfolioState(cash=1000.0),
            reference_open=reference_open,  # type: ignore[arg-type]
        )


def test_weight_converter_rejects_non_finite_intermediate_arithmetic() -> None:
    with pytest.raises(PositioningError, match="current_asset_value.*finite"):
        convert_weight(
            PortfolioState(cash=0.0, position_quantity=1e308),
            reference_open=1e308,
        )
    with pytest.raises(PositioningError, match="pre_trade_portfolio_value.*finite"):
        convert_weight(
            PortfolioState(cash=1e308, position_quantity=1e308),
            reference_open=1.0,
        )


def test_weight_converter_rejects_underflowed_resulting_quantity() -> None:
    with pytest.raises(PositioningError, match="quantity.*strictly positive"):
        convert_weight(
            PortfolioState(cash=5e-324),
            1.0,
            reference_open=1e308,
        )


@pytest.mark.parametrize(
    ("weight", "side", "quantity"),
    [
        (0.70, Side.BUY, 3.0),
        (0.25, Side.SELL, 1.5),
        (1.00, Side.BUY, 6.0),
        (0.00, Side.SELL, 4.0),
    ],
)
def test_central_weight_sizing_is_hand_checkable(
    weight: float, side: Side, quantity: float
) -> None:
    order = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0), weight
    )
    assert order == MarketOrder(DECISION_AT, side, quantity)


def test_central_current_weight_returns_none_exactly() -> None:
    assert (
        convert_weight(PortfolioState(cash=600.0, position_quantity=4.0), 0.4)
        is None
    )


@pytest.mark.parametrize(
    ("state", "weight", "expected_side"),
    [
        (PortfolioState(cash=1000.0), 0.25, Side.BUY),
        (PortfolioState(cash=1000.0), 0.50, Side.BUY),
        (PortfolioState(cash=1000.0), 1.00, Side.BUY),
        (PortfolioState(cash=0.0, position_quantity=10.0), 0.75, Side.SELL),
        (PortfolioState(cash=0.0, position_quantity=10.0), 0.50, Side.SELL),
        (PortfolioState(cash=0.0, position_quantity=10.0), 0.00, Side.SELL),
        (PortfolioState(cash=600.0, position_quantity=4.0), 0.70, Side.BUY),
        (PortfolioState(cash=600.0, position_quantity=4.0), 0.25, Side.SELL),
    ],
)
def test_representative_weight_orders_reach_target_under_zero_costs(
    state: PortfolioState, weight: float, expected_side: Side
) -> None:
    initial_value = state.cash + state.position_quantity * REFERENCE_OPEN
    order, final = apply_zero_cost_weight_order(state, weight)
    assert order is not None
    assert order.side is expected_side
    assert final.cash >= 0
    assert final.position_quantity >= 0
    assert final.cumulative_fees == state.cumulative_fees
    final_value = final.cash + final.position_quantity * REFERENCE_OPEN
    assert final_value == pytest.approx(initial_value)
    assert final.position_quantity * REFERENCE_OPEN / final_value == pytest.approx(
        weight
    )


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_zero_value_portfolio_returns_none_for_every_target(weight: float) -> None:
    assert convert_weight(PortfolioState(cash=0.0), weight) is None


def test_zero_cash_full_long_same_and_full_targets_return_none() -> None:
    state = PortfolioState(cash=0.0, position_quantity=10.0)
    assert convert_weight(state, 1.0) is None


def test_tiny_nonzero_weight_delta_is_not_suppressed_by_tolerance() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(state, math.nextafter(0.4, 1.0))
    assert order is not None
    assert order.side is Side.BUY
    assert order.quantity > 0


def test_weight_buy_affordability_uses_conditional_float_step() -> None:
    state = PortfolioState(cash=0.1)
    order = convert_weight(state, 1.0, reference_open=11.0)
    assert order is not None
    assert order.quantity == math.nextafter(state.cash / 11.0, 0.0)
    assert order.quantity * 11.0 <= state.cash


def test_weight_buy_and_sell_caps_are_respected() -> None:
    buy_state = PortfolioState(cash=600.0, position_quantity=4.0)
    buy = convert_weight(buy_state, 1.0)
    sell = convert_weight(buy_state, 0.0)
    assert buy is not None and buy.quantity <= buy_state.cash / REFERENCE_OPEN
    assert sell is not None and sell.quantity <= buy_state.position_quantity


def test_weight_order_preserves_timestamp_is_deterministic_and_inputs_unchanged() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0, cumulative_fees=2.0)
    intent = pending_weight(0.7)
    state_copy = replace(state)
    intent_copy = replace(intent)
    first = convert_weight(state, pending_target=intent)
    second = convert_weight(state, pending_target=intent)
    assert first == second
    assert first is not second
    assert first is not None and first.created_at == intent.decision_bar_timestamp
    assert state == state_copy
    assert intent == intent_copy


def test_weight_positioning_returns_only_order_and_does_not_execute_or_apply() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    result = convert_weight(state, 0.7)
    assert type(result) is MarketOrder
    assert not isinstance(result, Fill)
    assert state == PortfolioState(cash=600.0, position_quantity=4.0)
    source = inspect.getsource(
        positioning_module.market_order_for_target_weight_at_open
    )
    assert "execute_market_order" not in source
    assert "apply_fill" not in source


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.40, 0.70, 1.0])
def test_central_post_trade_accounting_at_zero_costs(weight: float) -> None:
    initial = PortfolioState(cash=600.0, position_quantity=4.0, cumulative_fees=2.0)
    _, final = apply_zero_cost_weight_order(initial, weight)
    value = final.cash + final.position_quantity * REFERENCE_OPEN
    assert final.cash >= 0
    assert final.position_quantity >= 0
    assert final.cumulative_fees == 2.0
    assert value == pytest.approx(1000.0)
    assert final.position_quantity * REFERENCE_OPEN / value == pytest.approx(weight)


def test_continuous_cost_arguments_are_required_keyword_only() -> None:
    kwargs = {
        "state": PortfolioState(cash=1000.0),
        "pending_target": pending_weight(0.5),
        "execution_bar_timestamp": EXECUTION_AT,
        "reference_open": REFERENCE_OPEN,
    }
    with pytest.raises(TypeError, match="fee_rate"):
        market_order_for_target_weight_at_open(**kwargs, slippage_rate=0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="slippage_rate"):
        market_order_for_target_weight_at_open(**kwargs, fee_rate=0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("fee_rate", [0.0, 0.01, 1.0])
def test_continuous_valid_fee_rates_are_accepted(fee_rate: float) -> None:
    assert convert_weight(PortfolioState(cash=1000.0), fee_rate=fee_rate) is not None


@pytest.mark.parametrize(
    "fee_rate",
    [-0.01, True, "0.01", None, float("nan"), float("inf"), float("-inf")],
)
def test_continuous_invalid_fee_rates_are_rejected(fee_rate: object) -> None:
    with pytest.raises(PositioningError, match="fee_rate"):
        convert_weight(
            PortfolioState(cash=1000.0), fee_rate=fee_rate  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("slippage_rate", [0.0, 0.02, math.nextafter(1.0, 0.0)])
def test_continuous_valid_slippage_rates_are_accepted(
    slippage_rate: float,
) -> None:
    assert (
        convert_weight(
            PortfolioState(cash=1000.0), slippage_rate=slippage_rate
        )
        is not None
    )


@pytest.mark.parametrize(
    "slippage_rate",
    [
        -0.01,
        1.0,
        1.01,
        True,
        "0.02",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_continuous_invalid_slippage_rates_are_rejected(
    slippage_rate: object,
) -> None:
    with pytest.raises(PositioningError, match="slippage_rate"):
        convert_weight(
            PortfolioState(cash=1000.0),
            slippage_rate=slippage_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("reference_open", "fee_rate", "slippage_rate"),
    [(100.0, 1e308, 0.02), (1e308, 0.0, 0.9)],
)
def test_non_finite_effective_buy_unit_cost_is_rejected(
    reference_open: float, fee_rate: float, slippage_rate: float
) -> None:
    with pytest.raises(PositioningError, match="effective buy unit cost.*finite"):
        convert_weight(
            PortfolioState(cash=600.0),
            1.0,
            reference_open=reference_open,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )


def execute_cost_aware_central(weight: float):
    initial = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(
        initial,
        weight,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    if order is None:
        return initial, None, None, initial
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    return initial, order, fill, apply_fill(initial, fill)


def test_cost_aware_central_partial_buy_execution_and_accounting() -> None:
    initial, order, fill, final = execute_cost_aware_central(0.70)
    assert order == MarketOrder(DECISION_AT, Side.BUY, 3.0)
    assert fill is not None
    assert fill.fill_price == 102.0
    assert fill.notional == 306.0
    assert fill.fee == pytest.approx(3.06)
    assert fill.cash_flow == pytest.approx(-309.06)
    assert final.cash == pytest.approx(290.94)
    assert final.position_quantity == 7.0
    assert final.cumulative_fees == pytest.approx(3.06)
    final_value = final.cash + final.position_quantity * REFERENCE_OPEN
    realized_weight = final.position_quantity * REFERENCE_OPEN / final_value
    assert final_value == pytest.approx(990.94)
    assert final_value == pytest.approx(1000.0 + fill.cash_flow + 300.0)
    assert math.isfinite(realized_weight)
    assert 0.0 <= realized_weight <= 1.0
    assert realized_weight != pytest.approx(0.70)
    assert initial == PortfolioState(cash=600.0, position_quantity=4.0)


def test_cost_aware_full_buy_is_capped_and_affordable() -> None:
    initial, order, fill, final = execute_cost_aware_central(1.0)
    assert order is not None and fill is not None
    maximum = initial.cash / (REFERENCE_OPEN * 1.02 * 1.01)
    assert order.side is Side.BUY
    assert order.quantity == pytest.approx(maximum)
    assert order.quantity < 6.0
    assert -fill.cash_flow <= initial.cash
    assert final.cash >= 0
    assert final.position_quantity > initial.position_quantity
    realized = final.position_quantity * REFERENCE_OPEN / (
        final.cash + final.position_quantity * REFERENCE_OPEN
    )
    assert 0.0 <= realized <= 1.0


def test_cost_aware_buy_nextafter_is_conditional_and_deterministic() -> None:
    state = PortfolioState(cash=0.1)
    raw_maximum = state.cash / (0.1 * 1.001)
    first = convert_weight(state, 1.0, reference_open=0.1, fee_rate=0.001)
    second = convert_weight(state, 1.0, reference_open=0.1, fee_rate=0.001)
    unbound = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        0.70,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    assert first == second
    assert first is not None
    assert first.quantity == math.nextafter(raw_maximum, 0.0)
    assert first.quantity * 0.1 + first.quantity * 0.1 * 0.001 <= state.cash
    assert unbound is not None and unbound.quantity == 3.0


def test_stage_6_full_weight_buy_uses_two_float_steps_and_remains_affordable() -> None:
    state = PortfolioState(cash=1000.0)
    decision_at = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    execution_at = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)
    reference_open = 42613.57
    fee_rate = 0.001
    slippage_rate = 0.0005
    raw_maximum = state.cash / (
        reference_open * (1 + slippage_rate) * (1 + fee_rate)
    )

    order = convert_weight(
        state,
        1.0,
        pending_target=create_pending_target_weight(
            decision_bar_timestamp=decision_at,
            target=TargetWeight(1.0),
        ),
        execution_bar_timestamp=execution_at,
        reference_open=reference_open,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )

    assert order is not None
    assert order.side is Side.BUY
    one_step_quantity = math.nextafter(raw_maximum, 0.0)
    expected_fill_price = reference_open * (1 + slippage_rate)
    one_step_notional = one_step_quantity * expected_fill_price
    assert one_step_notional + one_step_notional * fee_rate > state.cash
    assert order.quantity == math.nextafter(one_step_quantity, 0.0)

    fill = execute_market_order(
        order,
        executed_at=execution_at,
        reference_price=reference_open,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    expected_cash_requirement = fill.notional + fill.fee
    assert expected_cash_requirement <= state.cash

    final = apply_fill(state, fill)
    assert final.cash == 0.0
    assert final.cash >= 0.0


def test_cost_aware_central_partial_sell_execution_and_accounting() -> None:
    initial, order, fill, final = execute_cost_aware_central(0.25)
    assert order == MarketOrder(DECISION_AT, Side.SELL, 1.5)
    assert fill is not None
    assert fill.fill_price == 98.0
    assert fill.notional == 147.0
    assert fill.fee == pytest.approx(1.47)
    assert fill.cash_flow == pytest.approx(145.53)
    assert final == PortfolioState(
        cash=pytest.approx(745.53),
        position_quantity=2.5,
        cumulative_fees=pytest.approx(1.47),
    )
    final_value = final.cash + final.position_quantity * REFERENCE_OPEN
    assert final_value == pytest.approx(995.53)
    assert final_value < 1000.0
    assert order.quantity <= initial.position_quantity


def test_cost_aware_full_sell_does_not_create_short_position() -> None:
    _, order, fill, final = execute_cost_aware_central(0.0)
    assert order == MarketOrder(DECISION_AT, Side.SELL, 4.0)
    assert fill is not None
    assert final.cash == pytest.approx(988.08)
    assert final.position_quantity == 0.0
    assert final.cumulative_fees == pytest.approx(3.92)


def test_cost_aware_exact_target_creates_no_trade_or_cost() -> None:
    initial, order, fill, final = execute_cost_aware_central(0.4)
    assert order is None
    assert fill is None
    assert final is initial
    assert final.cumulative_fees == 0.0


def test_continuous_positioning_source_preserves_responsibility_boundaries() -> None:
    source = inspect.getsource(
        positioning_module.market_order_for_target_weight_at_open
    )
    for forbidden in (
        "execute_market_order",
        "apply_fill",
        "Fill(",
        "summarize_backtest",
        "build_trade_log",
    ):
        assert forbidden not in source


def test_rebalance_tolerance_is_required_and_keyword_only() -> None:
    kwargs = {
        "state": PortfolioState(cash=1000.0),
        "pending_target": pending_weight(0.5),
        "execution_bar_timestamp": EXECUTION_AT,
        "reference_open": REFERENCE_OPEN,
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
    }
    with pytest.raises(TypeError, match="rebalance_tolerance"):
        market_order_for_target_weight_at_open(**kwargs)  # type: ignore[arg-type]
    parameter = inspect.signature(
        market_order_for_target_weight_at_open
    ).parameters["rebalance_tolerance"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("rebalance_tolerance", [0.0, 0.01, 0.5, 1.0])
def test_valid_rebalance_tolerances_are_accepted(
    rebalance_tolerance: float,
) -> None:
    result = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        0.4,
        rebalance_tolerance=rebalance_tolerance,
    )
    assert result is None


@pytest.mark.parametrize(
    "rebalance_tolerance",
    [
        -0.01,
        1.01,
        True,
        False,
        "0.01",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_rebalance_tolerances_are_rejected(
    rebalance_tolerance: object,
) -> None:
    with pytest.raises(PositioningError, match="rebalance_tolerance"):
        convert_weight(
            PortfolioState(cash=1000.0),
            rebalance_tolerance=rebalance_tolerance,  # type: ignore[arg-type]
        )


def test_inside_rebalance_band_returns_none() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    assert convert_weight(state, 0.405, rebalance_tolerance=0.01) is None


def test_exact_rebalance_boundary_is_inclusive_without_hidden_epsilon() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    target = 0.65
    exact_deviation = abs(0.4 - target)
    assert exact_deviation == 0.25
    assert convert_weight(state, target, rebalance_tolerance=exact_deviation) is None
    below_boundary = math.nextafter(exact_deviation, 0.0)
    order = convert_weight(state, target, rebalance_tolerance=below_boundary)
    assert order is not None
    assert order.side is Side.BUY


@pytest.mark.parametrize(
    ("target", "side", "quantity"),
    [(0.42, Side.BUY, 0.2), (0.38, Side.SELL, 0.2)],
)
def test_outside_rebalance_band_sizes_to_exact_target(
    target: float, side: Side, quantity: float
) -> None:
    order = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        target,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
    )
    assert order is not None
    assert order.side is side
    assert order.quantity == pytest.approx(quantity)


@pytest.mark.parametrize(
    ("state", "target"),
    [
        (PortfolioState(cash=1000.0), 1.0),
        (PortfolioState(cash=0.0, position_quantity=10.0), 0.0),
        (PortfolioState(cash=600.0, position_quantity=4.0), 0.75),
    ],
)
def test_tolerance_one_suppresses_every_valid_rebalance(
    state: PortfolioState, target: float
) -> None:
    assert convert_weight(state, target, rebalance_tolerance=1.0) is None


def test_zero_value_portfolio_remains_no_trade_with_tolerance() -> None:
    assert (
        convert_weight(
            PortfolioState(cash=0.0), 1.0, rebalance_tolerance=0.01
        )
        is None
    )


def test_tolerance_uses_pre_trade_reference_open_weight_before_sizing() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    # The pre-trade reference-open weight is 0.4. A slipped fill price is not
    # part of this trigger, and the huge finite fee would overflow if BUY
    # sizing were incorrectly attempted before the tolerance check.
    assert (
        convert_weight(
            state,
            0.405,
            fee_rate=1e308,
            slippage_rate=0.9,
            rebalance_tolerance=0.01,
        )
        is None
    )


def test_tiny_deviation_depends_only_on_explicit_tolerance() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    current_weight = 0.4
    tiny_target = math.nextafter(current_weight, 1.0)
    deviation = tiny_target - current_weight
    zero_tolerance_order = convert_weight(
        state, tiny_target, rebalance_tolerance=0.0
    )
    positive_tolerance_order = convert_weight(
        state, tiny_target, rebalance_tolerance=deviation
    )
    assert zero_tolerance_order is not None
    assert zero_tolerance_order.side is Side.BUY
    assert zero_tolerance_order.quantity > 0
    assert positive_tolerance_order is None


@pytest.mark.parametrize("target", [0.42, 0.38])
def test_outside_band_orders_execute_and_apply_with_costs(target: float) -> None:
    initial = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(
        initial,
        target,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
    )
    assert order is not None
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    final = apply_fill(initial, fill)
    assert final.cash >= 0
    assert final.position_quantity >= 0
    assert final.cumulative_fees == pytest.approx(fill.fee)
    assert initial == PortfolioState(cash=600.0, position_quantity=4.0)


def test_inside_band_integration_has_no_fill_fee_or_state_change() -> None:
    initial = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(
        initial,
        0.405,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
    )
    fill = None if order is None else execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    final = initial if fill is None else apply_fill(initial, fill)
    assert order is None
    assert fill is None
    assert final is initial
    assert final.cumulative_fees == 0.0


def test_minimum_trade_notional_is_required_and_keyword_only() -> None:
    kwargs = {
        "state": PortfolioState(cash=1000.0),
        "pending_target": pending_weight(0.5),
        "execution_bar_timestamp": EXECUTION_AT,
        "reference_open": REFERENCE_OPEN,
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
        "rebalance_tolerance": 0.0,
    }
    with pytest.raises(TypeError, match="minimum_trade_notional"):
        market_order_for_target_weight_at_open(**kwargs)  # type: ignore[arg-type]
    parameter = inspect.signature(
        market_order_for_target_weight_at_open
    ).parameters["minimum_trade_notional"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("minimum", [0.0, 1.0, 20.0, 1e9])
def test_valid_minimum_trade_notionals_are_accepted(minimum: float) -> None:
    result = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        0.42,
        minimum_trade_notional=minimum,
    )
    assert result is None or isinstance(result, MarketOrder)


@pytest.mark.parametrize(
    "minimum",
    [
        -0.01,
        True,
        False,
        "20",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_minimum_trade_notionals_are_rejected(minimum: object) -> None:
    with pytest.raises(PositioningError, match="minimum_trade_notional"):
        convert_weight(
            PortfolioState(cash=1000.0),
            minimum_trade_notional=minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("minimum", "expected_order"),
    [(21.0, False), (20.400000000000002, True), (20.0, True)],
)
def test_buy_minimum_notional_boundary(
    minimum: float, expected_order: bool
) -> None:
    order = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        0.42,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=minimum,
    )
    assert (order is not None) is expected_order
    if order is not None:
        assert order == MarketOrder(DECISION_AT, Side.BUY, 0.2)


@pytest.mark.parametrize(
    ("minimum", "expected_order"),
    [(20.0, False), (19.6, True), (19.0, True)],
)
def test_sell_minimum_notional_boundary(
    minimum: float, expected_order: bool
) -> None:
    order = convert_weight(
        PortfolioState(cash=600.0, position_quantity=4.0),
        0.38,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=minimum,
    )
    assert (order is not None) is expected_order
    if order is not None:
        assert order == MarketOrder(DECISION_AT, Side.SELL, 0.2)


def test_minimum_boundary_uses_slipped_notional_and_excludes_fee_and_cash_flow() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    # BUY reference notional is 20.0, slipped notional is 20.4, and cash
    # outflow including fee is 20.604.
    assert (
        convert_weight(
            state,
            0.42,
            fee_rate=0.01,
            slippage_rate=0.02,
            rebalance_tolerance=0.01,
            minimum_trade_notional=20.2,
        )
        is not None
    )
    assert (
        convert_weight(
            state,
            0.42,
            fee_rate=0.01,
            slippage_rate=0.02,
            rebalance_tolerance=0.01,
            minimum_trade_notional=20.5,
        )
        is None
    )
    # SELL reference notional is 20.0 while adverse slipped notional is 19.6.
    assert (
        convert_weight(
            state,
            0.38,
            fee_rate=0.01,
            slippage_rate=0.02,
            rebalance_tolerance=0.01,
            minimum_trade_notional=19.8,
        )
        is None
    )


def test_minimum_boundary_has_no_hidden_epsilon() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    actual_notional = 0.2 * (REFERENCE_OPEN * 1.02)
    at_boundary = convert_weight(
        state,
        0.42,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=actual_notional,
    )
    one_step_above = convert_weight(
        state,
        0.42,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=math.nextafter(actual_notional, math.inf),
    )
    assert at_boundary is not None
    assert one_step_above is None


def test_tolerance_suppresses_before_minimum_notional_calculation() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    assert (
        convert_weight(
            state,
            0.405,
            fee_rate=1e308,
            slippage_rate=0.9,
            rebalance_tolerance=0.01,
            minimum_trade_notional=1e308,
        )
        is None
    )


def test_affordability_cap_precedes_minimum_notional_filter() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    base = convert_weight(
        state,
        1.0,
        fee_rate=0.01,
        slippage_rate=0.02,
        minimum_trade_notional=0.0,
    )
    assert base is not None
    final_notional = base.quantity * (REFERENCE_OPEN * 1.02)
    permitted = convert_weight(
        state,
        1.0,
        fee_rate=0.01,
        slippage_rate=0.02,
        minimum_trade_notional=math.nextafter(final_notional, 0.0),
    )
    suppressed = convert_weight(
        state,
        1.0,
        fee_rate=0.01,
        slippage_rate=0.02,
        minimum_trade_notional=math.nextafter(final_notional, math.inf),
    )
    assert base.quantity == pytest.approx(600.0 / 103.02)
    assert base.quantity < 6.0
    assert permitted == base
    assert suppressed is None


def test_tiny_trade_is_controlled_by_minimum_notional_only_after_tolerance() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    tiny_target = math.nextafter(0.4, 1.0)
    allowed = convert_weight(
        state,
        tiny_target,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    assert allowed is not None
    tiny_notional = allowed.quantity * REFERENCE_OPEN
    suppressed = convert_weight(
        state,
        tiny_target,
        rebalance_tolerance=0.0,
        minimum_trade_notional=math.nextafter(tiny_notional, math.inf),
    )
    assert allowed.quantity == 5.684341886080802e-16
    assert suppressed is None


@pytest.mark.parametrize(
    ("target", "side"), [(0.42, Side.BUY), (0.38, Side.SELL)]
)
def test_allowed_minimum_notional_order_executes_and_applies(
    target: float, side: Side
) -> None:
    initial = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(
        initial,
        target,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=0.0,
    )
    assert order is not None and order.side is side
    fill = execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    final = apply_fill(initial, fill)
    assert final.cash >= 0
    assert final.position_quantity >= 0
    assert final.cumulative_fees == pytest.approx(fill.fee)


@pytest.mark.parametrize("target", [0.42, 0.38])
def test_suppressed_minimum_notional_order_has_no_fill_fee_or_state_change(
    target: float,
) -> None:
    initial = PortfolioState(cash=600.0, position_quantity=4.0)
    order = convert_weight(
        initial,
        target,
        fee_rate=0.01,
        slippage_rate=0.02,
        rebalance_tolerance=0.01,
        minimum_trade_notional=1e9,
    )
    fill = None if order is None else execute_market_order(
        order,
        executed_at=EXECUTION_AT,
        reference_price=REFERENCE_OPEN,
        fee_rate=0.01,
        slippage_rate=0.02,
    )
    assert order is None
    assert fill is None
    assert initial == PortfolioState(cash=600.0, position_quantity=4.0)
    assert initial.cumulative_fees == 0.0


def test_minimum_notional_calls_are_deterministic_and_not_accumulated() -> None:
    state = PortfolioState(cash=600.0, position_quantity=4.0)
    first = convert_weight(state, 0.42, minimum_trade_notional=21.0)
    second = convert_weight(state, 0.42, minimum_trade_notional=21.0)
    third = convert_weight(state, 0.42, minimum_trade_notional=0.0)
    assert first is None
    assert second is None
    assert third == MarketOrder(DECISION_AT, Side.BUY, 0.2)
    assert state == PortfolioState(cash=600.0, position_quantity=4.0)
