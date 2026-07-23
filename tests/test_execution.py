from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtest.data import load_bars_csv
from backtest.execution import ExecutionError, Fill, execute_market_order
from backtest.orders import MarketOrder, Side

CREATED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
EXECUTED_AT = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "simple_bars.csv"


def order(side: Side = Side.BUY, quantity: float = 2.0) -> MarketOrder:
    return MarketOrder(created_at=CREATED_AT, side=side, quantity=quantity)


def execute(
    market_order: MarketOrder,
    *,
    executed_at: datetime = EXECUTED_AT,
    reference_price: float = 100.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> Fill:
    return execute_market_order(
        market_order,
        executed_at=executed_at,
        reference_price=reference_price,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def test_buy_hand_calculation() -> None:
    fill = execute(order(Side.BUY))

    assert fill.fill_price == pytest.approx(100.05)
    assert fill.notional == pytest.approx(200.10)
    assert fill.fee == pytest.approx(0.2001)
    assert fill.cash_flow == pytest.approx(-200.3001)


def test_sell_hand_calculation() -> None:
    fill = execute(order(Side.SELL))

    assert fill.fill_price == pytest.approx(99.95)
    assert fill.notional == pytest.approx(199.90)
    assert fill.fee == pytest.approx(0.1999)
    assert fill.cash_flow == pytest.approx(199.7001)


def test_buy_slippage_increases_fill_price() -> None:
    fill = execute(order(Side.BUY), slippage_rate=0.01)

    assert fill.fill_price > fill.reference_price


def test_sell_slippage_decreases_fill_price() -> None:
    fill = execute(order(Side.SELL), slippage_rate=0.01)

    assert fill.fill_price < fill.reference_price


@pytest.mark.parametrize("side", list(Side))
def test_zero_slippage_uses_reference_price(side: Side) -> None:
    fill = execute(order(side), slippage_rate=0.0)

    assert fill.fill_price == fill.reference_price


@pytest.mark.parametrize("side", list(Side))
def test_zero_fee_produces_zero_fee(side: Side) -> None:
    fill = execute(order(side), fee_rate=0.0)

    assert fill.fee == 0.0


def test_fee_is_calculated_once_from_fill_notional() -> None:
    fill = execute(order(Side.BUY))

    assert fill.fee == pytest.approx(fill.notional * fill.fee_rate)
    assert fill.fee != pytest.approx(fill.reference_price * fill.quantity * fill.fee_rate)


def test_buy_cash_flow_charges_fee_once() -> None:
    fill = execute(order(Side.BUY))

    assert fill.cash_flow == pytest.approx(-(fill.notional + fill.fee))
    assert fill.cash_flow < 0


def test_sell_cash_flow_charges_fee_once() -> None:
    fill = execute(order(Side.SELL))

    assert fill.cash_flow == pytest.approx(fill.notional - fill.fee)
    assert fill.cash_flow > 0


@pytest.mark.parametrize("side", list(Side))
def test_fill_preserves_quantity_and_side(side: Side) -> None:
    fill = execute(order(side, quantity=3.5))

    assert fill.side is side
    assert fill.quantity == 3.5


def test_returned_timestamps_are_normalized_to_utc() -> None:
    created_at = datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    executed_at = datetime(2023, 12, 31, 20, tzinfo=timezone(timedelta(hours=-5)))

    fill = execute(
        MarketOrder(created_at=created_at, side=Side.BUY, quantity=2.0),
        executed_at=executed_at,
    )

    assert fill.order_created_at == CREATED_AT
    assert fill.executed_at == EXECUTED_AT
    assert fill.order_created_at.tzinfo is timezone.utc
    assert fill.executed_at.tzinfo is timezone.utc


def test_equal_execution_instant_is_rejected_across_timezones() -> None:
    executed_at = datetime(2023, 12, 31, 19, tzinfo=timezone(timedelta(hours=-5)))

    with pytest.raises(ExecutionError, match="strictly later.*equal"):
        execute(order(), executed_at=executed_at)


def test_execution_before_creation_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="strictly later.*before"):
        execute(order(), executed_at=CREATED_AT - timedelta(seconds=1))


def test_naive_creation_timestamp_is_rejected() -> None:
    market_order = MarketOrder(
        created_at=datetime(2024, 1, 1), side=Side.BUY, quantity=2.0
    )

    with pytest.raises(ExecutionError, match="order.created_at.*timezone"):
        execute(market_order)


def test_naive_execution_timestamp_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="executed_at.*timezone"):
        execute(order(), executed_at=datetime(2024, 1, 1, 1))


def test_non_datetime_creation_timestamp_is_rejected() -> None:
    market_order = MarketOrder(
        created_at="2024-01-01T00:00:00Z",  # type: ignore[arg-type]
        side=Side.BUY,
        quantity=2.0,
    )

    with pytest.raises(ExecutionError, match="order.created_at must be a datetime"):
        execute(market_order)


def test_non_datetime_execution_timestamp_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="executed_at must be a datetime"):
        execute(order(), executed_at="2024-01-01T01:00:00Z")  # type: ignore[arg-type]


def test_invalid_side_is_rejected() -> None:
    market_order = MarketOrder(
        created_at=CREATED_AT, side="BUY", quantity=2.0  # type: ignore[arg-type]
    )

    with pytest.raises(ExecutionError, match="order.side must be a Side"):
        execute(market_order)


@pytest.mark.parametrize("quantity", [0.0, -1.0])
def test_non_positive_quantity_is_rejected(quantity: float) -> None:
    with pytest.raises(ExecutionError, match="order.quantity.*strictly positive"):
        execute(order(quantity=quantity))


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_quantity_is_rejected(quantity: float) -> None:
    with pytest.raises(ExecutionError, match="order.quantity.*finite"):
        execute(order(quantity=quantity))


def test_boolean_quantity_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="order.quantity.*not boolean"):
        execute(order(quantity=True))  # type: ignore[arg-type]


def test_nonnumeric_quantity_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="order.quantity.*numeric"):
        execute(order(quantity="2"))  # type: ignore[arg-type]


@pytest.mark.parametrize("reference_price", [0.0, -1.0])
def test_non_positive_reference_price_is_rejected(reference_price: float) -> None:
    with pytest.raises(ExecutionError, match="reference_price.*strictly positive"):
        execute(order(), reference_price=reference_price)


@pytest.mark.parametrize(
    "reference_price", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_reference_price_is_rejected(reference_price: float) -> None:
    with pytest.raises(ExecutionError, match="reference_price.*finite"):
        execute(order(), reference_price=reference_price)


def test_boolean_reference_price_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="reference_price.*not boolean"):
        execute(order(), reference_price=True)  # type: ignore[arg-type]


def test_nonnumeric_reference_price_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="reference_price.*numeric"):
        execute(order(), reference_price="100")  # type: ignore[arg-type]


def test_negative_fee_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="fee_rate.*non-negative"):
        execute(order(), fee_rate=-0.001)


@pytest.mark.parametrize("fee_rate", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fee_rate_is_rejected(fee_rate: float) -> None:
    with pytest.raises(ExecutionError, match="fee_rate.*finite"):
        execute(order(), fee_rate=fee_rate)


def test_boolean_fee_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="fee_rate.*not boolean"):
        execute(order(), fee_rate=True)  # type: ignore[arg-type]


def test_nonnumeric_fee_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="fee_rate.*numeric"):
        execute(order(), fee_rate="0.001")  # type: ignore[arg-type]


def test_negative_slippage_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*0 <= slippage_rate < 1"):
        execute(order(), slippage_rate=-0.001)


@pytest.mark.parametrize(
    "slippage_rate", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_slippage_rate_is_rejected(slippage_rate: float) -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*finite"):
        execute(order(), slippage_rate=slippage_rate)


def test_boolean_slippage_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*not boolean"):
        execute(order(), slippage_rate=True)  # type: ignore[arg-type]


def test_nonnumeric_slippage_rate_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*numeric"):
        execute(order(), slippage_rate="0.0005")  # type: ignore[arg-type]


@pytest.mark.parametrize("slippage_rate", [1.0, 2.0])
def test_sell_slippage_cannot_make_fill_price_non_positive(slippage_rate: float) -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*< 1"):
        execute(order(Side.SELL), slippage_rate=slippage_rate)


def test_buy_also_rejects_slippage_rate_of_one() -> None:
    with pytest.raises(ExecutionError, match="slippage_rate.*< 1"):
        execute(order(Side.BUY), slippage_rate=1.0)


def test_calculated_non_finite_output_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="calculated fill_price.*finite"):
        execute(order(), reference_price=1e308, slippage_rate=0.9)


def test_calculated_non_finite_notional_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="calculated notional.*finite"):
        execute(order(quantity=1e200), reference_price=1e200, slippage_rate=0.0)


def test_calculated_non_finite_fee_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="calculated fee.*finite"):
        execute(order(quantity=1.0), reference_price=1e200, fee_rate=1e200)


def test_calculated_non_finite_cash_flow_is_rejected() -> None:
    with pytest.raises(ExecutionError, match="calculated cash_flow.*finite"):
        execute(
            order(quantity=1.0),
            reference_price=1e308,
            fee_rate=0.9,
            slippage_rate=0.0,
        )


def test_sell_fee_cannot_make_cash_flow_non_positive() -> None:
    with pytest.raises(ExecutionError, match="cash_flow for a sell must be positive"):
        execute(order(Side.SELL), fee_rate=1.0)


def test_market_order_is_immutable() -> None:
    market_order = order()

    with pytest.raises(FrozenInstanceError):
        market_order.quantity = 3.0  # type: ignore[misc]


def test_fill_is_immutable() -> None:
    fill = execute(order())

    with pytest.raises(FrozenInstanceError):
        fill.fee = 0.0  # type: ignore[misc]


def test_repeated_calls_are_deterministic_and_do_not_modify_order() -> None:
    market_order = order()
    original = MarketOrder(
        created_at=market_order.created_at,
        side=market_order.side,
        quantity=market_order.quantity,
    )

    first = execute(market_order)
    second = execute(market_order)

    assert first == second
    assert first is not second
    assert market_order == original


def test_supplied_second_bar_open_is_used_as_reference_price() -> None:
    bars = load_bars_csv(FIXTURE)
    market_order = MarketOrder(
        created_at=bars[0].timestamp,
        side=Side.BUY,
        quantity=2.0,
    )

    fill = execute_market_order(
        market_order,
        executed_at=bars[1].timestamp,
        reference_price=bars[1].open,
        fee_rate=0.001,
        slippage_rate=0.0005,
    )

    assert fill.reference_price == bars[1].open
    assert fill.reference_price != bars[0].open
    assert fill.fill_price == pytest.approx(bars[1].open * 1.0005)
