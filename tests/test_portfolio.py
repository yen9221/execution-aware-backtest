from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from backtest.execution import Fill, execute_market_order
from backtest.orders import MarketOrder, Side
from backtest.portfolio import PortfolioError, PortfolioState, apply_fill

CREATED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
EXECUTED_AT = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)


def fill(
    side: Side,
    *,
    quantity: float = 2.0,
    reference_price: float = 100.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> Fill:
    return execute_market_order(
        MarketOrder(created_at=CREATED_AT, side=side, quantity=quantity),
        executed_at=EXECUTED_AT,
        reference_price=reference_price,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def test_valid_initial_all_cash_state() -> None:
    state = PortfolioState(cash=10_000.0)

    assert state == PortfolioState(
        cash=10_000.0, position_quantity=0.0, cumulative_fees=0.0
    )


def test_buy_updates_cash_position_and_fee_once() -> None:
    state = PortfolioState(cash=10_000.0)
    buy_fill = fill(Side.BUY)

    result = apply_fill(state, buy_fill)

    assert result.cash == pytest.approx(state.cash + buy_fill.cash_flow)
    assert result.cash != pytest.approx(
        state.cash + buy_fill.cash_flow - buy_fill.fee
    )
    assert result.position_quantity == pytest.approx(
        state.position_quantity + buy_fill.quantity
    )
    assert result.cumulative_fees == pytest.approx(
        state.cumulative_fees + buy_fill.fee
    )


def test_sell_updates_cash_position_and_fee_once() -> None:
    state = PortfolioState(cash=9_000.0, position_quantity=2.0)
    sell_fill = fill(Side.SELL)

    result = apply_fill(state, sell_fill)

    assert result.cash == pytest.approx(state.cash + sell_fill.cash_flow)
    assert result.cash != pytest.approx(
        state.cash + sell_fill.cash_flow - sell_fill.fee
    )
    assert result.position_quantity == pytest.approx(
        state.position_quantity - sell_fill.quantity
    )
    assert result.cumulative_fees == pytest.approx(
        state.cumulative_fees + sell_fill.fee
    )


def test_hand_calculated_buy_sell_round_trip() -> None:
    initial = PortfolioState(cash=10_000.0)

    after_buy = apply_fill(initial, fill(Side.BUY))
    final = apply_fill(after_buy, fill(Side.SELL))

    assert after_buy.cash == pytest.approx(9_799.6999)
    assert after_buy.position_quantity == pytest.approx(2.0)
    assert after_buy.cumulative_fees == pytest.approx(0.2001)
    assert final.cash == pytest.approx(9_999.4)
    assert final.position_quantity == pytest.approx(0.0)
    assert final.cumulative_fees == pytest.approx(0.4)


def test_zero_cost_round_trip_conserves_cash() -> None:
    initial = PortfolioState(cash=10_000.0)
    buy_fill = fill(Side.BUY, fee_rate=0.0, slippage_rate=0.0)
    sell_fill = fill(Side.SELL, fee_rate=0.0, slippage_rate=0.0)

    final = apply_fill(apply_fill(initial, buy_fill), sell_fill)

    assert final.cash == pytest.approx(initial.cash)
    assert final.position_quantity == pytest.approx(0.0)
    assert final.cumulative_fees == pytest.approx(0.0)


def test_insufficient_cash_is_rejected_without_mutation() -> None:
    state = PortfolioState(cash=200.0)
    original = replace(state)

    with pytest.raises(PortfolioError, match="insufficient cash"):
        apply_fill(state, fill(Side.BUY))

    assert state == original


@pytest.mark.parametrize("position", [0.0, 1.0])
def test_insufficient_position_prevents_short_selling(position: float) -> None:
    state = PortfolioState(cash=1_000.0, position_quantity=position)

    with pytest.raises(PortfolioError, match="insufficient position"):
        apply_fill(state, fill(Side.SELL))

    assert state.position_quantity == position


def test_success_returns_new_state_without_mutating_old_state() -> None:
    state = PortfolioState(cash=10_000.0)
    original = replace(state)

    result = apply_fill(state, fill(Side.BUY))

    assert state == original
    assert result is not state


def test_portfolio_state_is_immutable() -> None:
    state = PortfolioState(cash=10_000.0)

    with pytest.raises(FrozenInstanceError):
        state.cash = 0.0  # type: ignore[misc]


def test_invalid_state_type_is_rejected() -> None:
    with pytest.raises(PortfolioError, match="state must be a PortfolioState"):
        apply_fill({"cash": 10_000.0}, fill(Side.BUY))  # type: ignore[arg-type]


def test_invalid_fill_type_is_rejected() -> None:
    with pytest.raises(PortfolioError, match="fill must be a Fill"):
        apply_fill(PortfolioState(cash=10_000.0), object())  # type: ignore[arg-type]


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
    with pytest.raises(PortfolioError, match=field):
        apply_fill(state, fill(Side.BUY))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_non_finite_state_field_is_rejected(field: str, value: float) -> None:
    state = PortfolioState(cash=10_000.0)
    invalid = replace(state, **{field: value})

    with pytest.raises(PortfolioError, match=f"state.{field}.*finite"):
        apply_fill(invalid, fill(Side.BUY))


@pytest.mark.parametrize("field", ["cash", "position_quantity", "cumulative_fees"])
def test_boolean_state_field_is_rejected(field: str) -> None:
    state = PortfolioState(cash=10_000.0)
    invalid = replace(state, **{field: True})

    with pytest.raises(PortfolioError, match=f"state.{field}.*not boolean"):
        apply_fill(invalid, fill(Side.BUY))


def test_buy_fill_with_non_negative_cash_flow_is_rejected() -> None:
    invalid = replace(fill(Side.BUY), cash_flow=0.0)

    with pytest.raises(PortfolioError, match="buy fill.cash_flow must be negative"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


def test_sell_fill_with_non_positive_cash_flow_is_rejected() -> None:
    invalid = replace(fill(Side.SELL), cash_flow=0.0)

    with pytest.raises(PortfolioError, match="sell fill.cash_flow must be positive"):
        apply_fill(PortfolioState(cash=10_000.0, position_quantity=2.0), invalid)


def test_invalid_fill_side_is_rejected() -> None:
    invalid = replace(fill(Side.BUY), side="buy")  # type: ignore[arg-type]

    with pytest.raises(PortfolioError, match="fill.side must be a Side"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


@pytest.mark.parametrize("quantity", [0.0, -1.0])
def test_non_positive_fill_quantity_is_rejected(quantity: float) -> None:
    invalid = replace(fill(Side.BUY), quantity=quantity)

    with pytest.raises(PortfolioError, match="fill.quantity.*strictly positive"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fill_quantity_is_rejected(quantity: float) -> None:
    invalid = replace(fill(Side.BUY), quantity=quantity)

    with pytest.raises(PortfolioError, match="fill.quantity.*finite"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


@pytest.mark.parametrize("fee", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_fill_fee_is_rejected(fee: float) -> None:
    invalid = replace(fill(Side.BUY), fee=fee)
    message = "non-negative" if fee == -1.0 else "finite"

    with pytest.raises(PortfolioError, match=f"fill.fee.*{message}"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


@pytest.mark.parametrize("cash_flow", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fill_cash_flow_is_rejected(cash_flow: float) -> None:
    invalid = replace(fill(Side.BUY), cash_flow=cash_flow)

    with pytest.raises(PortfolioError, match="fill.cash_flow.*finite"):
        apply_fill(PortfolioState(cash=10_000.0), invalid)


def test_tiny_negative_cash_residual_normalizes_to_zero() -> None:
    state = PortfolioState(cash=1.0 - 5e-13)
    buy_fill = replace(fill(Side.BUY), quantity=1.0, cash_flow=-1.0)

    result = apply_fill(state, buy_fill, tolerance=1e-12)

    assert result.cash == 0.0


def test_material_negative_cash_is_rejected() -> None:
    state = PortfolioState(cash=0.99)
    buy_fill = replace(fill(Side.BUY), quantity=1.0, cash_flow=-1.0)

    with pytest.raises(PortfolioError, match="insufficient cash"):
        apply_fill(state, buy_fill, tolerance=1e-12)


def test_tiny_negative_position_residual_normalizes_to_zero() -> None:
    state = PortfolioState(cash=1_000.0, position_quantity=1.0 - 5e-13)
    sell_fill = replace(fill(Side.SELL), quantity=1.0, cash_flow=1.0)

    result = apply_fill(state, sell_fill, tolerance=1e-12)

    assert result.position_quantity == 0.0


def test_material_negative_position_is_rejected() -> None:
    state = PortfolioState(cash=1_000.0, position_quantity=0.99)
    sell_fill = replace(fill(Side.SELL), quantity=1.0, cash_flow=1.0)

    with pytest.raises(PortfolioError, match="insufficient position"):
        apply_fill(state, sell_fill, tolerance=1e-12)


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, 1.1e-6, float("nan"), float("inf"), float("-inf"), True, "1e-12"],
)
def test_invalid_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(PortfolioError, match="tolerance"):
        apply_fill(
            PortfolioState(cash=10_000.0),
            fill(Side.BUY),
            tolerance=tolerance,  # type: ignore[arg-type]
        )


def test_repeated_application_is_deterministic() -> None:
    state = PortfolioState(cash=10_000.0)
    buy_fill = fill(Side.BUY)

    first = apply_fill(state, buy_fill)
    second = apply_fill(state, buy_fill)

    assert first == second
    assert first is not second


def test_non_finite_new_state_values_are_rejected() -> None:
    state = PortfolioState(cash=1e308, position_quantity=1e308, cumulative_fees=1e308)
    buy_fill = replace(
        fill(Side.BUY),
        quantity=1e308,
        fee=1e308,
        cash_flow=-1.0,
    )

    with pytest.raises(PortfolioError, match="must remain finite"):
        apply_fill(state, buy_fill)
