import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backtest.allocation import (
    probabilities_to_continuous_target_weights,
    probabilities_to_target_weights,
)
from backtest.engine import run_target_weight_backtest
from backtest.events import Bar
from backtest.orders import Side
from backtest.portfolio import PortfolioState
from backtest.prediction_alignment import (
    TimestampedProbability,
    align_probabilities_to_bars,
)
from backtest.reporting import build_trade_log, summarize_backtest

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROBABILITIES = (0.50, 0.60, 0.70, 0.65, 0.80, 0.55, 0.55)
EXPECTED_WEIGHTS = (0.00, 0.20, 0.40, 0.30, 0.60, 0.10, 0.10)


def synthetic_bars(count: int = len(PROBABILITIES)) -> tuple[Bar, ...]:
    return tuple(
        Bar(
            timestamp=START + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
        for index in range(count)
    )


def timestamped_predictions(
    bars: tuple[Bar, ...],
    probabilities: tuple[float, ...] = PROBABILITIES,
) -> tuple[TimestampedProbability, ...]:
    return tuple(
        TimestampedProbability(bar.timestamp, probability)
        for bar, probability in zip(bars, probabilities, strict=True)
    )


def run_workflow(*, fee_rate: float, slippage_rate: float):
    bars = synthetic_bars()
    predictions = timestamped_predictions(bars)
    aligned = align_probabilities_to_bars(bars, predictions)
    targets = probabilities_to_continuous_target_weights(aligned)
    result = run_target_weight_backtest(
        bars=bars,
        targets=targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    return bars, predictions, aligned, targets, result


def target_values(targets) -> tuple[float, ...]:
    return tuple(target.weight for target in targets)


def realized_exposures(result) -> tuple[float, ...]:
    return tuple(
        snapshot.position_quantity * snapshot.close_price / snapshot.portfolio_value
        for snapshot in result.portfolio_history
    )


def test_exact_alignment_mapping_input_integrity_and_determinism() -> None:
    bars = synthetic_bars()
    predictions = timestamped_predictions(bars)
    original_bars = tuple(replace(bar) for bar in bars)
    original_predictions = tuple(replace(item) for item in predictions)

    first = run_workflow(fee_rate=0.0, slippage_rate=0.0)
    second = run_workflow(fee_rate=0.0, slippage_rate=0.0)

    assert len(bars) == len(predictions) == len(PROBABILITIES)
    assert all(
        prediction.timestamp == bars[index].timestamp
        for index, prediction in enumerate(predictions)
    )
    assert first[2] == PROBABILITIES
    assert target_values(first[3]) == pytest.approx(EXPECTED_WEIGHTS)
    assert first == second
    assert bars == original_bars
    assert predictions == original_predictions


def test_zero_cost_fill_timing_partial_rebalances_and_accounting() -> None:
    bars, _, _, targets, result = run_workflow(fee_rate=0.0, slippage_rate=0.0)
    expected_sides = (Side.BUY, Side.BUY, Side.SELL, Side.BUY, Side.SELL)
    expected_quantities = (2.0, 2.0, 1.0, 3.0, 5.0)
    expected_notionals = (200.0, 200.0, 100.0, 300.0, 500.0)
    expected_created_indices = (1, 2, 3, 4, 5)
    expected_execution_indices = (2, 3, 4, 5, 6)

    assert len(result.fills) == 5
    for fill, side, quantity, notional, created, executed in zip(
        result.fills,
        expected_sides,
        expected_quantities,
        expected_notionals,
        expected_created_indices,
        expected_execution_indices,
        strict=True,
    ):
        assert fill.side is side
        assert fill.order_created_at == bars[created].timestamp
        assert fill.executed_at == bars[executed].timestamp
        assert fill.executed_at == fill.order_created_at + timedelta(hours=1)
        assert fill.reference_price == 100.0
        assert fill.fill_price == 100.0
        assert fill.quantity == pytest.approx(quantity)
        assert fill.notional == pytest.approx(notional)
        expected_cash_flow = -notional if side is Side.BUY else notional
        assert fill.cash_flow == pytest.approx(expected_cash_flow)
        assert fill.fee == 0.0

    assert all(fill.executed_at != fill.order_created_at for fill in result.fills)
    assert all(fill.executed_at != bars[0].timestamp for fill in result.fills)
    assert all(fill.order_created_at != bars[0].timestamp for fill in result.fills)
    assert all(fill.order_created_at != bars[-1].timestamp for fill in result.fills)

    expected_cash = (1000.0, 1000.0, 800.0, 600.0, 700.0, 400.0, 900.0)
    expected_position = (0.0, 0.0, 2.0, 4.0, 3.0, 6.0, 1.0)
    expected_exposure = (0.0, 0.0, 0.2, 0.4, 0.3, 0.6, 0.1)
    for snapshot, cash, position, exposure in zip(
        result.portfolio_history,
        expected_cash,
        expected_position,
        expected_exposure,
        strict=True,
    ):
        assert snapshot.cash == pytest.approx(cash)
        assert snapshot.position_quantity == pytest.approx(position)
        assert snapshot.cumulative_fees == 0.0
        assert snapshot.close_price == 100.0
        assert snapshot.portfolio_value == pytest.approx(1000.0)
        actual_exposure = (
            snapshot.position_quantity * snapshot.close_price
            / snapshot.portfolio_value
        )
        assert actual_exposure == pytest.approx(exposure)

    assert result.fills[1].side is Side.BUY and result.fills[1].quantity == pytest.approx(2.0)
    assert result.fills[2].side is Side.SELL and result.fills[2].quantity == pytest.approx(1.0)
    assert result.portfolio_history[1].position_quantity == 0.0
    assert targets[0].weight == 0.0
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target == targets[-1]
    assert result.unexecuted_final_target.decision_bar_timestamp == bars[-1].timestamp


def test_zero_cost_reporting_uses_actual_fills_and_realized_snapshots() -> None:
    _, _, _, targets, result = run_workflow(fee_rate=0.0, slippage_rate=0.0)
    trade_log = build_trade_log(result)
    summary = summarize_backtest(result)
    exposures = realized_exposures(result)

    assert len(trade_log) == len(result.fills) == summary.trade_count == 5
    assert tuple(record.notional for record in trade_log) == pytest.approx(
        tuple(fill.notional for fill in result.fills)
    )
    assert summary.total_fees == 0.0
    assert summary.turnover == pytest.approx(1.3)
    assert summary.average_exposure == pytest.approx(math.fsum(exposures) / 7)
    assert summary.average_exposure == pytest.approx(1.6 / 7)
    assert exposures != target_values(targets)
    assert targets[-1].weight == pytest.approx(0.1)
    assert targets[-2].weight == pytest.approx(0.1)
    assert all(fill.order_created_at != result.unexecuted_final_target.decision_bar_timestamp
               for fill in result.fills)


def test_repeated_fractional_targets_with_execution_opportunities_do_not_retrade() -> None:
    probabilities = (0.50, 0.60, 0.70, 0.65, 0.65, 0.80, 0.55, 0.55)
    expected_weights = (0.00, 0.20, 0.40, 0.30, 0.30, 0.60, 0.10, 0.10)
    bars = synthetic_bars(len(probabilities))
    predictions = timestamped_predictions(bars, probabilities)
    aligned = align_probabilities_to_bars(bars, predictions)
    targets = probabilities_to_continuous_target_weights(aligned)
    result = run_target_weight_backtest(
        bars=bars,
        targets=targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )

    assert target_values(targets) == pytest.approx(expected_weights)
    assert targets[3] == targets[4]
    first_execution_bar = bars[4]
    second_execution_bar = bars[5]
    assert any(
        fill.order_created_at == bars[3].timestamp
        and fill.executed_at == first_execution_bar.timestamp
        and fill.side is Side.SELL
        for fill in result.fills
    )
    assert all(fill.order_created_at != bars[4].timestamp for fill in result.fills)
    assert all(fill.executed_at != second_execution_bar.timestamp for fill in result.fills)
    for index in (4, 5):
        snapshot = result.portfolio_history[index]
        assert snapshot.cash == pytest.approx(700.0)
        assert snapshot.position_quantity == pytest.approx(3.0)
        assert (
            snapshot.position_quantity * snapshot.close_price
            / snapshot.portfolio_value
        ) == pytest.approx(0.30)


def test_cost_aware_diagnostic_preserves_targets_and_uses_realized_outcomes() -> None:
    _, _, _, zero_targets, zero_result = run_workflow(
        fee_rate=0.0, slippage_rate=0.0
    )
    _, _, _, cost_targets, cost_result = run_workflow(
        fee_rate=0.001, slippage_rate=0.0005
    )
    repeated = run_workflow(fee_rate=0.001, slippage_rate=0.0005)[-1]

    assert cost_targets == zero_targets
    assert cost_result == repeated
    assert len(cost_result.fills) == len(build_trade_log(cost_result))
    assert summarize_backtest(cost_result).trade_count == len(cost_result.fills)
    assert all(
        fill.fill_price > fill.reference_price
        for fill in cost_result.fills
        if fill.side is Side.BUY
    )
    assert all(
        fill.fill_price < fill.reference_price
        for fill in cost_result.fills
        if fill.side is Side.SELL
    )
    assert all(fill.fee > 0.0 for fill in cost_result.fills)
    assert summarize_backtest(cost_result).total_fees == pytest.approx(
        math.fsum(fill.fee for fill in cost_result.fills)
    )
    assert cost_result.fills[1].quantity < zero_result.fills[1].quantity
    assert realized_exposures(cost_result) != pytest.approx(EXPECTED_WEIGHTS)
    assert cost_result.portfolio_history != zero_result.portfolio_history
    assert cost_result.final_state != zero_result.final_state


def test_existing_mapping_regressions_remain_unchanged() -> None:
    continuous = probabilities_to_continuous_target_weights(PROBABILITIES)
    three_state = probabilities_to_target_weights(
        (0.20, 0.80, 0.55, 0.35, 0.90, 0.45),
        lower_threshold=0.40,
        upper_threshold=0.70,
    )
    assert target_values(continuous) == pytest.approx(EXPECTED_WEIGHTS)
    assert target_values(three_state) == (0.0, 1.0, 0.5, 0.0, 1.0, 0.5)
