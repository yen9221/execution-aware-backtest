import inspect
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.allocation as allocation_module
from backtest.allocation import (
    probabilities_to_continuous_target_weights,
    probabilities_to_target_weights,
)
from backtest.engine import TargetWeightBacktestResult, run_target_weight_backtest
from backtest.events import Bar
from backtest.orders import Side
from backtest.portfolio import PortfolioState
from backtest.positioning import TargetWeight
from backtest.prediction_alignment import (
    TimestampedProbability,
    align_probabilities_to_bars,
)
from backtest.reporting import build_trade_log, summarize_backtest

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROBABILITIES = (0.50, 0.60, 0.75, 0.65, 0.80, 0.55, 0.55, 0.45)
POLICY_NAMES = ("binary", "three_state", "continuous")


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    policy_name: str
    target_weights: tuple[float, ...]
    unique_target_weights: tuple[float, ...]
    target_transition_count: int
    repeated_adjacent_target_count: int
    total_absolute_target_change: float
    fill_count: int
    buy_count: int
    sell_count: int
    turnover: float
    average_realized_exposure: float
    total_fees: float
    full_entry_count: int
    full_exit_count: int
    partial_buy_count: int
    partial_sell_count: int
    no_trade_target_count: int
    final_target_unexecuted: bool
    final_cash: float
    final_position_quantity: float
    final_portfolio_value: float


@dataclass(frozen=True, slots=True)
class PolicyRun:
    bars: tuple[Bar, ...]
    predictions: tuple[TimestampedProbability, ...]
    aligned_probabilities: tuple[float, ...]
    targets: tuple[TargetWeight, ...]
    result: TargetWeightBacktestResult
    row: ComparisonRow


def synthetic_bars() -> tuple[Bar, ...]:
    return tuple(
        Bar(
            timestamp=START + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
        for index in range(len(PROBABILITIES))
    )


def timestamped_predictions(
    bars: tuple[Bar, ...],
) -> tuple[TimestampedProbability, ...]:
    return tuple(
        TimestampedProbability(bar.timestamp, probability)
        for bar, probability in zip(bars, PROBABILITIES, strict=True)
    )


def policy_targets(
    policy_name: str, aligned_probabilities: tuple[float, ...]
) -> tuple[TargetWeight, ...]:
    if policy_name == "binary":
        return tuple(
            TargetWeight(0.0 if probability <= 0.50 else 1.0)
            for probability in aligned_probabilities
        )
    if policy_name == "three_state":
        return probabilities_to_target_weights(
            aligned_probabilities,
            lower_threshold=0.55,
            upper_threshold=0.70,
        )
    if policy_name == "continuous":
        return probabilities_to_continuous_target_weights(aligned_probabilities)
    raise ValueError(f"unknown policy_name {policy_name!r}")


def target_values(targets: tuple[TargetWeight, ...]) -> tuple[float, ...]:
    return tuple(target.weight for target in targets)


def exposure_at_snapshot(result: TargetWeightBacktestResult, index: int) -> float:
    snapshot = result.portfolio_history[index]
    return (
        snapshot.position_quantity * snapshot.close_price
        / snapshot.portfolio_value
    )


def comparison_row(
    policy_name: str,
    bars: tuple[Bar, ...],
    targets: tuple[TargetWeight, ...],
    result: TargetWeightBacktestResult,
) -> ComparisonRow:
    weights = target_values(targets)
    transitions = tuple(
        left != right for left, right in zip(weights, weights[1:])
    )
    summary = summarize_backtest(result)
    created_timestamps = {fill.order_created_at for fill in result.fills}
    observable_targets = bars[:-1]

    full_entry_count = 0
    full_exit_count = 0
    partial_buy_count = 0
    partial_sell_count = 0
    for fill in result.fills:
        execution_index = next(
            index
            for index, bar in enumerate(bars)
            if bar.timestamp == fill.executed_at
        )
        pre_exposure = exposure_at_snapshot(result, execution_index - 1)
        post_exposure = exposure_at_snapshot(result, execution_index)
        if fill.side is Side.BUY:
            if pre_exposure == pytest.approx(0.0) and post_exposure == pytest.approx(1.0):
                full_entry_count += 1
            elif 0.0 < post_exposure < 1.0:
                partial_buy_count += 1
        else:
            if pre_exposure == pytest.approx(1.0) and post_exposure == pytest.approx(0.0):
                full_exit_count += 1
            elif post_exposure > 0.0:
                partial_sell_count += 1

    return ComparisonRow(
        policy_name=policy_name,
        target_weights=weights,
        unique_target_weights=tuple(sorted(set(weights))),
        target_transition_count=sum(transitions),
        repeated_adjacent_target_count=len(transitions) - sum(transitions),
        total_absolute_target_change=math.fsum(
            abs(right - left) for left, right in zip(weights, weights[1:])
        ),
        fill_count=len(result.fills),
        buy_count=summary.buy_count,
        sell_count=summary.sell_count,
        turnover=summary.turnover,
        average_realized_exposure=summary.average_exposure,
        total_fees=summary.total_fees,
        full_entry_count=full_entry_count,
        full_exit_count=full_exit_count,
        partial_buy_count=partial_buy_count,
        partial_sell_count=partial_sell_count,
        no_trade_target_count=sum(
            bar.timestamp not in created_timestamps for bar in observable_targets
        ),
        final_target_unexecuted=result.unexecuted_final_target is not None,
        final_cash=result.final_state.cash,
        final_position_quantity=result.final_state.position_quantity,
        final_portfolio_value=result.portfolio_history[-1].portfolio_value,
    )


def run_policy(policy_name: str) -> PolicyRun:
    bars = synthetic_bars()
    predictions = timestamped_predictions(bars)
    aligned = align_probabilities_to_bars(bars, predictions)
    targets = policy_targets(policy_name, aligned)
    result = run_target_weight_backtest(
        bars=bars,
        targets=targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    return PolicyRun(
        bars=bars,
        predictions=predictions,
        aligned_probabilities=aligned,
        targets=targets,
        result=result,
        row=comparison_row(policy_name, bars, targets, result),
    )


def all_runs() -> dict[str, PolicyRun]:
    return {policy_name: run_policy(policy_name) for policy_name in POLICY_NAMES}


def test_common_inputs_alignment_targets_and_input_integrity() -> None:
    runs = all_runs()
    expected_targets = {
        "binary": (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
        "three_state": (0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.0),
        "continuous": (0.0, 0.2, 0.5, 0.3, 0.6, 0.1, 0.1, 0.0),
    }
    original_bars = tuple(replace(bar) for bar in runs["binary"].bars)
    original_predictions = tuple(
        replace(item) for item in runs["binary"].predictions
    )

    for policy_name, run in runs.items():
        assert run.bars == runs["binary"].bars
        assert run.predictions == runs["binary"].predictions
        assert run.aligned_probabilities == PROBABILITIES
        assert target_values(run.targets) == pytest.approx(
            expected_targets[policy_name]
        )
    assert runs["binary"].bars == original_bars
    assert runs["binary"].predictions == original_predictions


def test_exact_comparison_rows_and_determinism() -> None:
    rows = {name: run.row for name, run in all_runs().items()}
    repeated_rows = {name: run_policy(name).row for name in POLICY_NAMES}
    assert rows == repeated_rows

    expected = {
        "binary": {
            "unique": (0.0, 1.0), "transitions": 2, "repeated": 5,
            "absolute_change": 2.0, "fills": 1, "buys": 1, "sells": 0,
            "turnover": 1.0, "exposure": 0.75, "full_entries": 1,
            "full_exits": 0, "partial_buys": 0, "partial_sells": 0,
            "no_trade": 6, "cash": 0.0, "position": 10.0,
        },
        "three_state": {
            "unique": (0.0, 0.5, 1.0), "transitions": 6, "repeated": 1,
            "absolute_change": 3.0, "fills": 5, "buys": 3, "sells": 2,
            "turnover": 2.5, "exposure": 0.5, "full_entries": 0,
            "full_exits": 0, "partial_buys": 1, "partial_sells": 2,
            "no_trade": 2, "cash": 500.0, "position": 5.0,
        },
        "continuous": {
            "unique": (0.0, 0.1, 0.2, 0.3, 0.5, 0.6),
            "transitions": 6, "repeated": 1, "absolute_change": 1.6,
            "fills": 5, "buys": 3, "sells": 2, "turnover": 1.5,
            "exposure": 0.225, "full_entries": 0, "full_exits": 0,
            "partial_buys": 3, "partial_sells": 2, "no_trade": 2,
            "cash": 900.0, "position": 1.0,
        },
    }
    for name, row in rows.items():
        values = expected[name]
        assert row.unique_target_weights == pytest.approx(values["unique"])
        assert row.target_transition_count == values["transitions"]
        assert row.repeated_adjacent_target_count == values["repeated"]
        assert row.total_absolute_target_change == pytest.approx(values["absolute_change"])
        assert row.fill_count == values["fills"]
        assert row.buy_count == values["buys"]
        assert row.sell_count == values["sells"]
        assert row.turnover == pytest.approx(values["turnover"])
        assert row.average_realized_exposure == pytest.approx(values["exposure"])
        assert row.total_fees == 0.0
        assert row.full_entry_count == values["full_entries"]
        assert row.full_exit_count == values["full_exits"]
        assert row.partial_buy_count == values["partial_buys"]
        assert row.partial_sell_count == values["partial_sells"]
        assert row.no_trade_target_count == values["no_trade"]
        assert row.final_target_unexecuted is True
        assert row.final_cash == pytest.approx(values["cash"])
        assert row.final_position_quantity == pytest.approx(values["position"])
        assert row.final_portfolio_value == pytest.approx(1000.0)
        assert row.target_transition_count != row.fill_count


def test_timing_reporting_repeated_targets_and_final_transition() -> None:
    runs = all_runs()
    repeated_indices = {"binary": 2, "three_state": 6, "continuous": 6}
    for name, run in runs.items():
        assert all(
            fill.executed_at == fill.order_created_at + timedelta(hours=1)
            for fill in run.result.fills
        )
        assert all(
            fill.executed_at != fill.order_created_at
            for fill in run.result.fills
        )
        assert len(build_trade_log(run.result)) == len(run.result.fills)
        assert summarize_backtest(run.result).trade_count == len(run.result.fills)
        assert summarize_backtest(run.result).turnover == pytest.approx(
            math.fsum(fill.notional for fill in run.result.fills) / 1000.0
        )
        exposures = tuple(
            exposure_at_snapshot(run.result, index)
            for index in range(len(run.bars))
        )
        assert summarize_backtest(run.result).average_exposure == pytest.approx(
            math.fsum(exposures) / len(exposures)
        )
        repeated_index = repeated_indices[name]
        assert run.targets[repeated_index] == run.targets[repeated_index - 1]
        assert all(
            fill.order_created_at != run.bars[repeated_index].timestamp
            for fill in run.result.fills
        )
        assert run.result.unexecuted_final_target is not None
        assert run.result.unexecuted_final_target.target == run.targets[-1]
        assert all(
            fill.order_created_at != run.bars[-1].timestamp
            for fill in run.result.fills
        )

    binary = runs["binary"]
    assert binary.targets[-2].weight == 1.0
    assert binary.targets[-1].weight == 0.0
    assert binary.row.target_transition_count == 2
    assert binary.row.fill_count == 1
    assert binary.row.full_entry_count == 1
    assert binary.row.full_exit_count == 0


def test_rebalance_classification_uses_actual_pre_and_post_exposure() -> None:
    rows = {name: run.row for name, run in all_runs().items()}
    assert rows["binary"].full_entry_count == 1
    assert rows["binary"].full_exit_count == 0
    assert rows["three_state"].partial_buy_count == 1
    assert rows["three_state"].partial_sell_count == 2
    assert rows["continuous"].partial_buy_count == 3
    assert rows["continuous"].partial_sell_count == 2


def test_no_policy_winner_or_ranking_logic_in_production_allocation() -> None:
    source = inspect.getsource(allocation_module).lower()
    for forbidden in ("winner", "ranking", "rank_policy", "select_policy"):
        assert forbidden not in source
