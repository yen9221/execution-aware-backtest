import inspect
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.allocation as allocation_module
from backtest.allocation import probabilities_to_continuous_target_weights
from backtest.engine import TargetWeightBacktestResult, run_target_weight_backtest
from backtest.events import Bar
from backtest.portfolio import PortfolioState
from backtest.positioning import TargetWeight
from backtest.prediction_alignment import (
    TimestampedProbability,
    align_probabilities_to_bars,
)
from backtest.reporting import build_trade_log, summarize_backtest

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROBABILITIES = (0.50, 0.60, 0.75, 0.65, 0.80, 0.55, 0.55, 0.45)
BASE_WEIGHTS = (0.0, 0.2, 0.5, 0.3, 0.6, 0.1, 0.1, 0.0)
PRICE_PATH = (
    (100.0, 101.0, 102.0, 99.0),
    (102.0, 103.0, 104.0, 101.0),
    (104.0, 105.0, 106.0, 103.0),
    (103.0, 102.0, 104.0, 101.0),
    (101.0, 100.0, 102.0, 99.0),
    (99.0, 100.0, 101.0, 98.0),
    (101.0, 102.0, 103.0, 100.0),
    (103.0, 104.0, 105.0, 102.0),
)
CAPS = {
    "max_weight_1_00": 1.00,
    "max_weight_0_50": 0.50,
    "max_weight_0_25": 0.25,
}
EXPECTED_CAPPED_WEIGHTS = {
    "max_weight_1_00": BASE_WEIGHTS,
    "max_weight_0_50": (0.0, 0.2, 0.5, 0.3, 0.5, 0.1, 0.1, 0.0),
    "max_weight_0_25": (0.0, 0.2, 0.25, 0.25, 0.25, 0.1, 0.1, 0.0),
}


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    scenario_name: str
    max_weight: float
    base_target_weights: tuple[float, ...]
    capped_target_weights: tuple[float, ...]
    capped_target_count: int
    target_transition_count: int
    total_absolute_target_change: float
    fill_count: int
    buy_count: int
    sell_count: int
    turnover: float
    average_realized_exposure: float
    maximum_realized_exposure: float
    total_fees: float
    initial_portfolio_value: float
    final_portfolio_value: float
    cumulative_return: float
    maximum_drawdown: float
    final_cash: float
    final_position_quantity: float
    gross_fill_notional: float
    cash_path: tuple[float, ...]
    position_path: tuple[float, ...]
    portfolio_value_path: tuple[float, ...]
    realized_exposure_path: tuple[float, ...]
    final_target_unexecuted: bool


@dataclass(frozen=True, slots=True)
class ScenarioCore:
    scenario_name: str
    max_weight: float
    bars: tuple[Bar, ...]
    predictions: tuple[TimestampedProbability, ...]
    aligned_probabilities: tuple[float, ...]
    base_targets: tuple[TargetWeight, ...]
    capped_targets: tuple[TargetWeight, ...]
    result: TargetWeightBacktestResult


def synthetic_bars() -> tuple[Bar, ...]:
    return tuple(
        Bar(
            timestamp=START + timedelta(hours=index),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=10.0,
        )
        for index, (open_price, close_price, high_price, low_price) in enumerate(
            PRICE_PATH
        )
    )


def timestamped_predictions(
    bars: tuple[Bar, ...],
) -> tuple[TimestampedProbability, ...]:
    return tuple(
        TimestampedProbability(bar.timestamp, probability)
        for bar, probability in zip(bars, PROBABILITIES, strict=True)
    )


def target_values(targets: tuple[TargetWeight, ...]) -> tuple[float, ...]:
    return tuple(target.weight for target in targets)


def cap_targets(
    base_targets: tuple[TargetWeight, ...], max_weight: float
) -> tuple[TargetWeight, ...]:
    return tuple(
        TargetWeight(min(target.weight, max_weight)) for target in base_targets
    )


def run_scenario(scenario_name: str) -> ScenarioCore:
    max_weight = CAPS[scenario_name]
    bars = synthetic_bars()
    predictions = timestamped_predictions(bars)
    aligned = align_probabilities_to_bars(bars, predictions)
    base_targets = probabilities_to_continuous_target_weights(aligned)
    capped_targets = cap_targets(base_targets, max_weight)
    result = run_target_weight_backtest(
        bars=bars,
        targets=capped_targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=0.001,
        slippage_rate=0.0005,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    return ScenarioCore(
        scenario_name=scenario_name,
        max_weight=max_weight,
        bars=bars,
        predictions=predictions,
        aligned_probabilities=aligned,
        base_targets=base_targets,
        capped_targets=capped_targets,
        result=result,
    )


def realized_exposure_path(core: ScenarioCore) -> tuple[float, ...]:
    return tuple(
        snapshot.position_quantity * snapshot.close_price / snapshot.portfolio_value
        for snapshot in core.result.portfolio_history
    )


def comparison_row(core: ScenarioCore) -> ComparisonRow:
    base_weights = target_values(core.base_targets)
    capped_weights = target_values(core.capped_targets)
    exposures = realized_exposure_path(core)
    summary = summarize_backtest(core.result)
    gross_notional = math.fsum(fill.notional for fill in core.result.fills)
    snapshots = core.result.portfolio_history
    return ComparisonRow(
        scenario_name=core.scenario_name,
        max_weight=core.max_weight,
        base_target_weights=base_weights,
        capped_target_weights=capped_weights,
        capped_target_count=sum(
            capped < base
            for base, capped in zip(base_weights, capped_weights, strict=True)
        ),
        target_transition_count=sum(
            left != right for left, right in zip(capped_weights, capped_weights[1:])
        ),
        total_absolute_target_change=math.fsum(
            abs(right - left)
            for left, right in zip(capped_weights, capped_weights[1:])
        ),
        fill_count=len(core.result.fills),
        buy_count=summary.buy_count,
        sell_count=summary.sell_count,
        turnover=summary.turnover,
        average_realized_exposure=summary.average_exposure,
        maximum_realized_exposure=max(exposures),
        total_fees=summary.total_fees,
        initial_portfolio_value=summary.initial_portfolio_value,
        final_portfolio_value=summary.final_portfolio_value,
        cumulative_return=summary.cumulative_return,
        maximum_drawdown=summary.max_drawdown,
        final_cash=core.result.final_state.cash,
        final_position_quantity=core.result.final_state.position_quantity,
        gross_fill_notional=gross_notional,
        cash_path=tuple(snapshot.cash for snapshot in snapshots),
        position_path=tuple(snapshot.position_quantity for snapshot in snapshots),
        portfolio_value_path=tuple(snapshot.portfolio_value for snapshot in snapshots),
        realized_exposure_path=exposures,
        final_target_unexecuted=core.result.unexecuted_final_target is not None,
    )


def all_cores() -> dict[str, ScenarioCore]:
    return {name: run_scenario(name) for name in CAPS}


def all_rows(cores: dict[str, ScenarioCore]) -> dict[str, ComparisonRow]:
    return {name: comparison_row(core) for name, core in cores.items()}


def test_common_inputs_exact_caps_targets_integrity_and_determinism() -> None:
    cores = all_cores()
    repeated = all_cores()
    reference = cores["max_weight_1_00"]
    original_bars = tuple(replace(bar) for bar in reference.bars)
    original_predictions = tuple(replace(item) for item in reference.predictions)

    assert CAPS == {
        "max_weight_1_00": 1.00,
        "max_weight_0_50": 0.50,
        "max_weight_0_25": 0.25,
    }
    for name, core in cores.items():
        base_weights = target_values(core.base_targets)
        capped_weights = target_values(core.capped_targets)
        assert core == repeated[name]
        assert core.bars == reference.bars
        assert core.predictions == reference.predictions
        assert core.aligned_probabilities == PROBABILITIES
        assert base_weights == pytest.approx(BASE_WEIGHTS)
        assert capped_weights == pytest.approx(EXPECTED_CAPPED_WEIGHTS[name])
        assert len(capped_weights) == len(base_weights)
        assert all(0.0 <= capped <= core.max_weight for capped in capped_weights)
        assert all(
            capped <= base
            for base, capped in zip(base_weights, capped_weights, strict=True)
        )
        assert capped_weights[-1] == 0.0
        assert core.result.initial_state == PortfolioState(cash=1000.0)

    assert target_values(reference.capped_targets) == target_values(
        reference.base_targets
    )
    assert [
        index
        for index, (base, capped) in enumerate(
            zip(
                target_values(cores["max_weight_0_50"].base_targets),
                target_values(cores["max_weight_0_50"].capped_targets),
                strict=True,
            )
        )
        if capped < base
    ] == [4]
    assert [
        index
        for index, (base, capped) in enumerate(
            zip(
                target_values(cores["max_weight_0_25"].base_targets),
                target_values(cores["max_weight_0_25"].capped_targets),
                strict=True,
            )
        )
        if capped < base
    ] == [2, 3, 4]
    assert reference.bars == original_bars
    assert reference.predictions == original_predictions


def test_execution_costs_timing_and_reporting_contracts() -> None:
    for core in all_cores().values():
        fills = core.result.fills
        assert all(fill.fee_rate == 0.001 for fill in fills)
        assert all(fill.slippage_rate == 0.0005 for fill in fills)
        assert all(
            fill.executed_at == fill.order_created_at + timedelta(hours=1)
            for fill in fills
        )
        assert all(fill.executed_at != fill.order_created_at for fill in fills)
        assert all(fill.executed_at != core.bars[0].timestamp for fill in fills)
        assert core.result.unexecuted_final_target is not None
        assert core.result.unexecuted_final_target.target == core.capped_targets[-1]
        assert len(build_trade_log(core.result)) == len(fills)
        summary = summarize_backtest(core.result)
        assert summary.trade_count == len(fills)
        assert summary.total_fees == pytest.approx(
            math.fsum(fill.fee for fill in fills)
        )


def test_rows_use_actual_targets_fills_snapshots_and_portfolio_history() -> None:
    cores = all_cores()
    rows = all_rows(cores)
    assert rows == all_rows(all_cores())

    expected = {
        "max_weight_1_00": {
            "capped": 0, "transitions": 6, "absolute_change": 1.6,
            "fills": 6, "buys": 3, "sells": 3,
            "turnover": 1.495575139014044,
            "average_exposure": 0.22523851150610966,
            "maximum_exposure": 0.6026829838539742,
            "fees": 1.4955751390140442,
            "final_value": 995.1074131527553,
            "return": -0.004892586847244695,
            "drawdown": -0.017179640257491413,
            "cash": 894.727724833263,
            "position": 0.9651893107643487,
            "gross_notional": 1495.575139014044,
        },
        "max_weight_0_50": {
            "capped": 1, "transitions": 6, "absolute_change": 1.4,
            "fills": 6, "buys": 3, "sells": 3,
            "turnover": 1.2974621747074377,
            "average_exposure": 0.21273446425725845,
            "maximum_exposure": 0.5026657287037446,
            "fees": 1.2974621747074377,
            "final_value": 993.4165166110478,
            "return": -0.006583483388952205,
            "drawdown": -0.017179640257491413,
            "cash": 893.2073967568216,
            "position": 0.9635492293675594,
            "gross_notional": 1297.4621747074377,
        },
        "max_weight_0_25": {
            "capped": 3, "transitions": 4, "absolute_change": 0.5,
            "fills": 6, "buys": 4, "sells": 2,
            "turnover": 0.41319175018585586,
            "average_exposure": 0.1439509811039993,
            "maximum_exposure": 0.25189057500903994,
            "fees": 0.41319175018585586,
            "final_value": 995.6198928199456,
            "return": -0.004380107180054438,
            "drawdown": -0.01121159780365344,
            "cash": 895.1885170847843,
            "position": 0.9656863051457814,
            "gross_notional": 413.1917501858559,
        },
    }

    for name, row in rows.items():
        core = cores[name]
        capped = row.capped_target_weights
        values = expected[name]
        assert row.capped_target_count == values["capped"]
        assert row.target_transition_count == values["transitions"]
        assert row.total_absolute_target_change == pytest.approx(
            values["absolute_change"]
        )
        assert row.fill_count == values["fills"]
        assert row.buy_count == values["buys"]
        assert row.sell_count == values["sells"]
        assert row.turnover == pytest.approx(values["turnover"])
        assert row.average_realized_exposure == pytest.approx(
            values["average_exposure"]
        )
        assert row.maximum_realized_exposure == pytest.approx(
            values["maximum_exposure"]
        )
        assert row.total_fees == pytest.approx(values["fees"])
        assert row.final_portfolio_value == pytest.approx(values["final_value"])
        assert row.cumulative_return == pytest.approx(values["return"])
        assert row.maximum_drawdown == pytest.approx(values["drawdown"])
        assert row.final_cash == pytest.approx(values["cash"])
        assert row.final_position_quantity == pytest.approx(values["position"])
        assert row.gross_fill_notional == pytest.approx(values["gross_notional"])
        assert row.capped_target_count == sum(
            actual < base
            for base, actual in zip(
                row.base_target_weights, capped, strict=True
            )
        )
        assert row.target_transition_count == sum(
            left != right for left, right in zip(capped, capped[1:])
        )
        assert row.total_absolute_target_change == pytest.approx(
            math.fsum(abs(right - left) for left, right in zip(capped, capped[1:]))
        )
        assert row.gross_fill_notional == pytest.approx(
            math.fsum(fill.notional for fill in core.result.fills)
        )
        assert row.turnover == pytest.approx(
            row.gross_fill_notional / row.initial_portfolio_value
        )
        assert row.average_realized_exposure == pytest.approx(
            math.fsum(row.realized_exposure_path) / len(row.realized_exposure_path)
        )
        assert row.maximum_realized_exposure == pytest.approx(
            max(row.realized_exposure_path)
        )
        peak = 0.0
        drawdowns = []
        for value in row.portfolio_value_path:
            peak = max(peak, value)
            drawdowns.append(value / peak - 1.0)
        assert row.maximum_drawdown == pytest.approx(min(drawdowns))
        assert row.cumulative_return == pytest.approx(
            row.final_portfolio_value / row.initial_portfolio_value - 1.0
        )


def test_lower_caps_reduce_realized_exposure_and_change_bound_paths() -> None:
    rows = all_rows(all_cores())
    uncapped = rows["max_weight_1_00"]
    moderate = rows["max_weight_0_50"]
    lower = rows["max_weight_0_25"]

    assert uncapped.average_realized_exposure >= moderate.average_realized_exposure
    assert moderate.average_realized_exposure >= lower.average_realized_exposure
    assert uncapped.maximum_realized_exposure >= moderate.maximum_realized_exposure
    assert moderate.maximum_realized_exposure >= lower.maximum_realized_exposure
    assert uncapped.realized_exposure_path != uncapped.capped_target_weights
    for capped in (moderate, lower):
        assert (
            capped.cash_path != uncapped.cash_path
            or capped.position_path != uncapped.position_path
            or capped.portfolio_value_path != uncapped.portfolio_value_path
            or capped.realized_exposure_path != uncapped.realized_exposure_path
        )


def test_no_production_cap_optimization_ranking_or_selection_logic() -> None:
    source = inspect.getsource(allocation_module).lower()
    for forbidden in (
        "max_weight",
        "optimize_cap",
        "rank_cap",
        "select_cap",
        "best_cap",
        "cap_winner",
    ):
        assert forbidden not in source
