import inspect
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.allocation as allocation_module
from backtest.allocation import probabilities_to_continuous_target_weights
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
EXPECTED_WEIGHTS = (0.0, 0.2, 0.5, 0.3, 0.6, 0.1, 0.1, 0.0)
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
SCENARIOS = {
    "zero_cost": (0.0, 0.0),
    "fee_only": (0.001, 0.0),
    "slippage_only": (0.0, 0.0005),
    "fee_and_slippage": (0.001, 0.0005),
}


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    scenario_name: str
    fee_rate: float
    slippage_rate: float
    target_weights: tuple[float, ...]
    fill_count: int
    buy_count: int
    sell_count: int
    turnover: float
    average_realized_exposure: float
    total_fees: float
    initial_portfolio_value: float
    final_portfolio_value: float
    cumulative_return: float
    maximum_drawdown: float
    final_cash: float
    final_position_quantity: float
    gross_fill_notional: float
    scenario_return_difference_vs_zero_cost: float
    final_value_difference_vs_zero_cost: float
    fill_quantities: tuple[float, ...]
    fill_prices: tuple[float, ...]
    cash_path: tuple[float, ...]
    position_path: tuple[float, ...]
    portfolio_value_path: tuple[float, ...]
    realized_exposure_path: tuple[float, ...]
    final_target_unexecuted: bool


@dataclass(frozen=True, slots=True)
class ScenarioCore:
    scenario_name: str
    fee_rate: float
    slippage_rate: float
    bars: tuple[Bar, ...]
    predictions: tuple[TimestampedProbability, ...]
    aligned_probabilities: tuple[float, ...]
    targets: tuple[TargetWeight, ...]
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


def run_core(scenario_name: str) -> ScenarioCore:
    fee_rate, slippage_rate = SCENARIOS[scenario_name]
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
    return ScenarioCore(
        scenario_name=scenario_name,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        bars=bars,
        predictions=predictions,
        aligned_probabilities=aligned,
        targets=targets,
        result=result,
    )


def target_values(core: ScenarioCore) -> tuple[float, ...]:
    return tuple(target.weight for target in core.targets)


def realized_exposure_path(core: ScenarioCore) -> tuple[float, ...]:
    return tuple(
        snapshot.position_quantity * snapshot.close_price / snapshot.portfolio_value
        for snapshot in core.result.portfolio_history
    )


def build_row(core: ScenarioCore, zero_core: ScenarioCore) -> ScenarioRow:
    summary = summarize_backtest(core.result)
    zero_summary = summarize_backtest(zero_core.result)
    snapshots = core.result.portfolio_history
    gross_notional = math.fsum(fill.notional for fill in core.result.fills)
    return ScenarioRow(
        scenario_name=core.scenario_name,
        fee_rate=core.fee_rate,
        slippage_rate=core.slippage_rate,
        target_weights=target_values(core),
        fill_count=len(core.result.fills),
        buy_count=summary.buy_count,
        sell_count=summary.sell_count,
        turnover=summary.turnover,
        average_realized_exposure=summary.average_exposure,
        total_fees=summary.total_fees,
        initial_portfolio_value=summary.initial_portfolio_value,
        final_portfolio_value=summary.final_portfolio_value,
        cumulative_return=summary.cumulative_return,
        maximum_drawdown=summary.max_drawdown,
        final_cash=core.result.final_state.cash,
        final_position_quantity=core.result.final_state.position_quantity,
        gross_fill_notional=gross_notional,
        scenario_return_difference_vs_zero_cost=(
            zero_summary.cumulative_return - summary.cumulative_return
        ),
        final_value_difference_vs_zero_cost=(
            zero_summary.final_portfolio_value - summary.final_portfolio_value
        ),
        fill_quantities=tuple(fill.quantity for fill in core.result.fills),
        fill_prices=tuple(fill.fill_price for fill in core.result.fills),
        cash_path=tuple(snapshot.cash for snapshot in snapshots),
        position_path=tuple(snapshot.position_quantity for snapshot in snapshots),
        portfolio_value_path=tuple(snapshot.portfolio_value for snapshot in snapshots),
        realized_exposure_path=realized_exposure_path(core),
        final_target_unexecuted=core.result.unexecuted_final_target is not None,
    )


def all_cores() -> dict[str, ScenarioCore]:
    return {name: run_core(name) for name in SCENARIOS}


def all_rows(cores: dict[str, ScenarioCore]) -> dict[str, ScenarioRow]:
    zero_core = cores["zero_cost"]
    return {name: build_row(core, zero_core) for name, core in cores.items()}


def test_common_inputs_settings_targets_integrity_and_determinism() -> None:
    cores = all_cores()
    repeated = all_cores()
    zero = cores["zero_cost"]
    original_bars = tuple(replace(bar) for bar in zero.bars)
    original_predictions = tuple(replace(item) for item in zero.predictions)

    assert tuple(SCENARIOS) == (
        "zero_cost", "fee_only", "slippage_only", "fee_and_slippage"
    )
    assert SCENARIOS == {
        "zero_cost": (0.0, 0.0),
        "fee_only": (0.001, 0.0),
        "slippage_only": (0.0, 0.0005),
        "fee_and_slippage": (0.001, 0.0005),
    }
    for name, core in cores.items():
        assert core == repeated[name]
        assert core.bars == zero.bars
        assert core.predictions == zero.predictions
        assert core.aligned_probabilities == PROBABILITIES
        assert target_values(core) == pytest.approx(EXPECTED_WEIGHTS)
        assert core.result.initial_state == PortfolioState(cash=1000.0)
    assert zero.bars == original_bars
    assert zero.predictions == original_predictions


def test_fill_prices_fees_timing_and_reporting_contracts() -> None:
    cores = all_cores()
    for name, core in cores.items():
        fills = core.result.fills
        assert all(
            fill.executed_at == fill.order_created_at + timedelta(hours=1)
            for fill in fills
        )
        assert all(fill.executed_at != fill.order_created_at for fill in fills)
        assert core.result.unexecuted_final_target is not None
        assert core.result.unexecuted_final_target.target == core.targets[-1]
        assert len(build_trade_log(core.result)) == len(fills)
        assert summarize_backtest(core.result).trade_count == len(fills)
        assert summarize_backtest(core.result).total_fees == pytest.approx(
            math.fsum(fill.fee for fill in fills)
        )

        if name in ("zero_cost", "fee_only"):
            assert all(fill.fill_price == fill.reference_price for fill in fills)
        else:
            assert all(
                fill.fill_price > fill.reference_price
                for fill in fills if fill.side is Side.BUY
            )
            assert all(
                fill.fill_price < fill.reference_price
                for fill in fills if fill.side is Side.SELL
            )
        if name in ("zero_cost", "slippage_only"):
            assert all(fill.fee == 0.0 for fill in fills)
        else:
            assert all(fill.fee > 0.0 for fill in fills)


def test_costs_change_quantities_and_realized_portfolio_paths_not_targets() -> None:
    cores = all_cores()
    rows = all_rows(cores)
    zero = rows["zero_cost"]
    for name, row in rows.items():
        assert row.target_weights == pytest.approx(zero.target_weights)
        if name != "zero_cost":
            assert (
                row.cash_path != zero.cash_path
                or row.position_path != zero.position_path
                or row.portfolio_value_path != zero.portfolio_value_path
                or row.realized_exposure_path != zero.realized_exposure_path
            )

    zero_buys = tuple(
        fill.quantity for fill in cores["zero_cost"].result.fills
        if fill.side is Side.BUY
    )
    for name in ("fee_only", "slippage_only", "fee_and_slippage"):
        scenario_buys = tuple(
            fill.quantity for fill in cores[name].result.fills
            if fill.side is Side.BUY
        )
        assert any(
            actual != pytest.approx(reference)
            for actual, reference in zip(scenario_buys, zero_buys, strict=True)
        )


def test_rows_use_actual_fills_snapshots_and_descriptive_differences() -> None:
    cores = all_cores()
    rows = all_rows(cores)
    zero = rows["zero_cost"]
    assert rows == all_rows(all_cores())

    for name, row in rows.items():
        core = cores[name]
        summary = summarize_backtest(core.result)
        assert row.fill_count == len(core.result.fills)
        assert row.gross_fill_notional == pytest.approx(
            math.fsum(fill.notional for fill in core.result.fills)
        )
        assert row.turnover == pytest.approx(
            row.gross_fill_notional / row.initial_portfolio_value
        )
        assert row.average_realized_exposure == pytest.approx(
            math.fsum(row.realized_exposure_path) / len(row.realized_exposure_path)
        )
        peak = 0.0
        drawdowns = []
        for value in row.portfolio_value_path:
            peak = max(peak, value)
            drawdowns.append(value / peak - 1.0)
        assert row.maximum_drawdown == pytest.approx(min(drawdowns))
        assert row.scenario_return_difference_vs_zero_cost == pytest.approx(
            zero.cumulative_return - summary.cumulative_return
        )
        assert row.final_value_difference_vs_zero_cost == pytest.approx(
            zero.final_portfolio_value - summary.final_portfolio_value
        )


def test_no_production_scenario_ranking_or_selection_logic() -> None:
    source = inspect.getsource(allocation_module).lower()
    for forbidden in (
        "scenario_winner", "rank_scenario", "select_scenario", "best_scenario"
    ):
        assert forbidden not in source
