import inspect
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.strategy as strategy_module
from backtest.engine import run_backtest
from backtest.events import Bar
from backtest.orders import Side
from backtest.portfolio import PortfolioState
from backtest.positioning import (
    TargetPosition,
    TargetWeight,
    target_position_to_weight,
    target_weight_to_position,
)
from backtest.reporting import build_trade_log, summarize_backtest
from backtest.strategy import (
    StrategyError,
    previous_close_momentum_targets,
    previous_close_momentum_target_weights,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
EXPECTED_TARGETS = (
    TargetPosition.CASH,
    TargetPosition.LONG,
    TargetPosition.CASH,
    TargetPosition.LONG,
    TargetPosition.CASH,
    TargetPosition.CASH,
)


def bar(
    index: int = 0,
    *,
    timestamp: datetime | None = None,
    open_price: float = 100.0,
    close_price: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=timestamp or START + timedelta(hours=index),
        open=open_price,
        high=max(open_price, close_price) + 1.0,
        low=min(open_price, close_price) - 1.0,
        close=close_price,
        volume=10.0,
    )


def bars_from_closes(closes: list[float]) -> list[Bar]:
    return [bar(index, close_price=close) for index, close in enumerate(closes)]


def integration_bars() -> list[Bar]:
    return [
        bar(0, open_price=100.0, close_price=100.0),
        bar(1, open_price=101.0, close_price=102.0),
        bar(2, open_price=103.0, close_price=101.0),
        bar(3, open_price=100.0, close_price=103.0),
        bar(4, open_price=104.0, close_price=103.0),
        bar(5, open_price=102.0, close_price=99.0),
    ]


def run_pipeline(source_bars: list[Bar]):
    targets = previous_close_momentum_targets(source_bars)
    result = run_backtest(
        bars=source_bars,
        targets=targets,
        initial_state=PortfolioState(cash=10_000.0),
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    return targets, result, build_trade_log(result), summarize_backtest(result)


@pytest.mark.parametrize("invalid", [None, 1, object(), {"bar": bar()}])
def test_non_sequence_bars_are_rejected(invalid: object) -> None:
    with pytest.raises(StrategyError, match="non-string sequence"):
        previous_close_momentum_targets(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["bars", b"bars", bytearray(b"bars")])
def test_string_like_bars_are_rejected(invalid: object) -> None:
    with pytest.raises(StrategyError, match="non-string sequence"):
        previous_close_momentum_targets(invalid)  # type: ignore[arg-type]


def test_empty_bars_are_rejected() -> None:
    with pytest.raises(StrategyError, match="at least one"):
        previous_close_momentum_targets([])


def test_invalid_bar_object_is_rejected() -> None:
    with pytest.raises(StrategyError, match=r"bars\[0\] must be a Bar"):
        previous_close_momentum_targets([object()])  # type: ignore[list-item]


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(StrategyError, match="timezone"):
        previous_close_momentum_targets(
            [bar(timestamp=datetime(2024, 1, 1))]
        )


def test_duplicate_timestamp_is_rejected() -> None:
    with pytest.raises(StrategyError, match="duplicates"):
        previous_close_momentum_targets(
            [bar(timestamp=START), bar(timestamp=START)]
        )


def test_descending_timestamp_is_rejected() -> None:
    with pytest.raises(StrategyError, match="not strictly later"):
        previous_close_momentum_targets(
            [bar(timestamp=START), bar(timestamp=START - timedelta(hours=1))]
        )


@pytest.mark.parametrize("delta", [timedelta(hours=2), timedelta(minutes=30)])
def test_non_hourly_interval_is_rejected(delta: timedelta) -> None:
    with pytest.raises(StrategyError, match="not exactly one hour"):
        previous_close_momentum_targets(
            [bar(timestamp=START), bar(timestamp=START + delta)]
        )


def test_equivalent_timezone_instants_are_compared_in_utc() -> None:
    first = datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    second = datetime(2023, 12, 31, 20, tzinfo=timezone(timedelta(hours=-5)))
    targets = previous_close_momentum_targets(
        [
            bar(timestamp=first, close_price=100.0),
            bar(timestamp=second, close_price=101.0),
        ]
    )
    assert targets == (TargetPosition.CASH, TargetPosition.LONG)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (True, "not boolean"),
        ("100", "numeric"),
        (0.0, "strictly positive"),
        (-1.0, "strictly positive"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
    ],
)
def test_invalid_close_is_rejected(invalid: object, message: str) -> None:
    invalid_bar = replace(bar(), close=invalid)  # type: ignore[arg-type]
    with pytest.raises(StrategyError, match=message):
        previous_close_momentum_targets([invalid_bar])


def test_single_bar_returns_cash_tuple() -> None:
    targets = previous_close_momentum_targets([bar()])
    assert targets == (TargetPosition.CASH,)
    assert isinstance(targets, tuple)


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        ([100.0, 101.0], (TargetPosition.CASH, TargetPosition.LONG)),
        ([100.0, 99.0], (TargetPosition.CASH, TargetPosition.CASH)),
        ([100.0, 100.0], (TargetPosition.CASH, TargetPosition.CASH)),
        (
            [100.0, 101.0, 99.0, 102.0, 98.0],
            (
                TargetPosition.CASH,
                TargetPosition.LONG,
                TargetPosition.CASH,
                TargetPosition.LONG,
                TargetPosition.CASH,
            ),
        ),
    ],
)
def test_close_comparisons_map_to_expected_targets(
    closes: list[float], expected: tuple[TargetPosition, ...]
) -> None:
    targets = previous_close_momentum_targets(bars_from_closes(closes))
    assert targets == expected
    assert len(targets) == len(closes)
    assert targets[0] is TargetPosition.CASH


def test_exact_six_close_hand_check() -> None:
    assert previous_close_momentum_targets(
        bars_from_closes([100.0, 102.0, 101.0, 103.0, 103.0, 99.0])
    ) == EXPECTED_TARGETS


def test_exact_six_close_weight_hand_check() -> None:
    weights = previous_close_momentum_target_weights(
        bars_from_closes([100.0, 102.0, 101.0, 103.0, 103.0, 99.0])
    )
    assert weights == tuple(
        TargetWeight(weight) for weight in (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    )
    assert isinstance(weights, tuple)
    assert all(type(weight) is TargetWeight for weight in weights)


@pytest.mark.parametrize(
    ("closes", "expected_weights"),
    [
        ([100.0], (0.0,)),
        ([100.0, 101.0], (0.0, 1.0)),
        ([100.0, 100.0], (0.0, 0.0)),
        ([100.0, 99.0], (0.0, 0.0)),
    ],
)
def test_weight_strategy_endpoint_behavior(
    closes: list[float], expected_weights: tuple[float, ...]
) -> None:
    source_bars = bars_from_closes(closes)
    original = list(source_bars)
    first = previous_close_momentum_target_weights(source_bars)
    second = previous_close_momentum_target_weights(source_bars)
    assert tuple(target.weight for target in first) == expected_weights
    assert len(first) == len(source_bars)
    assert first == second
    assert source_bars == original


def test_weight_targets_exactly_map_existing_binary_targets() -> None:
    source_bars = integration_bars()
    binary = previous_close_momentum_targets(source_bars)
    weights = previous_close_momentum_target_weights(source_bars)
    assert weights == tuple(target_position_to_weight(target) for target in binary)
    assert tuple(target_weight_to_position(target) for target in weights) == binary


def test_weight_strategy_has_no_fractional_rule() -> None:
    weights = previous_close_momentum_target_weights(
        bars_from_closes([100.0, 102.0, 101.0, 103.0])
    )
    assert {target.weight for target in weights} <= {0.0, 1.0}


def test_changing_future_bars_does_not_change_earlier_weight_targets() -> None:
    original = bars_from_closes([100.0, 102.0, 101.0, 103.0])
    changed = bars_from_closes([100.0, 102.0, 1.0, 1_000_000.0])
    assert previous_close_momentum_target_weights(original)[:2] == (
        previous_close_momentum_target_weights(changed)[:2]
    )


def test_binary_pipeline_is_identical_after_endpoint_round_trip() -> None:
    source_bars = integration_bars()
    binary_targets = previous_close_momentum_targets(source_bars)
    weight_targets = previous_close_momentum_target_weights(source_bars)
    round_trip_targets = tuple(
        target_weight_to_position(target) for target in weight_targets
    )
    original_pipeline = run_pipeline(source_bars)
    round_trip_result = run_backtest(
        bars=source_bars,
        targets=round_trip_targets,
        initial_state=PortfolioState(cash=10_000.0),
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    assert round_trip_targets == binary_targets
    assert round_trip_result == original_pipeline[1]
    assert build_trade_log(round_trip_result) == original_pipeline[2]
    assert summarize_backtest(round_trip_result) == original_pipeline[3]


def test_outputs_are_target_position_members_not_raw_integers() -> None:
    targets = previous_close_momentum_targets(bars_from_closes([100.0, 101.0, 99.0]))
    assert all(type(target) is TargetPosition for target in targets)


def test_output_length_equals_bar_count() -> None:
    source_bars = bars_from_closes([100.0, 101.0, 102.0, 99.0])
    assert len(previous_close_momentum_targets(source_bars)) == len(source_bars)


def test_first_target_is_cash_for_rising_or_falling_following_bars() -> None:
    for closes in ([100.0, 1_000.0], [100.0, 1.0]):
        assert previous_close_momentum_targets(bars_from_closes(closes))[0] is (
            TargetPosition.CASH
        )


def test_inputs_remain_unchanged_and_calls_are_deterministic() -> None:
    source_bars = bars_from_closes([100.0, 101.0, 99.0])
    original = list(source_bars)
    first = previous_close_momentum_targets(source_bars)
    second = previous_close_momentum_targets(source_bars)
    assert first == second
    assert first is not second
    assert source_bars == original


def test_changing_future_bars_does_not_change_earlier_targets() -> None:
    original = bars_from_closes([100.0, 102.0, 101.0, 103.0])
    changed = bars_from_closes([100.0, 102.0, 1.0, 1_000_000.0])
    assert previous_close_momentum_targets(original)[:2] == (
        previous_close_momentum_targets(changed)[:2]
    )


@pytest.mark.parametrize("prefix_length", [1, 2, 3, 4, 5])
def test_target_at_index_depends_only_on_closes_through_that_index(
    prefix_length: int,
) -> None:
    closes = [100.0, 102.0, 101.0, 103.0, 99.0]
    full_targets = previous_close_momentum_targets(bars_from_closes(closes))
    prefix_targets = previous_close_momentum_targets(
        bars_from_closes(closes[:prefix_length])
    )
    assert full_targets[:prefix_length] == prefix_targets


def test_strategy_source_has_no_execution_or_next_bar_logic() -> None:
    source = inspect.getsource(strategy_module.previous_close_momentum_targets)
    forbidden = (
        "bars[index + 1]",
        ".open",
        "run_backtest",
        "execute_market_order",
        "apply_fill",
        "summarize_backtest",
        "build_trade_log",
    )
    assert all(item not in source for item in forbidden)


def test_six_bar_pipeline_fill_alignment_and_reporting() -> None:
    source_bars = integration_bars()
    targets, result, trade_log, summary = run_pipeline(source_bars)
    assert targets == EXPECTED_TARGETS
    assert [item.side for item in result.fills] == [
        Side.BUY,
        Side.SELL,
        Side.BUY,
        Side.SELL,
    ]
    assert [item.executed_at for item in result.fills] == [
        source_bars[index].timestamp for index in (2, 3, 4, 5)
    ]
    assert [item.reference_price for item in result.fills] == [
        source_bars[index].open for index in (2, 3, 4, 5)
    ]
    assert [item.order_created_at for item in result.fills] == [
        source_bars[index].timestamp for index in (1, 2, 3, 4)
    ]
    assert len(result.portfolio_history) == len(source_bars)
    assert len(trade_log) == len(result.fills) == summary.trade_count == 4
    assert summary.total_fees == pytest.approx(
        result.final_state.cumulative_fees - result.initial_state.cumulative_fees
    )
    final_snapshot = result.portfolio_history[-1]
    assert final_snapshot.cash == result.final_state.cash
    assert final_snapshot.position_quantity == result.final_state.position_quantity
    assert final_snapshot.cumulative_fees == result.final_state.cumulative_fees
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is TargetPosition.CASH
    assert result.unexecuted_final_target.decision_bar_timestamp == source_bars[-1].timestamp


def test_strategy_output_passes_directly_to_engine() -> None:
    source_bars = integration_bars()
    targets = previous_close_momentum_targets(source_bars)
    result = run_backtest(
        bars=source_bars,
        targets=targets,
        initial_state=PortfolioState(cash=10_000.0),
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    assert len(result.portfolio_history) == len(source_bars)


def test_integration_produces_exactly_four_fills() -> None:
    _, result, _, _ = run_pipeline(integration_bars())
    assert len(result.fills) == 4


def test_integration_final_cash_target_is_unexecuted() -> None:
    _, result, _, _ = run_pipeline(integration_bars())
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.target is TargetPosition.CASH
    assert result.unexecuted_final_target.decision_bar_timestamp == START + timedelta(hours=5)


def test_integration_reporting_counts_match_fills() -> None:
    _, result, trade_log, summary = run_pipeline(integration_bars())
    assert len(trade_log) == len(result.fills) == summary.trade_count == 4


def test_integration_reporting_fees_match_engine_change() -> None:
    _, result, _, summary = run_pipeline(integration_bars())
    assert summary.total_fees == pytest.approx(
        result.final_state.cumulative_fees - result.initial_state.cumulative_fees
    )


def test_pipeline_does_not_mutate_inputs_or_result() -> None:
    source_bars = integration_bars()
    original_bars = list(source_bars)
    targets = previous_close_momentum_targets(source_bars)
    original_targets = tuple(targets)
    result = run_backtest(
        bars=source_bars,
        targets=targets,
        initial_state=PortfolioState(cash=10_000.0),
        fee_rate=FEE_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )
    original_result = replace(result)
    build_trade_log(result)
    summarize_backtest(result)
    assert source_bars == original_bars
    assert targets == original_targets
    assert result == original_result


def test_repeated_full_pipeline_runs_are_deterministic() -> None:
    source_bars = integration_bars()
    first = run_pipeline(source_bars)
    second = run_pipeline(source_bars)
    assert first == second
