import inspect
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

import backtest.allocation as allocation_module
from backtest.allocation import AllocationError, probabilities_to_target_weights
from backtest.engine import run_target_weight_backtest
from backtest.events import Bar
from backtest.portfolio import PortfolioState
from backtest.positioning import TargetWeight
from backtest.reporting import build_trade_log, summarize_backtest

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def weights(values: tuple[TargetWeight, ...]) -> tuple[float, ...]:
    return tuple(value.weight for value in values)


def synthetic_bars() -> tuple[Bar, ...]:
    prices = ((100.0, 101.0), (102.0, 103.0), (104.0, 105.0),
              (106.0, 107.0), (108.0, 109.0), (110.0, 111.0))
    return tuple(
        Bar(
            timestamp=START + timedelta(hours=index),
            open=open_price,
            high=max(open_price, close_price) + 1.0,
            low=min(open_price, close_price) - 1.0,
            close=close_price,
            volume=10.0,
        )
        for index, (open_price, close_price) in enumerate(prices)
    )


def allocate(values: object, lower: object = 0.4, upper: object = 0.7):
    return probabilities_to_target_weights(
        values,  # type: ignore[arg-type]
        lower_threshold=lower,  # type: ignore[arg-type]
        upper_threshold=upper,  # type: ignore[arg-type]
    )


def test_public_api_signature_and_types() -> None:
    assert callable(probabilities_to_target_weights)
    assert issubclass(AllocationError, ValueError)
    signature = inspect.signature(probabilities_to_target_weights)
    assert signature.parameters["lower_threshold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["upper_threshold"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["lower_threshold"].default is inspect.Parameter.empty
    assert signature.parameters["upper_threshold"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        probabilities_to_target_weights([0.5], 0.4, 0.7)  # type: ignore[misc]
    with pytest.raises(TypeError):
        probabilities_to_target_weights([0.5])  # type: ignore[call-arg]


def test_output_contract_order_input_integrity_determinism_and_immutability() -> None:
    source = [0.9, 0.1, 0.5]
    original = source.copy()
    first = allocate(source)
    second = allocate(source)
    assert type(first) is tuple
    assert len(first) == len(source)
    assert all(type(value) is TargetWeight for value in first)
    assert weights(first) == (1.0, 0.0, 0.5)
    assert source == original
    assert first == second
    assert set(weights(first)) <= {0.0, 0.5, 1.0}
    with pytest.raises(TypeError):
        first[0] = TargetWeight(0.0)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first[0].weight = 0.0  # type: ignore[misc]


def test_main_boundary_hand_check() -> None:
    result = allocate([0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90])
    assert weights(result) == (0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0)
    assert result[2] == TargetWeight(0.5)
    assert result[5] == TargetWeight(0.5)


def test_equal_threshold_hand_check() -> None:
    assert weights(allocate([0.49, 0.50, 0.51], 0.50, 0.50)) == (0.0, 0.5, 1.0)


@pytest.mark.parametrize(
    ("probability", "lower", "upper", "expected"),
    [
        (0.0, 0.0, 1.0, 0.5),
        (1.0, 0.0, 1.0, 0.5),
        (0.0, 0.0, 0.0, 0.5),
        (1.0, 1.0, 1.0, 0.5),
        (math.nextafter(0.4, 0.0), 0.4, 0.7, 0.0),
        (0.4, 0.4, 0.7, 0.5),
        (math.nextafter(0.4, 1.0), 0.4, 0.7, 0.5),
        (math.nextafter(0.7, 0.0), 0.4, 0.7, 0.5),
        (0.7, 0.4, 0.7, 0.5),
        (math.nextafter(0.7, 1.0), 0.4, 0.7, 1.0),
    ],
)
def test_exact_mapping_has_no_hidden_epsilon(
    probability: float, lower: float, upper: float, expected: float
) -> None:
    assert allocate([probability], lower, upper) == (TargetWeight(expected),)


@pytest.mark.parametrize(
    "invalid",
    [None, 1, 0.5, object(), {"p": 0.5}, {0.5}, (x for x in [0.5]),
     "0.5", b"0.5", bytearray(b"0.5")],
)
def test_invalid_probability_containers_are_rejected(invalid: object) -> None:
    with pytest.raises(AllocationError, match="probabilities must be a non-string sequence"):
        allocate(invalid)


def test_empty_sequence_is_rejected() -> None:
    with pytest.raises(AllocationError, match="at least one"):
        allocate([])


@pytest.mark.parametrize("invalid", [True, False, "0.5", None])
def test_non_numeric_or_boolean_probability_is_rejected(invalid: object) -> None:
    with pytest.raises(AllocationError, match=r"probabilities\[1\].*numeric and not boolean"):
        allocate([0.5, invalid])


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_probability_is_rejected(invalid: float) -> None:
    with pytest.raises(AllocationError, match=r"probabilities\[1\].*finite"):
        allocate([0.5, invalid])


@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_out_of_range_probability_is_rejected(invalid: float) -> None:
    with pytest.raises(AllocationError, match=r"probabilities\[1\].*inclusive range"):
        allocate([0.5, invalid])


@pytest.mark.parametrize("field", ["lower", "upper"])
@pytest.mark.parametrize("invalid", [True, False, "0.5", None])
def test_non_numeric_or_boolean_threshold_is_rejected(field: str, invalid: object) -> None:
    with pytest.raises(AllocationError, match=f"{field}_threshold.*numeric and not boolean"):
        allocate([0.5], invalid if field == "lower" else 0.4,
                 invalid if field == "upper" else 0.7)


@pytest.mark.parametrize("field", ["lower", "upper"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_threshold_is_rejected(field: str, invalid: float) -> None:
    with pytest.raises(AllocationError, match=f"{field}_threshold.*finite"):
        allocate([0.5], invalid if field == "lower" else 0.4,
                 invalid if field == "upper" else 0.7)


@pytest.mark.parametrize("field", ["lower", "upper"])
@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_out_of_range_threshold_is_rejected(field: str, invalid: float) -> None:
    with pytest.raises(AllocationError, match=f"{field}_threshold.*inclusive range"):
        allocate([0.5], invalid if field == "lower" else 0.4,
                 invalid if field == "upper" else 0.7)


def test_reversed_thresholds_are_rejected_not_reordered() -> None:
    with pytest.raises(AllocationError, match="lower_threshold must be less than or equal"):
        allocate([0.5], 0.7, 0.4)


def test_allocation_source_respects_responsibility_boundaries() -> None:
    source = inspect.getsource(allocation_module).lower()
    for forbidden in (
        "bar", "timestamp", "model", "fit(", "score(", "validation", "test split",
        "run_backtest", "run_target_weight_backtest", "portfolio", "fee", "slippage",
        "rebalance", "notional", "order", "fill", "reporting",
    ):
        assert forbidden not in source


def run_integration(*, fee_rate: float, slippage_rate: float,
                    rebalance_tolerance: float, minimum_trade_notional: float):
    probabilities = (0.20, 0.80, 0.55, 0.35, 0.90, 0.45)
    targets = allocate(probabilities)
    result = run_target_weight_backtest(
        bars=synthetic_bars(), targets=targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=fee_rate, slippage_rate=slippage_rate,
        rebalance_tolerance=rebalance_tolerance,
        minimum_trade_notional=minimum_trade_notional,
    )
    return probabilities, targets, result


def test_zero_cost_end_to_end_timing_reporting_and_determinism() -> None:
    probabilities, targets, result = run_integration(
        fee_rate=0.0, slippage_rate=0.0,
        rebalance_tolerance=0.0, minimum_trade_notional=0.0,
    )
    _, repeated_targets, repeated = run_integration(
        fee_rate=0.0, slippage_rate=0.0,
        rebalance_tolerance=0.0, minimum_trade_notional=0.0,
    )
    bars = synthetic_bars()
    assert len(probabilities) == len(targets) == len(bars)
    assert weights(targets) == (0.0, 1.0, 0.5, 0.0, 1.0, 0.5)
    assert result == repeated and targets == repeated_targets
    assert all(fill.executed_at > fill.order_created_at for fill in result.fills)
    assert all(fill.executed_at == fill.order_created_at + timedelta(hours=1)
               for fill in result.fills)
    assert all(fill.executed_at != bars[0].timestamp for fill in result.fills)
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.decision_bar_timestamp == bars[-1].timestamp
    assert all(fill.order_created_at != bars[-1].timestamp for fill in result.fills)
    assert len(build_trade_log(result)) == len(result.fills)
    assert summarize_backtest(result).trade_count == len(result.fills)


def test_cost_controls_change_execution_not_allocation() -> None:
    _, zero_targets, _ = run_integration(
        fee_rate=0.0, slippage_rate=0.0,
        rebalance_tolerance=0.0, minimum_trade_notional=0.0,
    )
    _, cost_targets, cost_result = run_integration(
        fee_rate=0.01, slippage_rate=0.02,
        rebalance_tolerance=0.05, minimum_trade_notional=25.0,
    )
    _, repeated_targets, repeated_result = run_integration(
        fee_rate=0.01, slippage_rate=0.02,
        rebalance_tolerance=0.05, minimum_trade_notional=25.0,
    )
    assert cost_targets == zero_targets == repeated_targets
    assert cost_result == repeated_result
    assert len(build_trade_log(cost_result)) == len(cost_result.fills)
    summary = summarize_backtest(cost_result)
    assert summary.trade_count == len(cost_result.fills)
    assert summary.total_fees == pytest.approx(sum(fill.fee for fill in cost_result.fills))
    assert cost_targets == (TargetWeight(0.0), TargetWeight(1.0), TargetWeight(0.5),
                            TargetWeight(0.0), TargetWeight(1.0), TargetWeight(0.5))


def test_suppressed_orders_create_no_fills_fees_or_target_mutation() -> None:
    _, baseline_targets, _ = run_integration(
        fee_rate=0.0, slippage_rate=0.0,
        rebalance_tolerance=0.0, minimum_trade_notional=0.0,
    )
    _, suppressed_targets, suppressed_result = run_integration(
        fee_rate=0.01, slippage_rate=0.02,
        rebalance_tolerance=0.05, minimum_trade_notional=2000.0,
    )
    assert suppressed_targets == baseline_targets
    assert suppressed_result.fills == ()
    assert suppressed_result.final_state.cumulative_fees == 0.0
    assert build_trade_log(suppressed_result) == ()
    summary = summarize_backtest(suppressed_result)
    assert summary.trade_count == 0
    assert summary.total_fees == 0.0
