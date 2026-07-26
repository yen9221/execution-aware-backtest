import inspect
import math
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

import backtest.prediction_alignment as alignment_module
from backtest.allocation import probabilities_to_target_weights
from backtest.engine import run_target_weight_backtest
from backtest.events import Bar
from backtest.portfolio import PortfolioState
from backtest.prediction_alignment import (
    PredictionAlignmentError,
    TimestampedProbability,
    align_probabilities_to_bars,
)
from backtest.reporting import build_trade_log, summarize_backtest

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROBABILITIES = (0.20, 0.80, 0.55, 0.35, 0.90, 0.45)


def bar(index: int, *, timestamp: datetime | None = None) -> Bar:
    open_price = 100.0 + 2.0 * index
    close_price = open_price + 1.0
    return Bar(
        timestamp=timestamp or START + timedelta(hours=index),
        open=open_price,
        high=close_price + 1.0,
        low=open_price - 1.0,
        close=close_price,
        volume=10.0,
    )


def bars(count: int = 6) -> tuple[Bar, ...]:
    return tuple(bar(index) for index in range(count))


def predictions(
    source_bars: tuple[Bar, ...] | None = None,
    probabilities: tuple[float, ...] = PROBABILITIES,
) -> tuple[TimestampedProbability, ...]:
    source = source_bars or bars(len(probabilities))
    return tuple(
        TimestampedProbability(item.timestamp, probability)
        for item, probability in zip(source, probabilities, strict=True)
    )


def align(source_bars: object, source_predictions: object):
    return align_probabilities_to_bars(
        source_bars,  # type: ignore[arg-type]
        source_predictions,  # type: ignore[arg-type]
    )


def test_public_api_and_immutable_slotted_record() -> None:
    assert callable(align_probabilities_to_bars)
    assert issubclass(PredictionAlignmentError, ValueError)
    assert [item.name for item in fields(TimestampedProbability)] == [
        "timestamp", "probability"
    ]
    record = TimestampedProbability(START, 0.5)
    assert hasattr(TimestampedProbability, "__slots__")
    assert not hasattr(record, "__dict__")
    with pytest.raises(FrozenInstanceError):
        record.probability = 0.2  # type: ignore[misc]


def test_output_inputs_and_determinism_contract() -> None:
    source_bars = bars()
    source_predictions = predictions(source_bars)
    original_bars = tuple(replace(item) for item in source_bars)
    original_predictions = tuple(replace(item) for item in source_predictions)
    first = align(source_bars, source_predictions)
    second = align(source_bars, source_predictions)
    assert type(first) is tuple
    assert first == PROBABILITIES == second
    assert len(first) == len(source_bars)
    assert all(type(value) is float for value in first)
    assert source_bars == original_bars
    assert source_predictions == original_predictions


@pytest.mark.parametrize("invalid", [None, 1, object(), {"bar": bar(0)}, {bar(0)}, (bar(0) for _ in range(1)), "bar", b"bar", bytearray(b"bar")])
def test_invalid_bar_containers_are_rejected(invalid: object) -> None:
    with pytest.raises(PredictionAlignmentError, match="bars must be a non-string sequence"):
        align(invalid, predictions(bars(1), (0.5,)))


@pytest.mark.parametrize("invalid", [None, 1, object(), {"p": 0.5}, {TimestampedProbability(START, 0.5)}, (TimestampedProbability(START, 0.5) for _ in range(1)), "prediction", b"prediction", bytearray(b"prediction")])
def test_invalid_prediction_containers_are_rejected(invalid: object) -> None:
    with pytest.raises(PredictionAlignmentError, match="predictions must be a non-string sequence"):
        align(bars(1), invalid)


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(PredictionAlignmentError, match="bars must contain at least one"):
        align([], [])
    with pytest.raises(PredictionAlignmentError, match="predictions must contain at least one"):
        align(bars(1), [])


@pytest.mark.parametrize("invalid", [object(), (START, 0.5), {"timestamp": START, "probability": 0.5}])
def test_invalid_elements_and_raw_prediction_structures_are_rejected(invalid: object) -> None:
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].*exactly TimestampedProbability"):
        align(bars(1), [invalid])
    with pytest.raises(PredictionAlignmentError, match=r"bars\[0\].*exactly Bar"):
        align([invalid], [TimestampedProbability(START, 0.5)])


def test_bar_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(PredictionAlignmentError, match=r"bars\[0\].timestamp.*timezone"):
        align([bar(0, timestamp=datetime(2024, 1, 1))], [TimestampedProbability(START, 0.5)])


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ((START, START), r"bars\[1\].timestamp.*strictly later"),
        ((START + timedelta(hours=1), START), r"bars\[1\].timestamp.*strictly later"),
        ((START, START + timedelta(hours=2)), r"bars\[1\].timestamp.*exactly one hour"),
    ],
)
def test_bar_timestamps_are_strict_chronological_hourly(
    timestamps: tuple[datetime, datetime], message: str
) -> None:
    source_bars = tuple(bar(index, timestamp=value) for index, value in enumerate(timestamps))
    with pytest.raises(PredictionAlignmentError, match=message):
        align(source_bars, predictions(bars(2), (0.2, 0.8)))


def test_equivalent_timezone_bar_instants_are_consistent() -> None:
    offset = timezone(timedelta(hours=2))
    source_bars = tuple(
        bar(index, timestamp=(START + timedelta(hours=index)).astimezone(offset))
        for index in range(2)
    )
    assert align(source_bars, predictions(bars(2), (0.2, 0.8))) == (0.2, 0.8)


def test_prediction_timestamp_must_be_timezone_aware() -> None:
    invalid = TimestampedProbability(datetime(2024, 1, 1), 0.5)
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].timestamp.*timezone"):
        align(bars(1), [invalid])


@pytest.mark.parametrize(
    ("timestamps", "message"),
    [
        ((START, START), r"predictions\[1\].timestamp.*strictly later"),
        ((START + timedelta(hours=1), START), r"predictions\[1\].timestamp.*strictly later"),
        ((START, START + timedelta(hours=2)), r"predictions\[1\].timestamp.*exactly one hour"),
    ],
)
def test_prediction_timestamps_are_strict_chronological_hourly(
    timestamps: tuple[datetime, datetime], message: str
) -> None:
    source = tuple(TimestampedProbability(value, 0.5) for value in timestamps)
    with pytest.raises(PredictionAlignmentError, match=message):
        align(bars(2), source)


class BrokenTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        raise ValueError("broken offset")

    def dst(self, dt: datetime | None) -> timedelta | None:
        return timedelta(0)


def test_invalid_prediction_utc_normalization_is_rejected() -> None:
    invalid = TimestampedProbability(datetime(2024, 1, 1, tzinfo=BrokenTimezone()), 0.5)
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].timestamp.*invalid UTC offset"):
        align(bars(1), [invalid])


def test_timezone_equivalent_prediction_instants_are_accepted_without_mutation() -> None:
    offset = timezone(timedelta(hours=1))
    source_bars = bars(2)
    source = tuple(
        TimestampedProbability(item.timestamp.astimezone(offset), probability)
        for item, probability in zip(source_bars, (0.2, 0.8), strict=True)
    )
    original_timestamps = tuple(item.timestamp for item in source)
    assert align(source_bars, source) == (0.2, 0.8)
    assert tuple(item.timestamp for item in source) == original_timestamps


@pytest.mark.parametrize("invalid", [True, False, "0.5", None])
def test_non_numeric_or_boolean_probability_is_rejected(invalid: object) -> None:
    source = [TimestampedProbability(START, invalid)]  # type: ignore[arg-type]
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].probability.*numeric and not boolean"):
        align(bars(1), source)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_probability_is_rejected(invalid: float) -> None:
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].probability.*finite"):
        align(bars(1), [TimestampedProbability(START, invalid)])


@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_out_of_range_probability_is_rejected(invalid: float) -> None:
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].probability.*inclusive range"):
        align(bars(1), [TimestampedProbability(START, invalid)])


def test_probability_endpoints_are_accepted_and_normalized_to_float() -> None:
    assert align(bars(2), [TimestampedProbability(START, 0), TimestampedProbability(START + timedelta(hours=1), 1)]) == (0.0, 1.0)


@pytest.mark.parametrize("count", [5, 7])
def test_prediction_count_must_equal_bar_count(count: int) -> None:
    source = tuple(
        TimestampedProbability(START + timedelta(hours=index), 0.5)
        for index in range(count)
    )
    with pytest.raises(PredictionAlignmentError, match=rf"predictions={count}, bars=6"):
        align(bars(), source)


@pytest.mark.parametrize("shift", [timedelta(hours=1), -timedelta(hours=1)])
def test_forward_and_backward_shifts_are_rejected(shift: timedelta) -> None:
    source = tuple(
        TimestampedProbability(item.timestamp + shift, probability)
        for item, probability in zip(bars(), PROBABILITIES, strict=True)
    )
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[0\].timestamp.*does not match bars\[0\].timestamp"):
        align(bars(), source)


def test_single_internal_mismatch_identifies_index() -> None:
    source = list(predictions())
    source[3] = replace(source[3], timestamp=source[3].timestamp + timedelta(minutes=1))
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[3\].timestamp"):
        align(bars(), source)


def test_reordered_predictions_are_rejected_not_sorted() -> None:
    source = list(predictions())
    source[2], source[3] = source[3], source[2]
    with pytest.raises(PredictionAlignmentError, match=r"predictions\[2\].timestamp.*exactly one hour"):
        align(bars(), source)


def test_source_with_correct_length_is_not_dropped_filled_or_shifted() -> None:
    assert align(bars(), predictions()) == PROBABILITIES
    mismatch = list(predictions())
    mismatch[3] = replace(mismatch[3], timestamp=mismatch[3].timestamp + timedelta(minutes=30))
    with pytest.raises(PredictionAlignmentError):
        align(bars(), mismatch)


def test_source_respects_information_boundaries() -> None:
    source = inspect.getsource(alignment_module)
    for forbidden in (
        "TargetWeight", "probabilities_to_target_weights", "threshold", "model",
        "feature", "label", "training", "validation split", "test split", "PortfolioState",
        "fee", "slippage", "tolerance", "notional", "MarketOrder", "Fill",
        "run_backtest", "run_target_weight_backtest", "reporting",
    ):
        assert forbidden not in source


def test_prefix_alignment_and_future_change_boundary() -> None:
    source_bars = bars()
    source_predictions = predictions(source_bars)
    full = align(source_bars, source_predictions)
    for length in range(1, len(source_bars) + 1):
        assert align(source_bars[:length], source_predictions[:length]) == full[:length]
    changed = list(source_predictions)
    changed[-1] = replace(changed[-1], probability=0.01)
    assert align(source_bars, changed)[:-1] == full[:-1]


def integration_result():
    source_bars = bars()
    source_predictions = predictions(source_bars)
    aligned = align(source_bars, source_predictions)
    targets = probabilities_to_target_weights(
        aligned, lower_threshold=0.40, upper_threshold=0.70
    )
    result = run_target_weight_backtest(
        bars=source_bars,
        targets=targets,
        initial_state=PortfolioState(cash=1000.0),
        fee_rate=0.0,
        slippage_rate=0.0,
        rebalance_tolerance=0.0,
        minimum_trade_notional=0.0,
    )
    return source_bars, source_predictions, aligned, targets, result


def test_full_synthetic_integration_timing_and_reporting() -> None:
    source_bars, source_predictions, aligned, targets, result = integration_result()
    _, repeated_predictions, repeated_aligned, repeated_targets, repeated_result = integration_result()
    assert len(source_bars) == len(source_predictions) == len(aligned) == len(targets)
    assert aligned == PROBABILITIES
    assert tuple(target.weight for target in targets) == (0.0, 1.0, 0.5, 0.0, 1.0, 0.5)
    assert all(item.timestamp == source_bars[index].timestamp for index, item in enumerate(source_predictions))
    assert all(fill.executed_at == fill.order_created_at + timedelta(hours=1) for fill in result.fills)
    assert all(fill.executed_at != fill.order_created_at for fill in result.fills)
    assert all(fill.executed_at != source_bars[0].timestamp for fill in result.fills)
    assert result.unexecuted_final_target is not None
    assert result.unexecuted_final_target.decision_bar_timestamp == source_bars[-1].timestamp
    trade_log = build_trade_log(result)
    summary = summarize_backtest(result)
    assert len(trade_log) == len(result.fills) == summary.trade_count
    exposures = tuple(
        snapshot.position_quantity * snapshot.close_price / snapshot.portfolio_value
        for snapshot in result.portfolio_history
    )
    assert summary.average_exposure == pytest.approx(math.fsum(exposures) / len(exposures))
    assert source_predictions == repeated_predictions
    assert aligned == repeated_aligned and targets == repeated_targets and result == repeated_result
