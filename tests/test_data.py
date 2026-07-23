import csv
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtest.data import BarDataError, load_bars_csv
from backtest.events import Bar

HEADER = ("timestamp", "open", "high", "low", "close", "volume")
VALID_ROWS = [
    ("2024-01-01T00:00:00Z", "100", "103", "99", "102", "10"),
    ("2024-01-01T01:00:00Z", "102", "104", "100", "101", "12"),
]
FIXTURE = Path(__file__).parent / "fixtures" / "simple_bars.csv"


def write_csv(
    tmp_path: Path,
    rows: list[tuple[str, ...]],
    header: tuple[str, ...] = HEADER,
) -> Path:
    path = tmp_path / "bars.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_existing_fixture_loads_in_original_order() -> None:
    bars = load_bars_csv(FIXTURE)

    assert len(bars) == 6
    assert [bar.timestamp.hour for bar in bars] == [0, 1, 2, 3, 4, 5]
    assert bars[0].timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert bars[-1].timestamp == datetime(2024, 1, 1, 5, tzinfo=timezone.utc)
    assert bars[0].timestamp.utcoffset() == timedelta(0)
    assert bars[-1].timestamp.utcoffset() == timedelta(0)


def test_numeric_fields_are_floats() -> None:
    bar = load_bars_csv(FIXTURE)[0]

    assert all(
        isinstance(value, float)
        for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)
    )


def test_bar_is_immutable() -> None:
    bar = load_bars_csv(FIXTURE)[0]

    with pytest.raises(FrozenInstanceError):
        bar.close = 0.0  # type: ignore[misc]


def test_duplicate_timestamp_is_rejected(tmp_path: Path) -> None:
    rows = [VALID_ROWS[0], ("2024-01-01T00:00:00+00:00", *VALID_ROWS[1][1:])]

    with pytest.raises(BarDataError, match="duplicate timestamp"):
        load_bars_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize(
    "timestamps",
    [
        ("2024-01-01T01:00:00Z", "2024-01-01T00:00:00Z"),
        ("2024-01-01T00:00:00Z", "2023-12-31T23:00:00Z"),
    ],
)
def test_descending_or_non_monotonic_timestamp_is_rejected(
    tmp_path: Path, timestamps: tuple[str, str]
) -> None:
    rows = [(timestamps[index], *VALID_ROWS[index][1:]) for index in range(2)]

    with pytest.raises(BarDataError, match="not strictly later"):
        load_bars_csv(write_csv(tmp_path, rows))


def test_missing_hour_is_rejected_without_filling(tmp_path: Path) -> None:
    rows = [VALID_ROWS[0], ("2024-01-01T02:00:00Z", *VALID_ROWS[1][1:])]
    path = write_csv(tmp_path, rows)
    original = path.read_bytes()

    with pytest.raises(BarDataError, match="not exactly one hour"):
        load_bars_csv(path)

    assert path.read_bytes() == original


def test_naive_timestamp_is_rejected(tmp_path: Path) -> None:
    rows = [("2024-01-01T00:00:00", *VALID_ROWS[0][1:])]

    with pytest.raises(BarDataError, match="timezone information"):
        load_bars_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize(
    "header",
    [
        HEADER[:-1],
        (*HEADER, "trades"),
        ("open", "timestamp", "high", "low", "close", "volume"),
    ],
    ids=["missing-column", "extra-column", "incorrect-order"],
)
def test_invalid_schema_is_rejected(tmp_path: Path, header: tuple[str, ...]) -> None:
    row = VALID_ROWS[0][: len(header)]

    with pytest.raises(BarDataError, match="invalid CSV schema"):
        load_bars_csv(write_csv(tmp_path, [row], header))


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bars.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(BarDataError, match="empty CSV file"):
        load_bars_csv(path)


def test_header_only_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BarDataError, match="header but no data rows"):
        load_bars_csv(write_csv(tmp_path, []))


def test_empty_required_value_is_rejected(tmp_path: Path) -> None:
    row = list(VALID_ROWS[0])
    row[4] = ""

    with pytest.raises(BarDataError, match="field 'close' is empty"):
        load_bars_csv(write_csv(tmp_path, [tuple(row)]))


def test_invalid_numeric_text_is_rejected(tmp_path: Path) -> None:
    row = list(VALID_ROWS[0])
    row[1] = "not-a-number"

    with pytest.raises(BarDataError, match="field 'open' is not numeric"):
        load_bars_csv(write_csv(tmp_path, [tuple(row)]))


@pytest.mark.parametrize("value", ["NaN", "inf", "+inf", "-inf"])
def test_non_finite_number_is_rejected(tmp_path: Path, value: str) -> None:
    row = list(VALID_ROWS[0])
    row[5] = value

    with pytest.raises(BarDataError, match="field 'volume' must be finite"):
        load_bars_csv(write_csv(tmp_path, [tuple(row)]))


def test_negative_volume_is_rejected(tmp_path: Path) -> None:
    row = (*VALID_ROWS[0][:-1], "-1")

    with pytest.raises(BarDataError, match="field 'volume' must be non-negative"):
        load_bars_csv(write_csv(tmp_path, [row]))


@pytest.mark.parametrize(
    ("open_", "high", "low", "close", "message"),
    [
        ("100", "99", "98", "99", "high.*open"),
        ("100", "102", "98", "103", "high.*close"),
        ("100", "103", "101", "102", "low.*open"),
        ("102", "103", "101", "100", "low.*close"),
        ("100", "99", "101", "100", "high.*open"),
    ],
    ids=["high-below-open", "high-below-close", "low-above-open", "low-above-close", "high-below-low"],
)
def test_invalid_ohlc_is_rejected(
    tmp_path: Path, open_: str, high: str, low: str, close: str, message: str
) -> None:
    row = (VALID_ROWS[0][0], open_, high, low, close, "10")

    with pytest.raises(BarDataError, match=message):
        load_bars_csv(write_csv(tmp_path, [row]))


def test_loader_does_not_sort_non_chronological_input(tmp_path: Path) -> None:
    rows = [VALID_ROWS[1], VALID_ROWS[0]]
    path = write_csv(tmp_path, rows)
    original = path.read_bytes()

    with pytest.raises(BarDataError, match="not strictly later"):
        load_bars_csv(path)

    assert path.read_bytes() == original


def test_non_utc_timestamp_is_normalized_to_utc(tmp_path: Path) -> None:
    rows = [("2024-01-01T01:00:00+01:00", *VALID_ROWS[0][1:])]

    bar = load_bars_csv(write_csv(tmp_path, rows))[0]

    assert bar.timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert bar.timestamp.tzinfo is timezone.utc


def test_each_call_returns_a_new_list() -> None:
    first = load_bars_csv(FIXTURE)
    second = load_bars_csv(FIXTURE)

    assert first == second
    assert first is not second
    assert all(isinstance(bar, Bar) for bar in first)
