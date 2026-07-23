"""Strict loading and validation for hourly OHLCV CSV data."""

import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.events import Bar

_EXPECTED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_NUMERIC_FIELDS = ("open", "high", "low", "close", "volume")
_BAR_INTERVAL = timedelta(hours=1)


class BarDataError(ValueError):
    """Raised when OHLCV CSV data violates the required schema or invariants."""


def _parse_timestamp(value: str, row_number: int) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise BarDataError(
            f"row {row_number}: field 'timestamp' is not a valid ISO 8601 datetime: {value!r}"
        ) from error

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise BarDataError(
            f"row {row_number}: field 'timestamp' must include timezone information: {value!r}"
        )

    return timestamp.astimezone(timezone.utc)


def _parse_number(field: str, value: str, row_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise BarDataError(
            f"row {row_number}: field {field!r} is not numeric: {value!r}"
        ) from error

    if not math.isfinite(number):
        raise BarDataError(
            f"row {row_number}: field {field!r} must be finite: {value!r}"
        )

    return number


def _validate_prices(
    *, row_number: int, timestamp: datetime, open_: float, high: float, low: float, close: float
) -> None:
    context = f"row {row_number} ({timestamp.isoformat()})"
    if high < open_:
        raise BarDataError(f"{context}: field 'high' must be greater than or equal to 'open'")
    if high < close:
        raise BarDataError(f"{context}: field 'high' must be greater than or equal to 'close'")
    if low > open_:
        raise BarDataError(f"{context}: field 'low' must be less than or equal to 'open'")
    if low > close:
        raise BarDataError(f"{context}: field 'low' must be less than or equal to 'close'")
    if high < low:
        raise BarDataError(f"{context}: field 'high' must be greater than or equal to 'low'")


def load_bars_csv(path: str | Path) -> list[Bar]:
    """Load strictly ordered, consecutive hourly OHLCV bars from ``path``.

    Rows are validated in file order. Invalid ordering and missing intervals are
    rejected rather than sorted, filled, inferred, or repaired.
    """

    csv_path = Path(path)
    bars: list[Bar] = []
    previous_timestamp: datetime | None = None

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise BarDataError(f"{csv_path}: empty CSV file")
        if tuple(reader.fieldnames) != _EXPECTED_COLUMNS:
            expected = ",".join(_EXPECTED_COLUMNS)
            actual = ",".join(reader.fieldnames)
            raise BarDataError(
                f"{csv_path}: invalid CSV schema; expected {expected!r}, got {actual!r}"
            )

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise BarDataError(f"row {row_number}: unexpected extra values")

            values: dict[str, str] = {}
            for field in _EXPECTED_COLUMNS:
                raw_value = row.get(field)
                if raw_value is None or not raw_value.strip():
                    raise BarDataError(f"row {row_number}: field {field!r} is empty")
                values[field] = raw_value.strip()

            timestamp = _parse_timestamp(values["timestamp"], row_number)
            if previous_timestamp is not None:
                if timestamp == previous_timestamp:
                    raise BarDataError(
                        f"row {row_number}: duplicate timestamp {timestamp.isoformat()}"
                    )
                if timestamp < previous_timestamp:
                    raise BarDataError(
                        f"row {row_number}: timestamp {timestamp.isoformat()} is not "
                        f"strictly later than preceding timestamp {previous_timestamp.isoformat()}"
                    )
                if timestamp - previous_timestamp != _BAR_INTERVAL:
                    raise BarDataError(
                        f"row {row_number}: timestamp {timestamp.isoformat()} is not exactly "
                        f"one hour after preceding timestamp {previous_timestamp.isoformat()}"
                    )

            numeric = {
                field: _parse_number(field, values[field], row_number)
                for field in _NUMERIC_FIELDS
            }
            if numeric["volume"] < 0:
                raise BarDataError(
                    f"row {row_number} ({timestamp.isoformat()}): field 'volume' must be non-negative"
                )

            _validate_prices(
                row_number=row_number,
                timestamp=timestamp,
                open_=numeric["open"],
                high=numeric["high"],
                low=numeric["low"],
                close=numeric["close"],
            )
            bars.append(
                Bar(
                    timestamp=timestamp,
                    open=numeric["open"],
                    high=numeric["high"],
                    low=numeric["low"],
                    close=numeric["close"],
                    volume=numeric["volume"],
                )
            )
            previous_timestamp = timestamp

    if not bars:
        raise BarDataError(f"{csv_path}: CSV file contains a header but no data rows")

    return bars
