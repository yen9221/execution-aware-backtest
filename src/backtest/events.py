"""Minimal immutable market-data records."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Bar:
    """One validated OHLCV bar with a timezone-aware timestamp."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
