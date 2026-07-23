"""Minimal quantity-based market-order records."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    """Supported market-order directions."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class MarketOrder:
    """An immutable request to trade an asset quantity at the market."""

    created_at: datetime
    side: Side
    quantity: float
