# Bar Timestamp Semantics

## Scope

This document defines the timestamp convention used by the execution-aware backtest project for 1-hour OHLCV bars.

The convention applies to both the synthetic test fixture and any real BTC/ETH OHLCV snapshot used by the project, unless a separate data-source metadata file explicitly states otherwise.

## Timestamp convention

`Bar.timestamp` denotes the **bar open time**, which is the start of the bar interval.

For a 1-hour bar with:

```text
timestamp = 2024-01-01T00:00:00Z
```

the represented interval is:

```text
[2024-01-01T00:00:00Z, 2024-01-01T01:00:00Z)
```

The interval is left-inclusive and right-exclusive.

Therefore:

- `open` is the price at the start of the interval.
- `high` is the highest observed price during the interval.
- `low` is the lowest observed price during the interval.
- `close` is the final observed price before the interval ends.
- `volume` is the volume accumulated during the interval.

The following bar, timestamped `2024-01-01T01:00:00Z`, begins at the instant the previous bar ends.

## Signal availability

A strategy may use bar `t` only after bar `t` has completed.

For a bar covering `[00:00, 01:00)`:

- the bar becomes fully observable at `01:00`;
- the strategy may then use its close and other completed OHLCV values;
- no strategy decision may use values from an incomplete bar or any future bar.

## Execution convention

A signal generated after bar `t` closes may produce a pending market order.

That order is executed using the open price of bar `t+1`.

For example:

```text
bar t timestamp:       00:00
bar t interval:        [00:00, 01:00)
signal availability:   after bar t completes at 01:00
execution bar:         bar t+1
execution reference:   bar t+1 open at 01:00
```

The backtest must not execute a signal using bar `t` close.

The backtest must not use the final bar's close to simulate an order when no next bar exists.

## Event ordering

For each bar `t`, the intended event order is:

1. Execute any pending order generated from bar `t-1`, using `bar[t].open`.
2. Apply directional slippage.
3. Calculate the proportional fee.
4. Update cash and position.
5. Record the fill or trade.
6. Mark the portfolio to market using `bar[t].close`.
7. Store the portfolio snapshot.
8. Allow the strategy to use data through bar `t` only.
9. Generate a signal or target position.
10. If necessary, create a pending order for execution at `bar[t+1].open`.

The engine implementation, rather than timestamp comparison alone, is responsible for enforcing this event sequence.

## Order timestamp interpretation

Until the order model is renamed or extended, `MarketOrder.created_at` should be interpreted as the timestamp of the decision bar whose close generated the order.

It is therefore a **decision-bar identifier**, not necessarily the exact wall-clock order-submission instant.

For example:

```text
decision bar timestamp = 00:00
decision bar close     = 01:00
execution bar timestamp = 01:00
```

This convention allows the order to retain its source-bar identity while the engine separately enforces next-bar execution.

## Timezone

All timestamps must be timezone-aware and normalized to UTC.

Naive timestamps are invalid.

## Missing and irregular bars

Expected bar spacing is exactly one hour.

The data layer must reject:

- duplicated timestamps;
- non-monotonic timestamps;
- missing hourly intervals;
- irregular spacing.

Missing bars must not be silently sorted, forward-filled, interpolated, or synthesized.

## Data-source verification

The current timestamp convention is supported by matching the price at `01:00:00Z` to the open of the bar labeled `01:00:00Z`, indicating that timestamps denote interval starts.

For each real data snapshot, metadata should record:

- data source;
- symbol;
- bar frequency;
- timezone;
- timestamp semantics;
- retrieval date;
- any transformation applied after retrieval.

If timestamp semantics cannot be verified for a data source, the snapshot must not be used for execution-timing claims until the ambiguity is resolved.
