# Final Descriptive Evaluation — Rule-Based Real-Market Demonstration

## 1. Evaluation scope

This document is a descriptive evaluation of the already-completed Stage 6
rule-based real-market run. Stage 10 did not rerun the experiment, and no
parameter or policy was changed after performance inspection. The purpose is to
evaluate the execution path on a real hourly price series, not to validate a
signal. This is neither an ML result nor production trading evidence.

The formal metadata records the experiment as follows:

| Field | Frozen value |
| --- | --- |
| `experiment_id` | `rule_based_real_market_BTCUSDT_2024-01-01_2024-12-31_31e4ae5ef79d` |
| Instrument | Binance Spot `BTCUSDT` |
| `bar_interval` | `1h` |
| Evaluation period | 2024 calendar year |
| `timestamp_semantics` | `bar_open_time` |
| `initial_cash` | `1000.0` |
| `fee_rate` | `0.001` |
| `slippage_rate` | `0.0005` |
| `rebalance_tolerance` | `0.0` |
| `minimum_trade_notional` | `0.0` |
| Position domain | Long-only cash/asset accounting |
| Short selling | Not supported |
| Leverage | Not supported |
| `execution_timing` | `decision_after_bar_close_execute_next_bar_open` |
| Parameter selection performed | `False` |
| Performance-based adjustment performed | `False` |

## 2. Data provenance

The frozen input is Binance Public Data Spot `BTCUSDT` at a `1h` interval. It
covers `2024-01-01T00:00:00Z` through `2024-12-31T23:00:00Z`, contains `8784`
rows, and uses bar-open timestamps. The retrieval timestamp is
`2026-07-27T13:31:08Z`.

The processed input is `BTCUSDT_1h_2024.csv`, with SHA-256:

```text
31e4ae5ef79de2b3e8546fb21bef57300e0a43c4cabab10bfd81e53803fc9db9
```

That checksum agrees across the existing Stage 6 metadata, the frozen dataset
metadata, and the local processed file. The dataset metadata reports zero
duplicate timestamps and zero missing timestamps. It also records
`sorting_performed=False`, `filling_performed=False`,
`interpolation_performed=False`, `deduplication_performed=False`, and
`repair_performed=False`.

The provenance records are
[`BTCUSDT_1h_2024_metadata.csv`](../data/metadata/BTCUSDT_1h_2024_metadata.csv)
and
[`BTCUSDT_1h_2024_snapshot_spec.csv`](../data/metadata/BTCUSDT_1h_2024_snapshot_spec.csv).
The Stage 6 run metadata is
[`metadata.csv`](../results/rule_based_real_market_demo/metadata.csv).

The 2024 period was designated for this rule-based execution demonstration. It
is not an untouched ML final-test dataset and must not be described or reused as
one.

## 3. Fixed policy definitions

### 3.1 Previous-close fractional rule

The formal rule definition is:

```text
first=0.00;close_up=0.75;close_equal=0.50;close_down=0.25
```

The first completed bar maps to target weight `0.00`. Each later completed bar
maps to `0.75` when its close is higher than the immediately preceding close,
`0.50` when equal, and `0.25` when lower. The rule uses only the current and
immediately preceding completed closes; it does not use future data. Targets are
created after close and may execute only at the next bar open. The final target
remains unexecuted because no later bar exists.

This rule was fixed before the full result was reviewed. It is a workflow
demonstration, not a validated momentum strategy.

### 3.2 Execution-aligned buy-and-hold

The formal definition is:

```text
target[0]=0.0;target[1:]=1.0
```

The first target is zero and every later target is full long exposure. Under
the shared next-bar invariant, `target[1]` first becomes eligible at `bar[2]`
open. The baseline therefore does not enter at the first dataset open and uses
the same earliest eligible execution timing as the fractional rule.

### 3.3 Zero-position baseline

The formal definition is:

```text
all_target_weights=0.0
```

This baseline validates the no-trade path: cash remains unchanged, with no
fills, fees, turnover, or realized exposure.

## 4. Formal result table

The following values reproduce the three rows and relevant formal fields from
[`summary.csv`](../results/rule_based_real_market_demo/summary.csv). Returns and
drawdowns are raw decimals, not percentages.

| `policy_name` | `final_portfolio_value` | `cumulative_return` | `maximum_drawdown` | `turnover` | `average_realized_exposure` | `total_fees` | `buy_count` | `sell_count` | `trade_count` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `previous_close_momentum_fractional` | `31.806726377151563` | `-0.9681932736228485` | `-0.9691333630513939` | `689.1284190585849` | `0.5058236520364872` | `689.1284190585884` | `4010` | `4772` | `8782` |
| `execution_aligned_buy_and_hold` | `2192.6301781973702` | `1.19263017819737` | `-0.3231069072207683` | `0.9990009990009989` | `0.9997723132969034` | `0.9990009990009989` | `1` | `0` | `1` |
| `zero_position` | `1000.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0` | `0` | `0` |

All three rows report `initial_portfolio_value=1000.0` and
`final_target_unexecuted=True`.

## 5. Execution diagnostics

### Fractional rule

The target file contains `4680` changes between adjacent intended target
weights. That count is read from the formal target sequence and is not a field
in `summary.csv`. It is conceptually distinct from the `8782` actual fills:
target transitions describe changes in intent, while fills describe executed
portfolio adjustments. Price movement and realized exposure drift can require
partial rebalancing even when adjacent intended weights are equal.

The trade log contains `4010` buys and `4772` sells, confirming frequent
two-sided partial rebalancing. Formal turnover is `689.1284190585849`, and
accumulated fees are `689.1284190585884`. Average realized close-marked exposure
is `0.5058236520364872`; it is computed from actual cash and holdings, not from
the intended target weights. The final portfolio value is
`31.806726377151563`, and maximum drawdown is `-0.9691333630513939`.

### Execution-aligned buy-and-hold

The baseline has one intended target transition and one actual fill: a buy at
`2024-01-01T02:00:00Z`. Its turnover of `0.9990009990009989` is far below the
fractional rule's turnover, and its total fee is `0.9990009990009989`. Average
realized exposure is `0.9997723132969034`, reflecting the two initial
close-marked snapshots before entry rather than an assumed target exposure.
Final value is `2192.6301781973702`, and maximum drawdown is
`-0.3231069072207683`.

### Zero-position

The zero-position baseline has no target transitions, fills, or fees. Turnover
and average realized exposure are both `0.0`, and portfolio value remains
`1000.0` throughout the recorded history.

More fills and higher turnover are execution diagnostics, not indicators of a
better policy. Likewise, the difference between intended targets and realized
close-marked exposure is expected because execution prices, fees, price changes,
and retained cash affect the actual portfolio path.

## 6. Conservative interpretation

The fractional rule performed poorly in this fixed cost-inclusive scenario: its
formal cumulative return is `-0.9681932736228485`. This result does not support
a profitability claim for this rule during the designated period. It neither
establishes nor rejects momentum as a broad market phenomenon and cannot be
generalized beyond this fixed demonstration.

The difference from buy-and-hold does not mean that buy-and-hold was selected
as an optimal policy. Exposure differs materially between the cases, which
affects both return and drawdown comparisons. Transaction costs contributed to
the realized path, but the total underperformance cannot be attributed solely
to fees or slippage: costs also alter affordable quantities, cash, subsequent
trade sizes, and the entire portfolio path. No exact transaction-cost
attribution is claimed.

The zero-position case is an operational baseline, not an investment
recommendation. The main value of this experiment is the descriptive evidence
that next-bar timing, transaction-cost handling, immutable long-only
accounting, trade logging, and reporting operate on a full real price path. It
does not establish production readiness or a broadly validated trading result.

## 7. Separation from synthetic diagnostics

The repository separately contains synthetic diagnostics for allocation mapping
behavior, continuous rebalancing, fixed fee/slippage scenarios, and predefined
maximum target-weight caps. Those controlled cases validate mechanics; they are
not additional real-market policy-selection evidence.

Stage 10 neither combines nor ranks synthetic and real-market results. This
formal real-market evaluation uses only the fixed Stage 6 cost-inclusive
scenario. A broader real-market cost-sensitivity experiment was not conducted,
and Stage 10 did not create one.

## 8. Reproducibility

The existing Stage 6 runner command was:

```powershell
.\.venv\Scripts\python.exe scripts\run_rule_based_real_market_demo.py `
  --input data\processed\BTCUSDT_1h_2024.csv `
  --output-dir results\rule_based_real_market_demo `
  --symbol BTCUSDT
```

Stage 10 documents this command but did not execute it. The fixed parameters are
the values in Section 1. Expected formal outputs are `summary.csv`,
`trades.csv`, `portfolio_history.csv`, `targets.csv`, and `metadata.csv` in
`results/rule_based_real_market_demo/`.

The relevant runner regression command is:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_rule_based_real_market_demo.py
```

Reproduction uses the processed file
`data/processed/BTCUSDT_1h_2024.csv`, its SHA-256
`31e4ae5ef79de2b3e8546fb21bef57300e0a43c4cabab10bfd81e53803fc9db9`, and
`data/metadata/BTCUSDT_1h_2024_metadata.csv`. The processed market-data CSV is
locally ignored by Git. The run is therefore reproducible from the frozen local
snapshot, but it is not fully self-contained from a repository checkout alone.

## 9. Limitations

- The demonstration covers one asset, one venue, one calendar period, and
  hourly bars only.
- It uses one deterministic transparent rule with no parameter search.
- Short selling and leverage are not supported.
- The execution model includes neither order-book dynamics nor market impact.
- Orders are filled deterministically; partial fills are not modeled.
- Costs are limited to fixed proportional fee and slippage assumptions.
- Latency beyond the completed-bar decision and next-bar-open convention is not
  modeled.
- The repository is not a general-purpose strategy framework.
- No statistical inference is performed.
- No real ML artifact or ML execution integration is included.
- The period is not interpreted as an untouched ML final test.
- No broad profitability conclusion can be drawn.
- The designated rule-based demonstration period must not be reused as a future
  untouched ML final test.

## 10. Final conclusion

The fixed Stage 6 case shows that the engine preserves completed-bar decisions,
next-bar-open execution, explicit transaction costs, immutable long-only
accounting, and reproducible reporting across a full hourly BTCUSDT path. The
fractional rule produced poor cost-inclusive results in this period. The result
is therefore retained as a descriptive execution and research-process
demonstration, not as evidence of a tradable signal.
