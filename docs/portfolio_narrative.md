# Portfolio Narrative — Execution-Aware Backtest

## 1. Project positioning

This project is a deterministic execution-aware backtest prototype focused on
timing, transaction costs, immutable portfolio accounting, and reproducible
diagnostics. Signal validation is held separate, and the project is not a
production trading system.

## 2. One-sentence description

Built a deterministic single-asset backtest prototype that enforces completed-bar decisions and next-bar-open execution while making costs, portfolio accounting, formal outputs, and research limitations explicit and reproducible.

## 3. GitHub repository description

Deterministic single-asset backtest prototype with strict next-bar timing, explicit costs, immutable accounting, and reproducible diagnostics.

## 4. CV bullets

- Built a deterministic single-asset execution engine with completed-bar decisions, next-bar-open fills, explicit fee and slippage handling, and immutable long-only accounting.
- Validated timing, affordability, partial rebalancing, reporting, artifact alignment, and policy-freeze boundaries through synthetic tests and reproducible formal CSV outputs.
- Evaluated a fixed hourly BTCUSDT rule with execution-aligned baselines and reported its negative result transparently as an execution demonstration rather than evidence of profitability.

## 5. 30-second interview explanation

I built this project to test the execution assumptions that often get hidden
behind a strategy result. A decision can use only a completed bar and can trade
only at the next bar open. I separated target generation, positioning, fills,
portfolio accounting, and reporting, then validated each boundary with
deterministic synthetic cases. I also ran one fixed rule on a frozen hourly
BTCUSDT path with fees, slippage, buy-and-hold, and zero-position baselines. The
fractional rule performed poorly, so I present the project as evidence of
execution correctness and transparent research practice, not signal quality.

## 6. 90-second interview explanation

I built the project because a backtest can look convincing while still mixing
decision timing, sizing, fills, accounting, and evaluation in ways that create
look-ahead or hide transaction costs. The core contract is simple: a target is
formed only after a bar is complete, it stays pending, and it may execute only
at the next bar open. A final-bar target cannot execute without a later bar.

The implementation keeps responsibilities explicit. A strategy or allocation
function produces price-free target intent. Positioning observes the next open
and sizes a binary or fractional order, including affordability, rebalance
tolerance, and minimum-notional controls. Execution applies directional
slippage and proportional fees. Immutable portfolio accounting updates cash,
quantity, and cumulative fees. Reporting then works only from actual fills and
close-marked holdings, so realized exposure is not confused with intended
weight.

I validated those mechanics with hand-checkable synthetic cases covering
timing, partial rebalances, costs, accounting, repeated targets, final targets,
allocation mappings, and reporting reconciliation. Separately, I used a fixed
previous-close fractional rule on 8,784 hourly BTCUSDT bars, alongside
execution-aligned buy-and-hold and zero-position baselines. The fractional rule
lost heavily in that cost-inclusive period, which is retained as a transparent
negative result rather than reframed as a successful strategy.

The repository also defines strict prediction-artifact and policy-freeze
interfaces. Real ML integration is deliberately deferred because the existing
ML workflow is daily and this engine's contract is hourly; the project does not
manufacture intraday predictions to force compatibility.

## 7. Key methodology choices

### Why next-bar-open execution?

A completed bar's close is not available until that bar ends. Executing a
decision at the same close would use an unrealistically convenient price and
can introduce look-ahead. The next observable open provides a clear,
testable execution boundary.

### Why separate target intent from execution?

Target intent says what exposure is desired without embedding a future price,
fill, or portfolio mutation. Keeping positioning, execution, accounting, and
reporting separate makes each responsibility deterministic, independently
testable, and auditable.

### Why include zero-position and execution-aligned buy-and-hold?

Zero-position validates the no-trade path, unchanged cash, zero fees, and zero
exposure. Execution-aligned buy-and-hold provides an exposure benchmark under
the same delayed-entry and transaction-cost assumptions, rather than receiving
an advantaged first-open entry.

### Why not force the daily ML predictions into the hourly engine?

One daily probability cannot satisfy exact one-to-one hourly alignment.
Forward filling, duplication, interpolation, shifting, relabeling, or nearest
matching would create artificial intraday information and change the intended
execution path. Integration therefore waits for a genuine frozen hourly
artifact.

### Why retain a negative real-market result?

Rejection-oriented research should preserve outcomes that do not support the
initial rule. Keeping the result visible demonstrates transparency and keeps
the project's actual contribution in focus: execution timing, accounting,
cost handling, and reproducible diagnostics.

## 8. Common interview questions

### Is this a production backtester?

No. It is a correctness-focused prototype with deliberately narrow scope. It
does not model order books, market impact, partial fills, live trading, or
multi-asset portfolios.

### Why did the fractional rule lose so much?

It rebalanced very frequently under one fixed cost-inclusive scenario, while
its exposure and price path also differed from the baselines. Costs contributed
to the path, but they are not claimed as the sole cause, and the experiment was
not designed to establish signal quality.

### Does buy-and-hold outperforming mean your strategy failed?

No policy was selected from the result. Buy-and-hold is a descriptive exposure
benchmark, while the fractional rule is a transparent workflow demonstration.
Their different exposure, turnover, and trading paths prevent a broad signal
conclusion from one period.

### How did you avoid look-ahead bias?

The engine processes pending prior-bar targets at the current open, records the
current close-marked state, and only then permits a target from the completed
current bar. The final target remains unexecuted. Exact timestamp alignment is
also enforced, although that alone cannot certify upstream feature design.

### How do you know the accounting is correct?

Deterministic tests reconcile every fill with cash, position quantity,
cumulative fees, close-marked portfolio values, trade logs, and reporting
summaries. Focused tests also cover affordability residuals, repeated targets,
partial rebalances, and invalid states.

### Why did you not add market impact or order-book simulation?

The project prioritizes a small, auditable event and accounting model before
adding assumptions that require richer data and calibration. Those omissions
remain explicit limitations rather than being hidden behind a realism claim.

### Why is the ML model not connected yet?

The existing ML project produces daily outputs, while this execution interface
requires hourly predictions with exact bar correspondence. Rather than invent
hourly values, integration is deliberately deferred until a genuine immutable
hourly artifact exists.

### What would you add next?

First, I would produce a genuine leakage-aware frozen hourly prediction artifact
in the separate ML project. Then I would integrate it optionally through the
existing strict prediction-artifact and policy-freeze interfaces, without broad
architecture expansion or changing the standalone results.

## 9. Relationship to the Financial Signal Validation project

The projects demonstrate separate competencies:

```text
Financial Signal Validation Prototype
-> leakage-aware chronological model research
-> baseline comparison
-> validation-only decisions
-> held-out evaluation
-> weak or negative signal conclusion

Execution-Aware Backtest
-> next-bar timing
-> fills and transaction costs
-> portfolio accounting
-> trade logs and realized exposure
-> descriptive execution diagnostics
```

Model classification metrics are not interpreted directly as tradable alpha.
The projects are not currently connected end to end: this repository has
artifact boundaries for a future frozen hourly interface, while model training
and validation remain in the separate project.

## 10. Final limitations statement

This prototype covers one asset and one venue with deterministic bar-level
fills, fixed proportional costs, and no market impact or partial-fill model. It
makes no production-readiness claim and its fixed demonstration does not
establish robust profitability.
