"""Run the frozen rule-based real-market execution demonstration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backtest.data import BarDataError, load_bars_csv
from backtest.engine import TargetWeightBacktestResult, run_target_weight_backtest
from backtest.events import Bar
from backtest.portfolio import PortfolioState
from backtest.positioning import TargetWeight
from backtest.reporting import BacktestSummary, summarize_backtest
from backtest.strategy import previous_close_momentum_target_weights

PERIOD_LABEL = "rule_based_demonstration_period"
POLICY_NAMES = (
    "previous_close_momentum_fractional",
    "execution_aligned_buy_and_hold",
    "zero_position",
)
INITIAL_CASH = 1000.0
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
REBALANCE_TOLERANCE = 0.0
MINIMUM_TRADE_NOTIONAL = 0.0
OUTPUT_FILENAMES = (
    "summary.csv",
    "trades.csv",
    "portfolio_history.csv",
    "targets.csv",
    "metadata.csv",
)
SUMMARY_COLUMNS = (
    "experiment_id", "symbol", "policy_name", "period_label",
    "start_timestamp", "end_timestamp", "bar_count",
    "initial_portfolio_value", "final_portfolio_value", "cumulative_return",
    "maximum_drawdown", "turnover", "average_realized_exposure",
    "total_fees", "buy_count", "sell_count", "trade_count", "final_cash",
    "final_position_quantity", "final_target_unexecuted",
)
TRADES_COLUMNS = (
    "experiment_id", "symbol", "policy_name", "fill_index", "side",
    "order_created_at", "executed_at", "reference_price", "fill_price",
    "quantity", "notional", "fee", "cash_flow",
)
PORTFOLIO_COLUMNS = (
    "experiment_id", "symbol", "policy_name", "bar_index", "timestamp",
    "open", "close", "cash", "position_quantity", "cumulative_fees",
    "portfolio_value", "realized_exposure",
)
TARGETS_COLUMNS = (
    "experiment_id", "symbol", "policy_name", "bar_index",
    "decision_timestamp", "target_weight", "execution_eligible_timestamp",
    "is_final_unexecuted_target",
)
METADATA_COLUMNS = (
    "experiment_id", "generated_at_utc", "input_path_argument",
    "input_filename", "input_sha256", "symbol", "period_label",
    "start_timestamp", "end_timestamp", "bar_count", "bar_interval",
    "timestamp_semantics", "data_source", "retrieval_date", "rule_definition",
    "buy_and_hold_definition", "zero_position_definition", "initial_cash",
    "fee_rate", "slippage_rate", "rebalance_tolerance",
    "minimum_trade_notional", "execution_timing",
    "parameter_selection_performed", "performance_based_adjustment_performed",
    "future_ml_test_status", "limitations",
)


class DemoError(RuntimeError):
    """Raised when the frozen demonstration cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class PolicyRun:
    policy_name: str
    targets: tuple[TargetWeight, ...]
    result: TargetWeightBacktestResult
    summary: BacktestSummary


@dataclass(frozen=True, slots=True)
class DemoRun:
    experiment_id: str
    symbol: str
    input_sha256: str
    bars: tuple[Bar, ...]
    policies: tuple[PolicyRun, ...]
    output_paths: tuple[Path, ...]


def _iso_utc(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_symbol(symbol: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", symbol.strip()).strip("_")
    if not sanitized:
        raise DemoError("symbol must contain at least one alphanumeric character")
    return sanitized.upper()


def _validate_bars(bars: tuple[Bar, ...]) -> None:
    for index, bar in enumerate(bars):
        if min(bar.open, bar.high, bar.low, bar.close) <= 0:
            raise DemoError(f"bars[{index}] contains a non-positive price")
        if bar.volume < 0:
            raise DemoError(f"bars[{index}] contains negative volume")


def _policy_targets(bars: tuple[Bar, ...]) -> tuple[tuple[str, tuple[TargetWeight, ...]], ...]:
    return (
        (
            POLICY_NAMES[0],
            previous_close_momentum_target_weights(bars),
        ),
        (
            POLICY_NAMES[1],
            tuple(
                TargetWeight(0.0 if index == 0 else 1.0)
                for index in range(len(bars))
            ),
        ),
        (
            POLICY_NAMES[2],
            tuple(TargetWeight(0.0) for _ in bars),
        ),
    )


def _run_policies(bars: tuple[Bar, ...]) -> tuple[PolicyRun, ...]:
    runs: list[PolicyRun] = []
    for policy_name, targets in _policy_targets(bars):
        result = run_target_weight_backtest(
            bars=bars,
            targets=targets,
            initial_state=PortfolioState(cash=INITIAL_CASH),
            fee_rate=FEE_RATE,
            slippage_rate=SLIPPAGE_RATE,
            rebalance_tolerance=REBALANCE_TOLERANCE,
            minimum_trade_notional=MINIMUM_TRADE_NOTIONAL,
        )
        runs.append(
            PolicyRun(
                policy_name=policy_name,
                targets=targets,
                result=result,
                summary=summarize_backtest(result),
            )
        )
    return tuple(runs)


def _metadata_candidate(input_path: Path) -> Path:
    return input_path.parent.parent / "metadata" / f"{input_path.stem}_metadata.csv"


def _input_provenance(
    input_path: Path, *, input_sha256: str, symbol: str
) -> tuple[str, str]:
    candidate = _metadata_candidate(input_path)
    if not candidate.is_file():
        unknown = "unknown_not_in_repository_metadata"
        return unknown, unknown
    with candidate.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise DemoError(f"{candidate}: expected exactly one metadata row")
    row = rows[0]
    checks = {
        "processed_filename": input_path.name,
        "processed_sha256": input_sha256,
        "symbol": symbol,
    }
    for field, expected in checks.items():
        if row.get(field) != expected:
            raise DemoError(
                f"{candidate}: {field} does not match input; "
                f"expected {expected!r}, got {row.get(field)!r}"
            )
    return (
        row.get("source") or "unknown_not_in_repository_metadata",
        row.get("retrieval_date_utc") or "unknown_not_in_repository_metadata",
    )


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_rows(
    *, experiment_id: str, symbol: str, bars: tuple[Bar, ...], policies: tuple[PolicyRun, ...]
) -> list[dict[str, object]]:
    rows = []
    for policy in policies:
        summary = policy.summary
        rows.append({
            "experiment_id": experiment_id,
            "symbol": symbol,
            "policy_name": policy.policy_name,
            "period_label": PERIOD_LABEL,
            "start_timestamp": _iso_utc(bars[0].timestamp),
            "end_timestamp": _iso_utc(bars[-1].timestamp),
            "bar_count": len(bars),
            "initial_portfolio_value": summary.initial_portfolio_value,
            "final_portfolio_value": summary.final_portfolio_value,
            "cumulative_return": summary.cumulative_return,
            "maximum_drawdown": summary.max_drawdown,
            "turnover": summary.turnover,
            "average_realized_exposure": summary.average_exposure,
            "total_fees": summary.total_fees,
            "buy_count": summary.buy_count,
            "sell_count": summary.sell_count,
            "trade_count": summary.trade_count,
            "final_cash": policy.result.final_state.cash,
            "final_position_quantity": policy.result.final_state.position_quantity,
            "final_target_unexecuted": policy.result.unexecuted_final_target is not None,
        })
    return rows


def _trade_rows(
    *, experiment_id: str, symbol: str, policies: tuple[PolicyRun, ...]
) -> list[dict[str, object]]:
    rows = []
    for policy in policies:
        for fill_index, fill in enumerate(policy.result.fills):
            rows.append({
                "experiment_id": experiment_id,
                "symbol": symbol,
                "policy_name": policy.policy_name,
                "fill_index": fill_index,
                "side": fill.side.value,
                "order_created_at": _iso_utc(fill.order_created_at),
                "executed_at": _iso_utc(fill.executed_at),
                "reference_price": fill.reference_price,
                "fill_price": fill.fill_price,
                "quantity": fill.quantity,
                "notional": fill.notional,
                "fee": fill.fee,
                "cash_flow": fill.cash_flow,
            })
    return rows


def _portfolio_rows(
    *, experiment_id: str, symbol: str, bars: tuple[Bar, ...], policies: tuple[PolicyRun, ...]
) -> list[dict[str, object]]:
    rows = []
    for policy in policies:
        for bar_index, (bar, snapshot) in enumerate(
            zip(bars, policy.result.portfolio_history, strict=True)
        ):
            exposure = (
                snapshot.position_quantity * snapshot.close_price
                / snapshot.portfolio_value
            )
            rows.append({
                "experiment_id": experiment_id,
                "symbol": symbol,
                "policy_name": policy.policy_name,
                "bar_index": bar_index,
                "timestamp": _iso_utc(snapshot.bar_timestamp),
                "open": bar.open,
                "close": snapshot.close_price,
                "cash": snapshot.cash,
                "position_quantity": snapshot.position_quantity,
                "cumulative_fees": snapshot.cumulative_fees,
                "portfolio_value": snapshot.portfolio_value,
                "realized_exposure": exposure,
            })
    return rows


def _target_rows(
    *, experiment_id: str, symbol: str, bars: tuple[Bar, ...], policies: tuple[PolicyRun, ...]
) -> list[dict[str, object]]:
    rows = []
    final_index = len(bars) - 1
    for policy in policies:
        for bar_index, (bar, target) in enumerate(
            zip(bars, policy.targets, strict=True)
        ):
            is_final = bar_index == final_index
            rows.append({
                "experiment_id": experiment_id,
                "symbol": symbol,
                "policy_name": policy.policy_name,
                "bar_index": bar_index,
                "decision_timestamp": _iso_utc(bar.timestamp),
                "target_weight": target.weight,
                "execution_eligible_timestamp": (
                    "" if is_final else _iso_utc(bars[bar_index + 1].timestamp)
                ),
                "is_final_unexecuted_target": is_final,
            })
    return rows


def run_demo(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    symbol: str,
    generated_at_utc: str | None = None,
) -> DemoRun:
    input_argument = Path(input_path).as_posix()
    source_path = Path(input_path)
    destination = Path(output_dir)
    symbol_label = _sanitize_symbol(symbol)
    input_sha256 = _sha256_file(source_path)
    try:
        bars = tuple(load_bars_csv(source_path))
    except (BarDataError, OSError) as error:
        raise DemoError(f"input validation failed: {error}") from error
    _validate_bars(bars)
    original_bars = tuple(bars)
    start = bars[0].timestamp
    end = bars[-1].timestamp
    experiment_id = (
        f"rule_based_real_market_{symbol_label}_{start.date().isoformat()}_"
        f"{end.date().isoformat()}_{input_sha256[:12]}"
    )
    data_source, retrieval_date = _input_provenance(
        source_path, input_sha256=input_sha256, symbol=symbol_label
    )
    policies = _run_policies(bars)
    if bars != original_bars:
        raise DemoError("input bars were mutated during demonstration")

    destination.mkdir(parents=True, exist_ok=True)
    summary_rows = _summary_rows(
        experiment_id=experiment_id, symbol=symbol_label, bars=bars, policies=policies
    )
    trade_rows = _trade_rows(
        experiment_id=experiment_id, symbol=symbol_label, policies=policies
    )
    portfolio_rows = _portfolio_rows(
        experiment_id=experiment_id, symbol=symbol_label, bars=bars, policies=policies
    )
    target_rows = _target_rows(
        experiment_id=experiment_id, symbol=symbol_label, bars=bars, policies=policies
    )
    generated = generated_at_utc or _iso_utc(
        datetime.now(timezone.utc).replace(microsecond=0)
    )
    metadata_rows = [{
        "experiment_id": experiment_id,
        "generated_at_utc": generated,
        "input_path_argument": input_argument,
        "input_filename": source_path.name,
        "input_sha256": input_sha256,
        "symbol": symbol_label,
        "period_label": PERIOD_LABEL,
        "start_timestamp": _iso_utc(start),
        "end_timestamp": _iso_utc(end),
        "bar_count": len(bars),
        "bar_interval": "1h",
        "timestamp_semantics": "bar_open_time",
        "data_source": data_source,
        "retrieval_date": retrieval_date,
        "rule_definition": (
            "first=0.00;close_up=0.75;close_equal=0.50;close_down=0.25"
        ),
        "buy_and_hold_definition": "target[0]=0.0;target[1:]=1.0",
        "zero_position_definition": "all_target_weights=0.0",
        "initial_cash": INITIAL_CASH,
        "fee_rate": FEE_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "rebalance_tolerance": REBALANCE_TOLERANCE,
        "minimum_trade_notional": MINIMUM_TRADE_NOTIONAL,
        "execution_timing": "decision_after_bar_close_execute_next_bar_open",
        "parameter_selection_performed": False,
        "performance_based_adjustment_performed": False,
        "future_ml_test_status": (
            "inspected_rule_based_demonstration_period_not_eligible_as_"
            "untouched_future_ml_final_test"
        ),
        "limitations": (
            "single_asset_single_period_rule_based_execution_demonstration;"
            "not_signal_validation_not_alpha_not_production"
        ),
    }]
    output_specs = (
        ("summary.csv", SUMMARY_COLUMNS, summary_rows),
        ("trades.csv", TRADES_COLUMNS, trade_rows),
        ("portfolio_history.csv", PORTFOLIO_COLUMNS, portfolio_rows),
        ("targets.csv", TARGETS_COLUMNS, target_rows),
        ("metadata.csv", METADATA_COLUMNS, metadata_rows),
    )
    for filename, columns, rows in output_specs:
        _write_csv(destination / filename, columns, rows)
    return DemoRun(
        experiment_id=experiment_id,
        symbol=symbol_label,
        input_sha256=input_sha256,
        bars=bars,
        policies=policies,
        output_paths=tuple(destination / name for name in OUTPUT_FILENAMES),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--symbol", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        run = run_demo(
            input_path=arguments.input,
            output_dir=arguments.output_dir,
            symbol=arguments.symbol,
        )
    except (DemoError, OSError, ValueError) as error:
        print(f"real-market demonstration failed: {error}", file=sys.stderr)
        return 1
    print(
        f"demonstration complete: experiment_id={run.experiment_id} "
        f"bars={len(run.bars)} policies={len(run.policies)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
