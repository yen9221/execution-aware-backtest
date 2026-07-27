import csv
import hashlib
import inspect
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtest.data import load_bars_csv
from scripts import run_rule_based_real_market_demo as demo

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def write_bars(path: Path, *, timestamps: list[datetime] | None = None) -> Path:
    source_timestamps = timestamps or [START + timedelta(hours=i) for i in range(6)]
    opens = (100.0, 101.0, 103.0, 36.0, 104.0, 102.0)
    closes = (100.0, 102.0, 102.0, 99.0, 101.0, 98.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
        for timestamp, open_price, close_price in zip(
            source_timestamps, opens, closes, strict=True
        ):
            writer.writerow((
                timestamp.isoformat().replace("+00:00", "Z"),
                open_price,
                max(open_price, close_price) + 1.0,
                min(open_price, close_price) - 1.0,
                close_price,
                10.0,
            ))
    return path


def read_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        return tuple(reader.fieldnames or ()), list(reader)


def run_fixture(tmp_path: Path, *, generated: str = "2026-07-27T14:00:00Z"):
    input_path = write_bars(tmp_path / "processed" / "bars.csv")
    original = input_path.read_bytes()
    result = demo.run_demo(
        input_path=input_path,
        output_dir=tmp_path / "output",
        symbol="BTCUSDT",
        generated_at_utc=generated,
    )
    assert input_path.read_bytes() == original
    return input_path, result


def test_runner_uses_production_loader_and_exact_fixed_target_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = write_bars(tmp_path / "processed" / "bars.csv")
    calls: list[Path] = []
    production_loader = load_bars_csv

    def recording_loader(path):
        calls.append(Path(path))
        return production_loader(path)

    monkeypatch.setattr(demo, "load_bars_csv", recording_loader)
    run = demo.run_demo(
        input_path=input_path,
        output_dir=tmp_path / "output",
        symbol="BTCUSDT",
        generated_at_utc="2026-07-27T14:00:00Z",
    )
    assert calls == [input_path]
    assert tuple(policy.policy_name for policy in run.policies) == demo.POLICY_NAMES
    assert tuple(target.weight for target in run.policies[0].targets) == (
        0.0, 0.75, 0.50, 0.25, 0.75, 0.25
    )
    assert tuple(target.weight for target in run.policies[1].targets) == (
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0
    )
    assert tuple(target.weight for target in run.policies[2].targets) == (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )


def test_next_bar_timing_aligned_first_entry_and_final_targets(tmp_path: Path) -> None:
    _, run = run_fixture(tmp_path)
    first_bar = run.bars[0].timestamp
    expected_first_entry = run.bars[2].timestamp
    for policy in run.policies:
        assert all(
            fill.executed_at == fill.order_created_at + timedelta(hours=1)
            for fill in policy.result.fills
        )
        assert all(
            fill.executed_at != fill.order_created_at
            for fill in policy.result.fills
        )
        assert all(fill.executed_at != first_bar for fill in policy.result.fills)
        assert policy.result.unexecuted_final_target is not None
        assert policy.result.unexecuted_final_target.target == policy.targets[-1]
    assert run.policies[0].result.fills[0].executed_at == expected_first_entry
    assert run.policies[1].result.fills[0].executed_at == expected_first_entry
    assert run.policies[2].result.fills == ()


def test_output_schemas_row_counts_and_actual_fill_only_trade_rows(tmp_path: Path) -> None:
    _, run = run_fixture(tmp_path)
    output = tmp_path / "output"
    summary_header, summary = read_rows(output / "summary.csv")
    trades_header, trades = read_rows(output / "trades.csv")
    portfolio_header, portfolio = read_rows(output / "portfolio_history.csv")
    targets_header, targets = read_rows(output / "targets.csv")
    metadata_header, metadata = read_rows(output / "metadata.csv")

    assert summary_header == demo.SUMMARY_COLUMNS
    assert trades_header == demo.TRADES_COLUMNS
    assert portfolio_header == demo.PORTFOLIO_COLUMNS
    assert targets_header == demo.TARGETS_COLUMNS
    assert metadata_header == demo.METADATA_COLUMNS
    assert [row["policy_name"] for row in summary] == list(demo.POLICY_NAMES)
    assert len(summary) == 3
    assert len(trades) == sum(len(policy.result.fills) for policy in run.policies)
    assert len(portfolio) == 3 * len(run.bars)
    assert len(targets) == 3 * len(run.bars)
    assert len(metadata) == 1
    assert not any(row["policy_name"] == "zero_position" for row in trades)
    assert tuple(path.name for path in run.output_paths) == demo.OUTPUT_FILENAMES


def test_summary_reconciles_to_fills_history_and_reporting(tmp_path: Path) -> None:
    _, run = run_fixture(tmp_path)
    output = tmp_path / "output"
    _, summary_rows = read_rows(output / "summary.csv")
    _, trade_rows = read_rows(output / "trades.csv")
    _, portfolio_rows = read_rows(output / "portfolio_history.csv")

    for policy, row in zip(run.policies, summary_rows, strict=True):
        policy_trades = [
            item for item in trade_rows if item["policy_name"] == policy.policy_name
        ]
        policy_history = [
            item for item in portfolio_rows if item["policy_name"] == policy.policy_name
        ]
        assert int(row["trade_count"]) == len(policy.result.fills) == len(policy_trades)
        assert float(row["total_fees"]) == pytest.approx(
            math.fsum(float(item["fee"]) for item in policy_trades)
        )
        assert float(row["turnover"]) == pytest.approx(
            math.fsum(float(item["notional"]) for item in policy_trades)
            / float(row["initial_portfolio_value"])
        )
        assert float(row["average_realized_exposure"]) == pytest.approx(
            math.fsum(float(item["realized_exposure"]) for item in policy_history)
            / len(policy_history)
        )
        assert float(row["initial_portfolio_value"]) == pytest.approx(
            policy.summary.initial_portfolio_value
        )
        assert float(row["final_portfolio_value"]) == pytest.approx(
            float(policy_history[-1]["portfolio_value"])
        )
        assert float(row["cumulative_return"]) == pytest.approx(
            policy.summary.cumulative_return
        )
        assert float(row["maximum_drawdown"]) == pytest.approx(
            policy.summary.max_drawdown
        )


def test_zero_position_baseline_reconciles_exactly(tmp_path: Path) -> None:
    _, run = run_fixture(tmp_path)
    zero = run.policies[2]
    assert zero.result.fills == ()
    assert zero.summary.trade_count == 0
    assert zero.summary.turnover == 0.0
    assert zero.summary.total_fees == 0.0
    assert zero.summary.average_exposure == 0.0
    assert zero.summary.max_drawdown == 0.0
    assert zero.summary.cumulative_return == 0.0
    assert zero.summary.initial_portfolio_value == 1000.0
    assert zero.summary.final_portfolio_value == 1000.0
    assert zero.result.final_state.cash == 1000.0
    assert zero.result.final_state.position_quantity == 0.0


def test_target_rows_record_eligibility_not_execution(tmp_path: Path) -> None:
    _, run = run_fixture(tmp_path)
    _, rows = read_rows(tmp_path / "output" / "targets.csv")
    for policy in run.policies:
        policy_rows = [row for row in rows if row["policy_name"] == policy.policy_name]
        assert len(policy_rows) == len(run.bars)
        for index, row in enumerate(policy_rows):
            assert int(row["bar_index"]) == index
            assert float(row["target_weight"]) == policy.targets[index].weight
            if index == len(run.bars) - 1:
                assert row["execution_eligible_timestamp"] == ""
                assert row["is_final_unexecuted_target"] == "True"
            else:
                assert row["execution_eligible_timestamp"] == demo._iso_utc(
                    run.bars[index + 1].timestamp
                )
                assert row["is_final_unexecuted_target"] == "False"


def test_metadata_provenance_hash_and_determinism_except_generation_time(
    tmp_path: Path,
) -> None:
    input_path = write_bars(tmp_path / "processed" / "bars.csv")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    metadata_source = tmp_path / "metadata" / "bars_metadata.csv"
    metadata_source.parent.mkdir(parents=True)
    with metadata_source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "source", "retrieval_date_utc", "processed_filename",
                "processed_sha256", "symbol",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({
            "source": "Fixture Source",
            "retrieval_date_utc": "2024-02-01T00:00:00Z",
            "processed_filename": input_path.name,
            "processed_sha256": input_hash,
            "symbol": "BTCUSDT",
        })
    first = demo.run_demo(
        input_path=input_path,
        output_dir=tmp_path / "first",
        symbol="BTCUSDT",
        generated_at_utc="2026-07-27T14:00:00Z",
    )
    demo.run_demo(
        input_path=input_path,
        output_dir=tmp_path / "second",
        symbol="BTCUSDT",
        generated_at_utc="2026-07-27T15:00:00Z",
    )
    assert first.input_sha256 == input_hash
    for filename in demo.OUTPUT_FILENAMES[:-1]:
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    _, first_metadata = read_rows(tmp_path / "first" / "metadata.csv")
    _, second_metadata = read_rows(tmp_path / "second" / "metadata.csv")
    assert first_metadata[0]["generated_at_utc"] != second_metadata[0]["generated_at_utc"]
    first_metadata[0].pop("generated_at_utc")
    second_metadata[0].pop("generated_at_utc")
    assert first_metadata == second_metadata
    assert first_metadata[0]["data_source"] == "Fixture Source"
    assert first_metadata[0]["retrieval_date"] == "2024-02-01T00:00:00Z"
    assert first_metadata[0]["parameter_selection_performed"] == "False"
    assert first_metadata[0]["performance_based_adjustment_performed"] == "False"


def test_invalid_csv_is_rejected_without_output_or_repair(tmp_path: Path) -> None:
    timestamps = [START + timedelta(hours=index) for index in (0, 1, 3, 4, 5, 6)]
    input_path = write_bars(tmp_path / "processed" / "bars.csv", timestamps=timestamps)
    original = input_path.read_bytes()
    with pytest.raises(demo.DemoError, match="input validation failed.*one hour"):
        demo.run_demo(
            input_path=input_path,
            output_dir=tmp_path / "output",
            symbol="BTCUSDT",
            generated_at_utc="2026-07-27T14:00:00Z",
        )
    assert input_path.read_bytes() == original
    assert not (tmp_path / "output").exists()


def test_runner_has_no_model_optimization_or_alternative_policy_logic() -> None:
    source = inspect.getsource(demo).lower()
    for forbidden in (
        "model.fit", "predict_proba", "feature_generation", "label_generation",
        "threshold_selection",
        "optimize", "max_weight", "volatility_target", "stop_loss", "sharpe",
        "annualized", "download", "urlopen",
    ):
        assert forbidden not in source
