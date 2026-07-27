import csv
import hashlib
import inspect
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backtest.data import load_bars_csv
from scripts import prepare_binance_snapshot as snapshot

RETRIEVED = "2026-07-27T12:00:00Z"


def hourly_timestamps(
    *, start: datetime = snapshot.EXPECTED_START, count: int = snapshot.EXPECTED_ROW_COUNT
) -> list[datetime]:
    return [start + timedelta(hours=index) for index in range(count)]


def kline_row(timestamp: datetime, *, invalid_ohlc: bool = False) -> list[str]:
    milliseconds = int(timestamp.timestamp() * 1000)
    high = "99" if invalid_ohlc else "103"
    return [
        str(milliseconds), "100", high, "99", "102", "10",
        str(milliseconds + 3_599_999), "1000", "5", "4", "400", "0",
    ]


def write_archives(
    raw_dir: Path,
    timestamps: list[datetime],
    *,
    invalid_ohlc_index: int | None = None,
) -> tuple[Path, ...]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    by_month = {month: [] for month in snapshot.MONTHS}
    for index, timestamp in enumerate(timestamps):
        month = timestamp.strftime("%Y-%m")
        if month in by_month:
            by_month[month].append(
                kline_row(timestamp, invalid_ohlc=index == invalid_ohlc_index)
            )
        else:
            by_month[snapshot.MONTHS[-1]].append(kline_row(timestamp))

    paths = []
    for month in snapshot.MONTHS:
        path = raw_dir / snapshot.archive_filename(month)
        member = f"BTCUSDT-1h-{month}.csv"
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(by_month[month])
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            member_info = zipfile.ZipInfo(member, date_time=(2024, 1, 1, 0, 0, 0))
            member_info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member_info, buffer.getvalue())
        paths.append(path)
    return tuple(paths)


def process_fixture(tmp_path: Path, timestamps: list[datetime] | None = None):
    raw_dir = tmp_path / "raw"
    write_archives(raw_dir, timestamps or hourly_timestamps())
    processed = tmp_path / "processed" / "BTCUSDT_1h_2024.csv"
    metadata = tmp_path / "metadata" / "BTCUSDT_1h_2024_metadata.csv"
    row = snapshot.process_archives(
        raw_dir=raw_dir,
        processed_output=processed,
        metadata_output=metadata,
        retrieval_date_utc=RETRIEVED,
    )
    return raw_dir, processed, metadata, row


def test_exact_frozen_months_filenames_and_urls() -> None:
    assert snapshot.MONTHS == tuple(f"2024-{month:02d}" for month in range(1, 13))
    assert snapshot.archive_urls() == tuple(
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"BTCUSDT/1h/BTCUSDT-1h-2024-{month:02d}.zip"
        for month in range(1, 13)
    )
    with pytest.raises(snapshot.SnapshotPreparationError, match="outside frozen"):
        snapshot.archive_url("2023-12")


def test_timestamp_conversion_and_exact_column_mapping() -> None:
    timestamp = snapshot._timestamp_from_milliseconds(
        "1704067200000", "fixture"
    )
    assert timestamp == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert timestamp.tzinfo is timezone.utc


def test_valid_snapshot_is_deterministic_loader_compatible_and_exactly_described(
    tmp_path: Path,
) -> None:
    raw_dir, processed, metadata, row = process_fixture(tmp_path / "first")
    _, repeated, repeated_metadata, repeated_row = process_fixture(tmp_path / "second")

    assert processed.read_bytes() == repeated.read_bytes()
    assert metadata.read_bytes() == repeated_metadata.read_bytes()
    assert row == repeated_row
    bars = load_bars_csv(processed)
    assert len(bars) == snapshot.EXPECTED_ROW_COUNT
    assert bars[0].timestamp == snapshot.EXPECTED_START
    assert bars[-1].timestamp == snapshot.EXPECTED_END
    assert processed.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        snapshot.PROCESSED_COLUMNS
    )
    assert row["processed_sha256"] == snapshot.sha256_file(processed)
    assert row["actual_row_count"] == 8784
    assert row["duplicate_timestamp_count"] == 0
    assert row["missing_timestamp_count"] == 0
    assert row["sorting_performed"] is False
    assert row["filling_performed"] is False
    assert row["interpolation_performed"] is False
    assert row["deduplication_performed"] is False
    assert row["repair_performed"] is False
    assert row["strategy_results_inspected"] is False
    with metadata.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == snapshot.METADATA_COLUMNS
        assert len(list(reader)) == 1
    assert tuple(path.name for path in raw_dir.glob("*.zip")) == tuple(
        snapshot.archive_filename(month) for month in snapshot.MONTHS
    )


def test_download_uses_only_frozen_urls_and_preserves_response_bytes(
    tmp_path: Path,
) -> None:
    requested: list[str] = []
    payloads = {
        url: f"unaltered-{index}".encode()
        for index, url in enumerate(snapshot.archive_urls())
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def opener(url: str):
        requested.append(url)
        return Response(payloads[url])

    paths = snapshot.download_archives(tmp_path, opener=opener)
    assert tuple(requested) == snapshot.archive_urls()
    assert len(paths) == 12
    for url, path in zip(requested, paths, strict=True):
        assert path.read_bytes() == payloads[url]
        assert snapshot.sha256_file(path) == hashlib.sha256(payloads[url]).hexdigest()


def test_duplicate_input_is_rejected_without_sorting_or_repair(tmp_path: Path) -> None:
    timestamps = hourly_timestamps()
    timestamps[100] = timestamps[99]
    raw_dir = tmp_path / "raw"
    paths = write_archives(raw_dir, timestamps)
    originals = tuple(path.read_bytes() for path in paths)

    with pytest.raises(snapshot.SnapshotPreparationError, match="duplicate timestamp"):
        snapshot.process_archives(
            raw_dir=raw_dir,
            processed_output=tmp_path / "processed.csv",
            metadata_output=tmp_path / "metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )
    assert tuple(path.read_bytes() for path in paths) == originals
    assert not (tmp_path / "processed.csv").exists()


def test_non_chronological_input_is_rejected_not_sorted(tmp_path: Path) -> None:
    timestamps = hourly_timestamps()
    timestamps[100] = timestamps[99] - timedelta(minutes=30)
    raw_dir = tmp_path / "raw"
    write_archives(raw_dir, timestamps)
    with pytest.raises(snapshot.SnapshotPreparationError, match="not strictly chronological"):
        snapshot.process_archives(
            raw_dir=raw_dir,
            processed_output=tmp_path / "processed.csv",
            metadata_output=tmp_path / "metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )


def test_missing_hour_is_rejected_without_filling(tmp_path: Path) -> None:
    timestamps = hourly_timestamps()
    del timestamps[100]
    timestamps.append(snapshot.EXPECTED_END + timedelta(hours=1))
    raw_dir = tmp_path / "raw"
    write_archives(raw_dir, timestamps)
    with pytest.raises(snapshot.SnapshotPreparationError, match="not exactly one hour"):
        snapshot.process_archives(
            raw_dir=raw_dir,
            processed_output=tmp_path / "processed.csv",
            metadata_output=tmp_path / "metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )


def test_wrong_first_last_and_row_count_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shifted = hourly_timestamps(start=snapshot.EXPECTED_START + timedelta(hours=1))
    raw_dir = tmp_path / "shifted" / "raw"
    write_archives(raw_dir, shifted)
    with pytest.raises(snapshot.SnapshotPreparationError, match="first timestamp"):
        snapshot.process_archives(
            raw_dir=raw_dir,
            processed_output=tmp_path / "shifted.csv",
            metadata_output=tmp_path / "shifted_metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )

    valid_dir = tmp_path / "wrong_last" / "raw"
    write_archives(valid_dir, hourly_timestamps())
    monkeypatch.setattr(
        snapshot, "EXPECTED_END", snapshot.EXPECTED_END + timedelta(hours=1)
    )
    with pytest.raises(snapshot.SnapshotPreparationError, match="last timestamp"):
        snapshot.process_archives(
            raw_dir=valid_dir,
            processed_output=tmp_path / "wrong_last.csv",
            metadata_output=tmp_path / "wrong_last_metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )
    monkeypatch.undo()

    short_dir = tmp_path / "short" / "raw"
    write_archives(short_dir, hourly_timestamps(count=8783))
    with pytest.raises(snapshot.SnapshotPreparationError, match="row count"):
        snapshot.process_archives(
            raw_dir=short_dir,
            processed_output=tmp_path / "short.csv",
            metadata_output=tmp_path / "short_metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )


def test_missing_archive_and_invalid_ohlc_are_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "missing" / "raw"
    paths = write_archives(raw_dir, hourly_timestamps())
    paths[-1].unlink()
    with pytest.raises(snapshot.SnapshotPreparationError, match="missing monthly archives"):
        snapshot.process_archives(
            raw_dir=raw_dir,
            processed_output=tmp_path / "missing.csv",
            metadata_output=tmp_path / "missing_metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )

    invalid_dir = tmp_path / "invalid" / "raw"
    write_archives(invalid_dir, hourly_timestamps(), invalid_ohlc_index=0)
    with pytest.raises(snapshot.SnapshotPreparationError, match="production loader rejected"):
        snapshot.process_archives(
            raw_dir=invalid_dir,
            processed_output=tmp_path / "invalid.csv",
            metadata_output=tmp_path / "invalid_metadata.csv",
            retrieval_date_utc=RETRIEVED,
        )


def test_script_contains_no_strategy_backtest_or_repair_logic() -> None:
    source = inspect.getsource(snapshot).lower()
    assert "backtest.strategy" not in source
    assert "run_backtest" not in source
    assert "run_target_weight_backtest" not in source
    for forbidden in ("sorted(", ".sort(", "interpolate", "forward_fill", "drop_duplicates"):
        assert forbidden not in source
