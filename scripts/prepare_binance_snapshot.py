"""Prepare the frozen Binance Spot BTCUSDT hourly snapshot for 2024."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

from backtest.data import BarDataError, load_bars_csv

DATASET_ID = "binance_spot_BTCUSDT_1h_2024"
SOURCE = "Binance Public Data"
SOURCE_BASE_URL = (
    "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
)
VENUE = "Binance"
MARKET_TYPE = "spot"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 13))
EXPECTED_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
EXPECTED_END = datetime(2024, 12, 31, 23, tzinfo=timezone.utc)
EXPECTED_ROW_COUNT = 8784
ONE_HOUR = timedelta(hours=1)
PROCESSED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
METADATA_COLUMNS = (
    "dataset_id",
    "source",
    "source_base_url",
    "venue",
    "market_type",
    "symbol",
    "interval",
    "start_timestamp",
    "end_timestamp",
    "expected_row_count",
    "actual_row_count",
    "timestamp_semantics",
    "retrieval_date_utc",
    "raw_archive_names",
    "raw_archive_sha256",
    "processed_filename",
    "processed_sha256",
    "processing_script",
    "duplicate_timestamp_count",
    "missing_timestamp_count",
    "sorting_performed",
    "filling_performed",
    "interpolation_performed",
    "deduplication_performed",
    "repair_performed",
    "strategy_results_inspected",
    "future_ml_test_status",
)


class SnapshotPreparationError(RuntimeError):
    """Raised when the frozen snapshot cannot be prepared exactly."""


def archive_filename(month: str) -> str:
    if month not in MONTHS:
        raise SnapshotPreparationError(f"month is outside frozen specification: {month}")
    return f"BTCUSDT-1h-{month}.zip"


def archive_url(month: str) -> str:
    return f"{SOURCE_BASE_URL}/{archive_filename(month)}"


def archive_urls() -> tuple[str, ...]:
    return tuple(archive_url(month) for month in MONTHS)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_opener(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)  # noqa: S310 - frozen HTTPS URL


def download_archives(
    raw_dir: str | Path,
    *,
    opener: Callable[[str], BinaryIO] = _default_opener,
) -> tuple[Path, ...]:
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for month in MONTHS:
        url = archive_url(month)
        output = destination / archive_filename(month)
        temporary = output.with_suffix(output.suffix + ".part")
        try:
            with opener(url) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            if temporary.stat().st_size == 0:
                raise SnapshotPreparationError(f"downloaded archive is empty: {url}")
            os.replace(temporary, output)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, SnapshotPreparationError):
                raise
            raise SnapshotPreparationError(f"download failed for {url}: {error}") from error
        downloaded.append(output)
    return tuple(downloaded)


def _timestamp_from_milliseconds(raw_value: str, context: str) -> datetime:
    try:
        milliseconds = int(raw_value)
    except ValueError as error:
        raise SnapshotPreparationError(
            f"{context}: open timestamp is not integer milliseconds: {raw_value!r}"
        ) from error
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as error:
        raise SnapshotPreparationError(
            f"{context}: open timestamp is outside supported range: {raw_value!r}"
        ) from error


def _iso_utc(timestamp: datetime) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _archive_rows(archive_path: Path, month: str) -> Iterable[tuple[datetime, ...]]:
    member_name = f"BTCUSDT-1h-{month}.csv"
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            try:
                payload = archive.read(member_name)
            except KeyError as error:
                raise SnapshotPreparationError(
                    f"{archive_path}: missing expected member {member_name}"
                ) from error
    except (OSError, zipfile.BadZipFile) as error:
        raise SnapshotPreparationError(
            f"cannot read archive {archive_path}: {error}"
        ) from error

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SnapshotPreparationError(
            f"{archive_path}: kline CSV is not UTF-8"
        ) from error

    reader = csv.reader(io.StringIO(text, newline=""))
    for row_number, row in enumerate(reader, start=1):
        if not row:
            raise SnapshotPreparationError(
                f"{archive_path}:{row_number}: empty kline row"
            )
        if row_number == 1 and row[0].strip().lower() in {
            "open_time",
            "open time",
        }:
            continue
        if len(row) != 12:
            raise SnapshotPreparationError(
                f"{archive_path}:{row_number}: expected 12 kline fields, got {len(row)}"
            )
        context = f"{archive_path}:{row_number}"
        timestamp = _timestamp_from_milliseconds(row[0].strip(), context)
        values = tuple(value.strip() for value in row[1:6])
        if any(not value for value in values):
            raise SnapshotPreparationError(f"{context}: required OHLCV field is empty")
        yield (timestamp, *values)


def _validate_timestamp_sequence(timestamps: tuple[datetime, ...]) -> None:
    duplicate_count = len(timestamps) - len(set(timestamps))
    if duplicate_count:
        raise SnapshotPreparationError(
            f"duplicate timestamp count must be 0, got {duplicate_count}"
        )
    for index, (previous, current) in enumerate(
        zip(timestamps, timestamps[1:]), start=1
    ):
        if current <= previous:
            raise SnapshotPreparationError(
                f"timestamp at index {index} is not strictly chronological: "
                f"{_iso_utc(current)} after {_iso_utc(previous)}"
            )
        if current - previous != ONE_HOUR:
            raise SnapshotPreparationError(
                f"timestamp at index {index} is not exactly one hour after "
                f"the preceding timestamp: {_iso_utc(current)}"
            )


def _metadata_row(
    *,
    archive_paths: tuple[Path, ...],
    archive_hashes: tuple[str, ...],
    processed_output: Path,
    processed_sha256: str,
    retrieval_date_utc: str,
    actual_row_count: int,
) -> dict[str, object]:
    return {
        "dataset_id": DATASET_ID,
        "source": SOURCE,
        "source_base_url": SOURCE_BASE_URL,
        "venue": VENUE,
        "market_type": MARKET_TYPE,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start_timestamp": _iso_utc(EXPECTED_START),
        "end_timestamp": _iso_utc(EXPECTED_END),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "actual_row_count": actual_row_count,
        "timestamp_semantics": "bar_open_time",
        "retrieval_date_utc": retrieval_date_utc,
        "raw_archive_names": "|".join(path.name for path in archive_paths),
        "raw_archive_sha256": "|".join(
            f"{path.name}:{digest}"
            for path, digest in zip(archive_paths, archive_hashes, strict=True)
        ),
        "processed_filename": processed_output.name,
        "processed_sha256": processed_sha256,
        "processing_script": "scripts/prepare_binance_snapshot.py",
        "duplicate_timestamp_count": 0,
        "missing_timestamp_count": 0,
        "sorting_performed": False,
        "filling_performed": False,
        "interpolation_performed": False,
        "deduplication_performed": False,
        "repair_performed": False,
        "strategy_results_inspected": False,
        "future_ml_test_status": (
            "designated_rule_based_demonstration_period_not_eligible_as_"
            "untouched_ml_final_test"
        ),
    }


def _write_metadata(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=METADATA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def process_archives(
    *,
    raw_dir: str | Path,
    processed_output: str | Path,
    metadata_output: str | Path,
    retrieval_date_utc: str,
) -> dict[str, object]:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_output)
    metadata_path = Path(metadata_output)
    archive_paths = tuple(raw_path / archive_filename(month) for month in MONTHS)
    missing_archives = tuple(path.name for path in archive_paths if not path.is_file())
    if missing_archives:
        raise SnapshotPreparationError(
            "missing monthly archives: " + ", ".join(missing_archives)
        )
    archive_hashes = tuple(sha256_file(path) for path in archive_paths)

    rows: list[tuple[datetime, ...]] = []
    for month, archive_path in zip(MONTHS, archive_paths, strict=True):
        rows.extend(_archive_rows(archive_path, month))
    timestamps = tuple(row[0] for row in rows)
    _validate_timestamp_sequence(timestamps)
    if len(rows) != EXPECTED_ROW_COUNT:
        raise SnapshotPreparationError(
            f"row count must be {EXPECTED_ROW_COUNT}, got {len(rows)}"
        )
    if timestamps[0] != EXPECTED_START:
        raise SnapshotPreparationError(
            f"first timestamp must be {_iso_utc(EXPECTED_START)}, "
            f"got {_iso_utc(timestamps[0])}"
        )
    if timestamps[-1] != EXPECTED_END:
        raise SnapshotPreparationError(
            f"last timestamp must be {_iso_utc(EXPECTED_END)}, "
            f"got {_iso_utc(timestamps[-1])}"
        )

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = processed_path.with_suffix(processed_path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(PROCESSED_COLUMNS)
            for timestamp, *values in rows:
                writer.writerow((_iso_utc(timestamp), *values))
        try:
            bars = load_bars_csv(temporary)
        except (BarDataError, OSError) as error:
            raise SnapshotPreparationError(
                f"production loader rejected processed snapshot: {error}"
            ) from error
        if len(bars) != EXPECTED_ROW_COUNT:
            raise SnapshotPreparationError(
                "production loader returned unexpected row count: "
                f"{len(bars)}"
            )
        os.replace(temporary, processed_path)
    finally:
        temporary.unlink(missing_ok=True)

    processed_sha256 = sha256_file(processed_path)
    metadata = _metadata_row(
        archive_paths=archive_paths,
        archive_hashes=archive_hashes,
        processed_output=processed_path,
        processed_sha256=processed_sha256,
        retrieval_date_utc=retrieval_date_utc,
        actual_row_count=len(rows),
    )
    _write_metadata(metadata_path, metadata)
    return metadata


def prepare_snapshot(
    *,
    raw_dir: str | Path,
    processed_output: str | Path,
    metadata_output: str | Path,
    retrieval_date_utc: str | None = None,
    opener: Callable[[str], BinaryIO] = _default_opener,
) -> dict[str, object]:
    download_archives(raw_dir, opener=opener)
    retrieved = retrieval_date_utc or _iso_utc(
        datetime.now(timezone.utc).replace(microsecond=0)
    )
    return process_archives(
        raw_dir=raw_dir,
        processed_output=processed_output,
        metadata_output=metadata_output,
        retrieval_date_utc=retrieved,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--processed-output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        metadata = prepare_snapshot(
            raw_dir=arguments.raw_dir,
            processed_output=arguments.processed_output,
            metadata_output=arguments.metadata_output,
        )
    except (SnapshotPreparationError, OSError) as error:
        print(f"snapshot preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        "snapshot prepared: "
        f"rows={metadata['actual_row_count']} "
        f"sha256={metadata['processed_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
