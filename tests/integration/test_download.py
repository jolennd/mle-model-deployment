"""Download step: saves one source file per partition into the staging folder.

Every test passes a fake fetch, so nothing touches the network. The fake writes a tiny
parquet file (or bad bytes, or fails) to the path it is given.
"""

import importlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxi_duration import config

pytestmark = pytest.mark.integration

PARTITION = "2025-01"
FINAL_NAME = f"yellow_tripdata_{PARTITION}.parquet"


def _download_module():
    return importlib.import_module("taxi_duration.ingest.download")


def _write_parquet(dest, rows=3):
    table = pa.table({"trip_distance": [1.5 * (i + 1) for i in range(rows)]})
    pq.write_table(table, dest)


class RecordingFetch:
    """Fake fetch that records calls and writes a tiny valid parquet file."""

    def __init__(self, rows=3):
        self.rows = rows
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append((url, dest))
        _write_parquet(dest, self.rows)


def test_download_writes_staged_files(tmp_path):
    download = _download_module()
    fetch = RecordingFetch(rows=3)

    result = download.download_partition(PARTITION, fetch=fetch, staging_dir=tmp_path)

    assert result == tmp_path / FINAL_NAME
    assert pq.read_table(result).num_rows == 3
    assert [p.name for p in tmp_path.iterdir()] == [FINAL_NAME], "stray file left"
    assert len(fetch.calls) == 1
    _, dest = fetch.calls[0]
    assert dest.parent == tmp_path
    assert dest.name != FINAL_NAME, "fetch must write to a temporary name first"


def test_download_failure_is_loud(tmp_path):
    download = _download_module()

    def failing_fetch(url, dest):
        dest.write_bytes(b"partial")  # a half-written file must not survive
        raise ConnectionError("source unreachable")

    with pytest.raises(ConnectionError):
        download.download_partition(
            PARTITION, fetch=failing_fetch, staging_dir=tmp_path
        )

    assert list(tmp_path.iterdir()) == [], "failed download left a file behind"


def test_download_skips_existing_unless_forced(tmp_path):
    download = _download_module()
    first = RecordingFetch(rows=3)
    download.download_partition(PARTITION, fetch=first, staging_dir=tmp_path)

    second = RecordingFetch(rows=5)
    result = download.download_partition(PARTITION, fetch=second, staging_dir=tmp_path)
    assert second.calls == [], "existing file must be skipped"
    assert result == tmp_path / FINAL_NAME
    assert pq.read_table(result).num_rows == 3

    forced = RecordingFetch(rows=5)
    result = download.download_partition(
        PARTITION, force=True, fetch=forced, staging_dir=tmp_path
    )
    assert len(forced.calls) == 1
    assert pq.read_table(result).num_rows == 5
    assert [p.name for p in tmp_path.iterdir()] == [FINAL_NAME]


def test_download_unreadable_file_is_loud(tmp_path):
    download = _download_module()

    def bad_fetch(url, dest):
        dest.write_bytes(b"this is not a parquet file")

    with pytest.raises(Exception):  # noqa: B017 - the reader's error type is not fixed
        download.download_partition(PARTITION, fetch=bad_fetch, staging_dir=tmp_path)

    assert list(tmp_path.iterdir()) == [], "unreadable file left behind"


def test_download_urls_built_from_config(tmp_path):
    download = _download_module()
    fetch = RecordingFetch()

    download.download_partition(PARTITION, fetch=fetch, staging_dir=tmp_path)

    url, _ = fetch.calls[0]
    assert url.startswith(config.DOWNLOAD_BASE_URL)
    assert url.endswith(config.DOWNLOAD_FILE_PATTERN.format(partition=PARTITION))
    assert url == (
        f"{config.DOWNLOAD_BASE_URL}/"
        f"{config.DOWNLOAD_FILE_PATTERN.format(partition=PARTITION)}"
    )

    unknown = "1999-12"
    assert unknown not in config.PARTITIONS
    refused = RecordingFetch()
    with pytest.raises(ValueError):
        download.download_partition(unknown, fetch=refused, staging_dir=tmp_path)
    assert refused.calls == [], "unknown partition must not reach the network"
