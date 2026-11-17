"""Recording a staged file's hash: one command writes the sha256sum line.

Uses tiny temporary files and explicit paths. No real data, no network.
"""

import hashlib
import importlib

import pytest

from taxi_duration import config

pytestmark = pytest.mark.integration

PARTITION = "2025-01"
NAME = config.DOWNLOAD_FILE_PATTERN.format(partition=PARTITION)
CONTENT = b"synthetic bytes, seed-free and fixed"


def _module():
    return importlib.import_module("taxi_duration.ingest.source_hash")


def _digest(data=CONTENT):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def staging(tmp_path):
    folder = tmp_path / "staging"
    folder.mkdir()
    (folder / NAME).write_bytes(CONTENT)
    return folder


@pytest.fixture
def record(tmp_path):
    return tmp_path / "reference" / "source_sha256.txt"


def test_record_writes_sha256sum_line_and_returns_hash(staging, record):
    mod = _module()

    result = mod.record_partition_hash(
        PARTITION, staging_dir=staging, record_path=record
    )

    assert result == _digest()
    assert record.read_text() == f"{_digest()}  {NAME}\n"

    with pytest.raises(ValueError):
        mod.record_partition_hash(
            PARTITION, staging_dir=staging / "nowhere", record_path=record
        )

    unknown = "1999-12"
    assert unknown not in config.PARTITIONS
    with pytest.raises(ValueError):
        mod.record_partition_hash(unknown, staging_dir=staging, record_path=record)
    assert record.read_text() == f"{_digest()}  {NAME}\n"


def test_record_prints_the_hash(staging, record, capsys):
    mod = _module()

    mod.main([PARTITION, "--staging-dir", str(staging), "--record", str(record)])

    out = capsys.readouterr().out
    assert _digest() in out
    assert NAME in out
    assert record.read_text() == f"{_digest()}  {NAME}\n"


def test_record_refuses_different_existing_hash(staging, record):
    mod = _module()
    record.parent.mkdir(parents=True)
    original = f"{_digest(b'something else')}  {NAME}\n"
    record.write_text(original)

    with pytest.raises(mod.SourceHashError):
        mod.record_partition_hash(PARTITION, staging_dir=staging, record_path=record)

    assert record.read_text() == original


def test_record_same_hash_twice_is_idempotent(staging, record):
    mod = _module()

    first = mod.record_partition_hash(
        PARTITION, staging_dir=staging, record_path=record
    )
    second = mod.record_partition_hash(
        PARTITION, staging_dir=staging, record_path=record
    )

    assert first == second == _digest()
    assert record.read_text().splitlines() == [f"{_digest()}  {NAME}"]
