"""Source hash check: a changed, missing or unrecorded file is a loud error.

Uses tiny temporary files and a temporary record in sha256sum format
(`<64 hex>  <file name>`). No real data, no network.
"""

import hashlib
import importlib

import pytest

pytestmark = pytest.mark.integration

NAME = "yellow_tripdata_2025-01.parquet"
CONTENT = b"synthetic bytes, seed-free and fixed"


def _module():
    return importlib.import_module("taxi_duration.ingest.source_hash")


def _digest(data=CONTENT):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / NAME
    path.write_bytes(CONTENT)
    return path


def _record(tmp_path, lines):
    record = tmp_path / "source_sha256.txt"
    record.write_text("\n".join(lines) + "\n")
    return record


def test_hash_mismatch_is_loud(source, tmp_path):
    mod = _module()
    assert mod.sha256_of(source) == _digest()

    good = _record(
        tmp_path, [f"{'0' * 64}  other_file.parquet", f"{_digest()}  {NAME}"]
    )
    assert mod.verify_source_hash(source, good) is None

    wrong = _record(tmp_path, [f"{_digest(b'something else')}  {NAME}"])
    with pytest.raises(ValueError):
        mod.verify_source_hash(source, wrong)

    with pytest.raises(ValueError):
        mod.verify_source_hash(source, tmp_path / "no_such_record.txt")

    no_entry = _record(tmp_path, [f"{_digest()}  other_file.parquet"])
    with pytest.raises(ValueError):
        mod.verify_source_hash(source, no_entry)

    with pytest.raises(ValueError):
        mod.verify_source_hash(tmp_path / "missing.parquet", good)
