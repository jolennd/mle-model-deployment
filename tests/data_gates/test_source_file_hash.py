"""Data gate: the downloaded source file must match the recorded SHA-256.

Needs the real file in the staging folder, so it runs locally after a download and
skips (with a reason) when the file or the record is missing.
"""

import hashlib
from pathlib import Path

import pytest

from taxi_duration import config

pytestmark = pytest.mark.data

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "tests" / "reference" / "source_sha256.txt"


def _recorded_hashes():
    hashes = {}
    for line in RECORD.read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            hashes[name.strip()] = digest.lower()
    return hashes


@pytest.mark.parametrize("partition", config.PARTITIONS)
def test_source_file_sha256_matches_record(partition):
    name = config.DOWNLOAD_FILE_PATTERN.format(partition=partition)
    source = ROOT / config.STAGING_DIR / name
    if not RECORD.exists():
        pytest.skip(
            f"no hash record yet at {RECORD.relative_to(ROOT)} (first download)"
        )
    if not source.exists():
        pytest.skip(f"{source.relative_to(ROOT)} is not downloaded yet")

    recorded = _recorded_hashes()
    assert name in recorded, f"{name} has no entry in {RECORD.name}"
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == recorded[name], f"{name} changed since the hash was recorded"
