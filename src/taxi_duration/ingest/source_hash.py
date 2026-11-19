"""Source file hash check. The record uses sha256sum format: `<64 hex>  <file name>`."""

import argparse
import hashlib
from pathlib import Path

from .. import config

CHUNK = 1024 * 1024


class SourceHashError(ValueError, FileNotFoundError):
    """The source file does not match its record, or file or record is missing."""

    def __str__(self) -> str:
        return ValueError.__str__(self)


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_record(record_path: Path) -> dict[str, str]:
    if not record_path.is_file():
        raise SourceHashError(f"hash record not found: {record_path}")
    entries: dict[str, str] = {}
    for line in record_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SourceHashError(f"bad line in {record_path.name}: {line!r}")
        entries[parts[1].strip().lstrip("*")] = parts[0].lower()
    return entries


def verify_source_hash(path, record_path) -> None:
    path, record_path = Path(path), Path(record_path)
    if not path.is_file():
        raise SourceHashError(f"source file not found: {path}")
    entries = _read_record(record_path)
    if path.name not in entries:
        raise SourceHashError(f"{path.name} has no entry in {record_path.name}")
    actual = sha256_of(path)
    if actual != entries[path.name]:
        raise SourceHashError(
            f"{path.name} changed: recorded {entries[path.name]}, found {actual}"
        )


def record_source_hash(path, record_path) -> None:
    """Write or update the line for this file name. Never replaces a different hash."""
    path, record_path = Path(path), Path(record_path)
    if not path.is_file():
        raise SourceHashError(f"source file not found: {path}")
    digest = sha256_of(path)
    entries = _read_record(record_path) if record_path.exists() else {}
    if path.name in entries and entries[path.name] != digest:
        raise SourceHashError(
            f"{path.name} already recorded with a different hash "
            f"({entries[path.name]}); refusing to overwrite"
        )
    entries[path.name] = digest
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        "".join(f"{value}  {name}\n" for name, value in entries.items())
    )


def _project_root() -> Path:
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("project root (pyproject.toml) not found")


def record_partition_hash(partition, staging_dir=None, record_path=None) -> str:
    """Hash the staged file of one partition and record it. Returns the hex hash."""
    if partition not in config.PARTITIONS:
        raise ValueError(f"partition {partition!r} not in {config.PARTITIONS}")
    root = _project_root()
    staging = (
        Path(staging_dir) if staging_dir is not None else root / config.STAGING_DIR
    )
    record = (
        Path(record_path)
        if record_path is not None
        else root / "tests" / "reference" / "source_sha256.txt"
    )
    path = staging / config.DOWNLOAD_FILE_PATTERN.format(partition=partition)
    if not path.is_file():
        raise SourceHashError(f"source file not found: {path}")
    record_source_hash(path, record)
    return sha256_of(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record the hash of a staged file.")
    parser.add_argument("partition")
    parser.add_argument("--staging-dir", default=None)
    parser.add_argument("--record", default=None)
    args = parser.parse_args(argv)
    digest = record_partition_hash(
        args.partition, staging_dir=args.staging_dir, record_path=args.record
    )
    name = config.DOWNLOAD_FILE_PATTERN.format(partition=args.partition)
    print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
