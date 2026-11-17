"""Source file hash check. The record uses sha256sum format: `<64 hex>  <file name>`."""

import hashlib
from pathlib import Path

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
