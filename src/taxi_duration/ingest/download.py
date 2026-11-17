"""Download step: saves one source file per partition into the staging folder.

The file is written to a temporary name first, checked as readable parquet, and only then
renamed. A failed download or an unreadable file leaves nothing behind.
"""

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import requests

from taxi_duration import config

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30
CHUNK_BYTES = 1024 * 1024

Fetch = Callable[[str, Path], None]


def http_fetch(url: str, dest: Path) -> None:
    """Stream url to dest in chunks. https only; certificates are verified."""
    if not url.lower().startswith("https://"):
        raise ValueError(f"refusing non-https url: {url}")
    with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(dest, "wb") as out:
            out.writelines(response.iter_content(chunk_size=CHUNK_BYTES))


def partition_url(partition: str) -> str:
    name = config.DOWNLOAD_FILE_PATTERN.format(partition=partition)
    return f"{config.DOWNLOAD_BASE_URL}/{name}"


def download_partition(
    partition: str,
    force: bool = False,
    fetch: Fetch | None = None,
    staging_dir: Path | str | None = None,
) -> Path:
    if partition not in config.PARTITIONS:
        raise ValueError(f"partition {partition!r} not in {config.PARTITIONS}")
    fetch = fetch or http_fetch
    folder = Path(staging_dir if staging_dir is not None else config.STAGING_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / config.DOWNLOAD_FILE_PATTERN.format(partition=partition)

    if final.exists() and not force:
        logger.info("skip existing %s", final)
        return final

    tmp = folder / f".{final.name}.{os.getpid()}.part"
    try:
        fetch(partition_url(partition), tmp)
        pq.read_metadata(tmp)  # raises if the file is not readable parquet
        tmp.replace(final)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("downloaded %s", final)
    return final


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download one source partition.")
    parser.add_argument("partition")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(download_partition(args.partition, force=args.force))


if __name__ == "__main__":
    main()
