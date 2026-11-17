import os

# Download settings (ADR 6). Full links are built from these pieces.
DOWNLOAD_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DOWNLOAD_FILE_PATTERN = "yellow_tripdata_{partition}.parquet"
PARTITIONS = ["2025-01"]
STAGING_DIR = "data/raw"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value
