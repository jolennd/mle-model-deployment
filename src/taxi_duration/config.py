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


# Training settings (ADR 8). Values come from the brief, docs/tuning-plan.md and ADR 7.
LABEL_HORIZON_MINUTES = 60
EMBARGO_MINUTES = 60

# Scope rules (ADR 7). Bounds are inclusive.
MIN_DURATION_MINUTES = 1
MAX_DURATION_MINUTES = 60
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 100
MAX_SPEED_MPH = 80
EXCLUDED_ZONE_IDS = (264,)

# Split: train Jan 1 to 24, test Jan 25 to 31 (brief). Test pickups start at TEST_START.
TRAIN_START = "2025-01-01"
TEST_START = "2025-01-25"
TEST_END = "2025-02-01"

SAMPLE_SIZE = 300_000
SEED = 42

# CV folds (docs/tuning-plan.md): time-ordered expanding windows by pickup day.
# Each entry: (last train pickup day, first validation day, last validation day), inclusive.
CV_FOLDS = (
    ("2025-01-12", "2025-01-13", "2025-01-16"),
    ("2025-01-16", "2025-01-17", "2025-01-20"),
    ("2025-01-20", "2025-01-21", "2025-01-24"),
)

RESULTS_DIR = "reports/results"
CV_FOLDS_PATH = f"{RESULTS_DIR}/cv_folds.json"
LOCK_PATH = f"{RESULTS_DIR}/baseline_lock.json"

# Columns never used as features (ADR 2). Compared case-insensitively.
POST_TRIP_COLUMNS = (
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "payment_type",
    "RatecodeID",
    "extra",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
)
FEATURE_COLUMNS = (
    "trip_distance",
    "pickup_hour",
    "pickup_dow",
    "pickup_zone_id",
    "dropoff_zone_id",
)
CATEGORICAL_FEATURES = ("pickup_zone_id", "dropoff_zone_id")

MODEL_NAME = "taxi-duration"
BASELINE_ALIAS = "baseline"
