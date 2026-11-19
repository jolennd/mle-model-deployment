# Design
Components:
- Download: fetches the January 2025 yellow taxi file and records its SHA-256 (docs/02-data-sources.md).
- Raw table: the file loaded as is into the local DuckDB database (DB_CONN).
- dbt staging: renames and types the columns, keeps pickups on or after 2025-01-01 00:00 and before 2025-02-01 00:00, drops trips with dropoff before pickup, and counts the dropped rows. dbt tests check the output.
- dbt mart: one row per trip with a trip key, the pickup and dropoff timestamps and the allowed input columns only. No label column and no post-trip columns (ADR 2).
- Features and label: the label (duration in minutes) and the 1 to 60 minute scope are computed in the Python label code from the timestamps, the same way for training and for the test week. The mart keeps the dropoff timestamp for the label and the embargo. One feature package is used by training and by the API, so both build the same inputs.
- Training: CLI commands train, tune and evaluate. Time split (train Jan 1 to 24, test Jan 25 to 31), baseline lock before the test week is read, Optuna tuning as in docs/tuning-plan.md.
- MLflow server: local, 127.0.0.1:5001, sqlite store, artifacts served through the server (ADR 1). Holds runs and the registered model with an alias.
- API: FastAPI on 127.0.0.1:8000. Loads the model by alias, checks the pinned hash, and reports ready only after the check. Never trains.
Handoffs (who owns which files, and what each step hands to the next):
- Download to raw table: parquet file with a recorded hash.
- Raw table to staging and mart: tables in the DuckDB file. Pipeline owner: transform project and sql. The source has no natural trip key, so the trip key and the handling of exact duplicate rows are decided in the data step, with a uniqueness test. No rows are lost between staging and mart.
- Mart to training: read-only connection through one data door. The test week stays unreadable until the baseline lock exists. ML owner: package code and models.
- Training to MLflow: runs, metrics, and a registered model that carries its feature schema, version and locked metrics.
- MLflow to API: model URI by alias plus a pinned hash in the environment. Service owner: API code.
- Tests: each handoff has a contract test named in docs/01-traceability.md.
- Steps run through make targets in order, with no orchestrator (ADR 3).
