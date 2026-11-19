# Local settings: .env is git-ignored, copied from .env.example.
-include .env
export
# Every uv run fails when uv.lock is out of date, instead of quietly re-locking.
# Change dependencies on purpose with uv add or uv lock --upgrade-package, then commit uv.lock.
export UV_LOCKED := 1

-include make/db.mk
-include make/transform.mk
-include make/ingest.mk
-include make/notebook.mk
-include make/orchestrate.mk
-include make/assistant.mk
.PHONY: test-fast test test-behavioral lint format validate-gates check-versions reference-record upgrade-check mlflow-server api tune register evaluate

test-fast:
	uv run pytest -m "not dbserver and not docker and not deploy and not model and not data and not transform and not reference and not behavioral"
test:
	uv run pytest -m "not docker and not deploy and not transform and not reference"
# Behavioral cases and prediction sanity checks; needs the saved model.
test-behavioral:
	uv run pytest -m behavioral
lint:
	uv run ruff check --no-cache . && uv run ruff format --check .
format:
	uv run ruff format .
validate-gates: lint test-fast

# Local servers. Host, port and stores come from .env; both bind to the loopback address.
mlflow-server:
	uv run mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) --backend-store-uri $(MLFLOW_BACKEND_STORE_URI) --artifacts-destination $(MLFLOW_ARTIFACTS_DESTINATION) --serve-artifacts
api:
	uv run uvicorn app.main:app --host $(API_HOST) --port $(API_PORT)

# Tuning (docs/tuning-plan.md): runs the pre-registered study through the MLflow server in .env.
tune:
	uv run taxi_duration tune
# Registers the tuned model as taxi-duration@champion and prints the SHA-256 to pin. Run by hand after reading the tune report.
register:
	uv run taxi_duration register
# The one test-week read, after the champion is chosen. Refuses if metrics.json exists.
evaluate:
	uv run taxi_duration evaluate --test-week

# Versions: list direct dependencies with a newer release. Upgrade only through upgrade-check.
check-versions:
	uv tree --outdated --depth 1
# Reference numbers (tests/reference/*.json): fill the values from the current database.
reference-record:
	uv run python tests/integration/test_reference.py --record
# After a dependency upgrade: same tests, rebuilt transform, same reference numbers.
upgrade-check:
	uv sync
	$(MAKE) test-fast
	@if [ -f make/transform.mk ]; then $(MAKE) transform transform-test; fi
	uv run pytest -m reference
