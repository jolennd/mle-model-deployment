.PHONY: download
# Usage: make download PARTITION=2025-01 [FORCE=1]
download:
	@test -n "$(PARTITION)" || (echo "PARTITION is required" && exit 1)
	uv run python -m taxi_duration.ingest.download $(PARTITION) $(if $(FORCE),--force,)

.PHONY: data-load
# Usage: make data-load PARTITION=2025-01
data-load:
	@test -n "$(PARTITION)" || (echo "PARTITION is required" && exit 1)
	uv run python -m taxi_duration.ingest.load $(PARTITION)
