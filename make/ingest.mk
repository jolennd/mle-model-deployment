.PHONY: download
# Usage: make download PARTITION=2025-01 [FORCE=1]
download:
	@test -n "$(PARTITION)" || (echo "PARTITION is required" && exit 1)
	uv run python -m taxi_duration.ingest.download $(PARTITION) $(if $(FORCE),--force,)
