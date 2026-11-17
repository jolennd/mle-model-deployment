.PHONY: transform transform-test transform-debug
# Seeds (lookup CSV files in transform/seeds/) load before the models that join them.
transform:
	uv run python transform/dbt_env.py seed
	uv run python transform/dbt_env.py run
transform-test:
	uv run python transform/dbt_env.py test
transform-debug:
	uv run python transform/dbt_env.py debug
