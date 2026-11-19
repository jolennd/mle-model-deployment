.PHONY: notebook-run
# Re-runs the exploration notebook in place. Needs the warehouse built and no writer holding it.
notebook-run:
	uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_exploration.ipynb
