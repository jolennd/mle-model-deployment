"""Save and load the model in the safe format, with an explicit trusted type list."""

from pathlib import Path

from skops.io import dump, load

MODEL_FILE = "model.skops"


def save_model(fitted, folder) -> int:
    """Set the forest to one thread, dump into folder, return total bytes written."""
    fitted.named_steps["model"].set_params(n_jobs=1)
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    dump(fitted, root / MODEL_FILE)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def load_model(folder, trusted_types):
    """Load with the given trusted types only; never the list the file carries."""
    return load(Path(folder) / MODEL_FILE, trusted=list(trusted_types))
