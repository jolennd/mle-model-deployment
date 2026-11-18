"""Save and load the model in the safe format, with an explicit trusted type list."""

import shutil
from pathlib import Path

from skops.io import dump, load

MODEL_FILE = "model.skops"
BYTES_PER_NODE = 100
GB = 1_000_000_000


def estimate_size(fitted) -> int:
    """Rough file size: total tree nodes of the forest times 100 bytes."""
    forest = fitted.named_steps["model"]
    return sum(tree.tree_.node_count for tree in forest.estimators_) * BYTES_PER_NODE


def check_disk(fitted, folder) -> None:
    """Refuse when free space is under twice the estimated file size."""
    parent = Path(folder).resolve().parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    needed = 2 * estimate_size(fitted)
    free = shutil.disk_usage(parent).free
    if free < needed:
        raise RuntimeError(
            f"not enough disk space to save the model: {free / GB:.3f} GB free, "
            f"{needed / GB:.3f} GB needed"
        )


def save_model(fitted, folder) -> int:
    """Set the forest to one thread, dump into folder, return total bytes written."""
    check_disk(fitted, folder)
    fitted.named_steps["model"].set_params(n_jobs=1)
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    dump(fitted, root / MODEL_FILE)
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def load_model(folder, trusted_types):
    """Load with the given trusted types only; never the list the file carries."""
    return load(Path(folder) / MODEL_FILE, trusted=list(trusted_types))
