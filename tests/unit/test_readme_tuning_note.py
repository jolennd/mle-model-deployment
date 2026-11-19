"""README note on restarting tuning (review 4, item 1).

Required: one paragraph or list item (a block between blank lines) of README.md that
contains all of: "trials.json", "delete" (any case), "restart" (any case) and the
exact words "earlier MLflow runs remain". Everything else is the author's wording.
"""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_explains_how_to_restart_tuning():
    blocks = re.split(r"\n\s*\n", README.read_text())
    found = [
        b
        for b in blocks
        if "trials.json" in b
        and "earlier MLflow runs remain" in b
        and "delete" in b.lower()
        and "restart" in b.lower()
    ]
    assert found, (
        "README.md needs a paragraph that mentions trials.json, deleting it to "
        "restart, and that earlier MLflow runs remain"
    )
