"""The API binds to the loopback address only, with host and port from the environment."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_make_api_uses_env_host_and_port():
    makefile = (ROOT / "Makefile").read_text()
    recipe = re.search(r"^api:\n((?:\t.*\n)+)", makefile, re.MULTILINE)
    assert recipe, "Makefile has no api target"
    assert "app.main:app" in recipe.group(1)
    assert "--host $(API_HOST)" in recipe.group(1)
    assert "--port $(API_PORT)" in recipe.group(1)


def test_env_example_binds_loopback_port_8000():
    lines = (ROOT / ".env.example").read_text().splitlines()
    assert "API_HOST=127.0.0.1" in lines
    assert "API_PORT=8000" in lines


def test_no_wildcard_bind_in_api_or_make_files():
    files = [ROOT / "Makefile", ROOT / ".env.example", *ROOT.glob("make/*.mk")]
    files += list((ROOT / "src" / "app").rglob("*.py"))
    assert (ROOT / "src" / "app" / "main.py").exists(), "src/app/main.py is missing"
    for path in files:
        assert "0.0.0.0" not in path.read_text(), path
