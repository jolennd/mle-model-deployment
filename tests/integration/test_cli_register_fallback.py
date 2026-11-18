"""CLI register --approved-fallback (ADR 8 point 17). The package call is stubbed.

Interface cli.py must provide:
- register [--approved-fallback] [--format text|json]: always calls
  tune.register_from_results(approved_fallback=<bool>) with that one keyword and no
  other argument. Plain register passes False, the flag passes True. Errors from the
  package give exit 1 and "Error: <message>" as before.
- `make -n register` stays the plain command (no flag).
"""

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.integration, pytest.mark.cli]

ROOT = Path(__file__).resolve().parents[2]
REGISTERED = {
    "name": "taxi-duration",
    "alias": "champion",
    "version": "7",
    "sha256": "ab12" * 16,
}


def stub(monkeypatch, outcome):
    from taxi_duration import tune

    calls = []

    def register_from_results(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(tune, "register_from_results", register_from_results)
    return calls


def cli_group():
    from taxi_duration.cli import cli

    return cli


def test_plain_register_passes_false(monkeypatch):
    calls = stub(monkeypatch, REGISTERED)

    result = CliRunner().invoke(cli_group(), ["register"])

    assert result.exit_code == 0, result.output
    assert calls == [((), {"approved_fallback": False})]


def test_flag_passes_true(monkeypatch):
    calls = stub(monkeypatch, REGISTERED)

    text = CliRunner().invoke(cli_group(), ["register", "--approved-fallback"])
    as_json = CliRunner().invoke(
        cli_group(), ["register", "--approved-fallback", "--format", "json"]
    )

    assert text.exit_code == 0, text.output
    assert REGISTERED["sha256"] in text.stdout
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.stdout) == REGISTERED
    assert calls == [((), {"approved_fallback": True})] * 2


def test_flag_errors_exit_one_cleanly(monkeypatch):
    stub(monkeypatch, ValueError("no fallback ran; use plain register"))

    result = CliRunner().invoke(cli_group(), ["register", "--approved-fallback"])

    assert result.exit_code == 1, result.output
    assert "Error: no fallback ran" in result.stderr
    assert "Traceback" not in result.output


def test_make_register_is_unchanged():
    result = subprocess.run(
        ["make", "-n", "register"],
        check=False,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "taxi_duration register" in result.stdout
    assert "approved" not in result.stdout
