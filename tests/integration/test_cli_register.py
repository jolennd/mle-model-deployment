"""CLI register. The package call is stubbed.

Interface to implement in cli.py (tune imported as a module, so tests can patch):
- register [--format text|json]: calls tune.register_from_results() once with no
  required arguments. It returns {"name", "alias", "version", "sha256"}. Text output
  prints the alias, the version and the SHA-256; json prints the dict.
  ValueError, RuntimeError, FileNotFoundError and FileExistsError give exit 1,
  "Error: <message>" on stderr and no traceback.
"""

import json

import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.integration, pytest.mark.cli]

REGISTERED = {
    "name": "taxi-duration",
    "alias": "champion",
    "version": "7",
    "sha256": "ab12" * 16,
}


@pytest.fixture
def runner():
    return CliRunner()


def stub_register(monkeypatch, outcome):
    from taxi_duration import tune

    calls = []

    def register_from_results(*args, **kwargs):
        calls.append(1)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        tune, "register_from_results", register_from_results, raising=False
    )
    return calls


def cli_group():
    from taxi_duration.cli import cli

    return cli


def test_register_prints_alias_version_and_hash(runner, monkeypatch):
    calls = stub_register(monkeypatch, REGISTERED)

    text = runner.invoke(cli_group(), ["register"])
    assert text.exit_code == 0, text.output
    assert calls == [1]
    assert "champion" in text.stdout
    assert "7" in text.stdout
    assert REGISTERED["sha256"] in text.stdout

    as_json = runner.invoke(cli_group(), ["register", "--format", "json"])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.stdout) == REGISTERED
    assert calls == [1, 1]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("model fails the size or speed limit; not registering"),
        RuntimeError("the fallback stopped for approval"),
        FileNotFoundError("trials.json not found"),
    ],
)
def test_register_errors_exit_one_with_a_clean_message(runner, monkeypatch, error):
    stub_register(monkeypatch, error)

    result = runner.invoke(cli_group(), ["register"])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Error:" in result.stderr
    assert str(error) in result.stderr
    assert "Traceback" not in result.output
