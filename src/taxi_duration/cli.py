"""Command-line interface. Thin layer: parse options, call the package, print."""

import json
import logging
from importlib import metadata

import click

from taxi_duration import config
from taxi_duration import train as training

log = logging.getLogger(__name__)
EXPECTED = (FileNotFoundError, FileExistsError, ValueError, KeyError)


def _version() -> str:
    try:
        return metadata.version("mle-model-deployment-project")
    except metadata.PackageNotFoundError:
        return "unknown"


def _emit_json(data: dict) -> None:
    click.echo(json.dumps(data, default=str))


def _emit_scores(data: dict) -> None:
    for number, score in enumerate(data["fold_scores"], start=1):
        click.echo(f"fold {number} rmse: {score:.4f}")
    click.echo(f"mean rmse: {data['mean_rmse']:.4f}")
    click.echo(f"mean reference rmse: {data['mean_reference_mean_rmse']:.4f}")


@click.group()
@click.version_option(version=_version(), prog_name="taxi_duration")
@click.option(
    "--verbose", "-v", is_flag=True, help="Show debug logs on standard error."
)
def cli(verbose: bool) -> None:
    """Train the baseline and show its locked score."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


@cli.command()
@click.option(
    "--sample-size",
    type=int,
    default=lambda: config.SAMPLE_SIZE,
    show_default="config.SAMPLE_SIZE",
    help="Rows drawn from the training weeks.",
)
@click.option("--n-estimators", type=int, default=100, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def train(sample_size: int, n_estimators: int, fmt: str) -> None:
    """Cross-validate the baseline and write the lock. Never overwrites a lock."""
    try:
        result = training.run_baseline(sample_size, n_estimators)
    except EXPECTED as exc:
        raise click.ClickException(str(exc)) from exc
    log.info("baseline trained, lock written to %s", result["lock_path"])
    if fmt == "json":
        _emit_json(result)
    else:
        _emit_scores(result)
        click.echo(f"lock: {result['lock_path']}")


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def evaluate(fmt: str) -> None:
    """Print the locked cross-validation scores. Reads the lock file only."""
    try:
        lock = training.read_lock()
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if fmt == "json":
        _emit_json(lock)
    else:
        _emit_scores(lock)
