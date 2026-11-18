"""Command-line interface. Thin layer: parse options, call the package, print."""

import json
import logging
from importlib import metadata

import click

from taxi_duration import config, tune
from taxi_duration import evaluate as evaluation
from taxi_duration import train as training

log = logging.getLogger(__name__)
EXPECTED = (FileNotFoundError, FileExistsError, ValueError, KeyError)
STEP_ERRORS = (RuntimeError, FileNotFoundError, FileExistsError, ValueError)


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
    """Train the baseline, tune the model and show its scores."""
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


@cli.command("tune")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def tune_command(fmt: str) -> None:
    """Tune on the training weeks and pick the final model."""
    try:
        result = tune.run_tuning()
    except STEP_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc
    if fmt == "json":
        _emit_json(result)
    else:
        click.echo(f"name: {result['name']}")
        click.echo(f"size bytes: {result['size_bytes']}")
        click.echo(f"predict ms: {result['predict_ms']}")


@cli.command()
@click.option(
    "--test-week",
    is_flag=True,
    help="Score the final model and the baseline on the test week (once).",
)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def evaluate(test_week: bool, fmt: str) -> None:
    """Print the locked scores, or with --test-week score the test week."""
    if test_week:
        try:
            result = evaluation.run_evaluation()
        except STEP_ERRORS as exc:
            raise click.ClickException(str(exc)) from exc
        if fmt == "json":
            _emit_json(result)
        else:
            click.echo(f"chosen rmse: {result['chosen']['rmse']:.4f}")
            click.echo(f"baseline rmse: {result['baseline']['rmse']:.4f}")
        return
    try:
        lock = training.read_lock()
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if fmt == "json":
        _emit_json(lock)
    else:
        _emit_scores(lock)
