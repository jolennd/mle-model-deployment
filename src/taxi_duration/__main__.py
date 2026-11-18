"""Lets `python -m taxi_duration` run the same commands as the installed script."""

from taxi_duration.cli import cli

if __name__ == "__main__":
    cli(prog_name="taxi_duration")
