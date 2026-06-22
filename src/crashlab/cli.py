"""Minimal CLI stub. Autonomous agent implements full command surface."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="crashlab",
    help="Brisbane road-crash ML lab (bootstrap stub — implement full CLI during build).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Crashlab entrypoint."""


@app.command("version")
def version() -> None:
    """Print package version."""
    from crashlab import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
