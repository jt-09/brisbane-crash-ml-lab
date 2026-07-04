"""Crashlab Typer CLI — bootstrap foundation with staged pipeline hooks."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from crashlab import __version__
from crashlab.config import VALID_PROFILES, load_config
from crashlab.data.acquire import run_acquire
from crashlab.data.clean import run_prepare
from crashlab.data.validate import run_validate
from crashlab.logging import configure_logging, get_logger
from crashlab.models.binary import run_binary_training
from crashlab.models.multiclass import run_multiclass_training
from crashlab.models.ordinal import run_ordinal_training
from crashlab.paths import ensure_dirs
from crashlab.pipeline import PIPELINE_STAGES, run_all

app = typer.Typer(
    name="crashlab",
    help="Brisbane road-crash ML lab (CPU-first, reproducible).",
    no_args_is_help=True,
)

ProfileOption = Annotated[
    str,
    typer.Option(
        "--profile",
        "-p",
        help="Configuration profile: smoke, standard, or extended.",
    ),
]
ForceOption = Annotated[
    bool,
    typer.Option("--force", help="Force re-run even if outputs already exist."),
]


def _validate_profile(profile: str) -> str:
    if profile not in VALID_PROFILES:
        typer.secho(
            f"Unknown profile {profile!r}; choose from {', '.join(sorted(VALID_PROFILES))}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return profile


def _bootstrap_context(profile: str) -> tuple:
    profile = _validate_profile(profile)
    config = load_config(profile)
    paths = ensure_dirs(config)
    json_log = paths.manifests_dir / f"cli_{profile}.jsonl"
    configure_logging(json_log_path=json_log, force=True)
    return config, paths


def _not_implemented(command: str) -> None:
    msg = (
        f"Command '{command}' is not implemented yet (Phase B+). "
        "Bootstrap wiring is in place — see PROJECT_OVERVIEW.md delivery phases."
    )
    raise NotImplementedError(msg)


@app.callback()
def main() -> None:
    """Crashlab entrypoint."""


@app.command("version")
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command("doctor")
def doctor(
    profile: ProfileOption = "smoke",
) -> None:
    """Load config, ensure directories, and print bootstrap health summary."""
    config, paths = _bootstrap_context(profile)
    logger = get_logger("cli.doctor")
    logger.info("crashlab %s — doctor check for profile %s", __version__, profile)
    summary = {
        "version": __version__,
        "profile": config.profile,
        "repo_root": str(config.repo_root),
        "config_hash": config.digest,
        "use_fixture": config.use_fixture,
        "fixture_path": config.fixture_path,
        "seed": config.seed,
        "paths": {key: str(value) for key, value in paths.__dict__.items()},
    }
    typer.echo(json.dumps(summary, indent=2))


@app.command("acquire")
def acquire(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Download or reuse Brisbane crash raw data."""
    config, paths = _bootstrap_context(profile)
    result = run_acquire(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2))


@app.command("validate")
def validate_cmd(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Validate raw or fixture data against the project contract."""
    config, paths = _bootstrap_context(profile)
    result = run_validate(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2))


@app.command("prepare")
def prepare(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Clean and featurise data for modelling."""
    config, paths = _bootstrap_context(profile)
    result = run_prepare(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2))


@app.command("train-binary")
def train_binary(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Train binary severity classifiers."""
    config, paths = _bootstrap_context(profile)
    result = run_binary_training(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("train-multiclass")
def train_multiclass(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Train multiclass severity models."""
    config, paths = _bootstrap_context(profile)
    result = run_multiclass_training(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("train-ordinal")
def train_ordinal(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Train ordinal severity models."""
    config, paths = _bootstrap_context(profile)
    result = run_ordinal_training(config, paths, force=force)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("detect-anomalies")
def detect_anomalies(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Run anomaly and outlier detection."""
    _bootstrap_context(profile)
    del force
    _not_implemented("detect-anomalies")


@app.command("cluster-hotspots")
def cluster_hotspots(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Cluster spatial crash hotspots."""
    _bootstrap_context(profile)
    del force
    _not_implemented("cluster-hotspots")


@app.command("train-counts")
def train_counts(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Train suburb-month crash count models."""
    _bootstrap_context(profile)
    del force
    _not_implemented("train-counts")


@app.command("report")
def report(
    profile: ProfileOption = "standard",
    force: ForceOption = False,
) -> None:
    """Generate evaluation reports and figures."""
    _bootstrap_context(profile)
    del force
    _not_implemented("report")


@app.command("all")
def all_cmd(
    profile: ProfileOption = "smoke",
    force: ForceOption = False,
) -> None:
    """Run the full pipeline for the selected profile."""
    config, _paths = _bootstrap_context(profile)
    logger = get_logger("cli.all")
    logger.info("Starting staged pipeline (%d stages planned)", len(PIPELINE_STAGES))
    run_all(config, force=force)


if __name__ == "__main__":
    app()
