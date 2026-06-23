"""Central path helpers derived from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crashlab.config import CrashlabConfig


@dataclass(frozen=True)
class CrashlabPaths:
    """Resolved filesystem locations for pipeline stages."""

    repo_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_dir: Path
    samples_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    manifests_dir: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    metrics_dir: Path

    @classmethod
    def from_config(cls, config: CrashlabConfig) -> CrashlabPaths:
        paths = config.paths
        repo_root = config.repo_root

        def p(key: str, default: str) -> Path:
            value = paths.get(key, default)
            if not isinstance(value, str):
                return (repo_root / default).resolve()
            return Path(value).resolve()

        artifacts = p("artifacts_dir", "artifacts")
        reports = p("reports_dir", "reports")
        return cls(
            repo_root=repo_root,
            raw_dir=p("raw_dir", "data/raw"),
            interim_dir=p("interim_dir", "data/interim"),
            processed_dir=p("processed_dir", "data/processed"),
            external_dir=p("external_dir", "data/external"),
            samples_dir=p("samples_dir", "data/samples"),
            artifacts_dir=artifacts,
            reports_dir=reports,
            manifests_dir=artifacts / "manifests",
            models_dir=artifacts / "models",
            figures_dir=reports / "figures",
            tables_dir=reports / "tables",
            metrics_dir=reports / "metrics",
        )

    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.external_dir,
            self.samples_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.manifests_dir,
            self.models_dir,
            self.figures_dir,
            self.tables_dir,
            self.metrics_dir,
        )

    def ensure_dirs(self) -> None:
        """Create expected data, artifact, and report directories."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)


def ensure_dirs(config: CrashlabConfig) -> CrashlabPaths:
    """Build path helpers and ensure directories exist."""
    paths = CrashlabPaths.from_config(config)
    paths.ensure_dirs()
    return paths
