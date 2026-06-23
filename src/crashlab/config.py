"""YAML configuration loading with profile inheritance and path resolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR_NAME = "configs"
VALID_PROFILES = frozenset({"smoke", "standard", "extended"})


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root via pyproject.toml or .git directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return current


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Config file must contain a mapping: {path}"
        raise ValueError(msg)
    return data


def _resolve_inherited_config(config_path: Path, configs_dir: Path) -> dict[str, Any]:
    raw = _load_yaml_file(config_path)
    inherits = raw.get("inherits")
    if inherits is None:
        return raw
    if not isinstance(inherits, str):
        msg = f"'inherits' must be a string in {config_path}"
        raise ValueError(msg)
    parent_path = configs_dir / inherits
    if not parent_path.is_file():
        msg = f"Parent config not found: {parent_path}"
        raise FileNotFoundError(msg)
    parent = _resolve_inherited_config(parent_path, configs_dir)
    return _deep_merge(parent, raw)


def _resolve_path_value(repo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _resolve_paths_in_config(repo_root: Path, config: dict[str, Any]) -> None:
    paths_section = config.get("paths")
    if isinstance(paths_section, dict):
        for key, value in list(paths_section.items()):
            if isinstance(value, str) and key.endswith("_dir"):
                paths_section[key] = _resolve_path_value(repo_root, value)

    preseed = config.get("preseed_raw")
    if isinstance(preseed, dict) and isinstance(preseed.get("path"), str):
        preseed["path"] = _resolve_path_value(repo_root, preseed["path"])

    if isinstance(config.get("fixture_path"), str):
        config["fixture_path"] = _resolve_path_value(repo_root, config["fixture_path"])


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 digest of the resolved configuration mapping."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CrashlabConfig:
    """Resolved runtime configuration for crashlab."""

    profile: str
    repo_root: Path
    config_path: Path
    raw: dict[str, Any] = field(repr=False)
    digest: str = field(repr=False)

    @property
    def project(self) -> dict[str, Any]:
        value = self.raw.get("project")
        return value if isinstance(value, dict) else {}

    @property
    def paths(self) -> dict[str, Any]:
        value = self.raw.get("paths")
        return value if isinstance(value, dict) else {}

    @property
    def data(self) -> dict[str, Any]:
        value = self.raw.get("data")
        return value if isinstance(value, dict) else {}

    @property
    def models(self) -> dict[str, Any]:
        value = self.raw.get("models")
        return value if isinstance(value, dict) else {}

    @property
    def tuning(self) -> dict[str, Any]:
        value = self.raw.get("tuning")
        return value if isinstance(value, dict) else {}

    @property
    def use_fixture(self) -> bool:
        return bool(self.raw.get("use_fixture", False))

    @property
    def fixture_path(self) -> str | None:
        value = self.raw.get("fixture_path")
        return value if isinstance(value, str) else None

    @property
    def seed(self) -> int:
        project = self.project
        seed = project.get("seed", 42)
        return int(seed) if isinstance(seed, int) else 42

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def load_config(
    profile: str = "standard",
    *,
    repo_root: Path | None = None,
    configs_dir: Path | None = None,
) -> CrashlabConfig:
    """Load a profile YAML from ``configs/`` with inheritance and path resolution."""
    if profile not in VALID_PROFILES:
        msg = f"Unknown profile {profile!r}; expected one of {sorted(VALID_PROFILES)}"
        raise ValueError(msg)

    root = (repo_root or find_repo_root()).resolve()
    cfg_dir = (configs_dir or root / CONFIGS_DIR_NAME).resolve()
    config_path = cfg_dir / f"{profile}.yaml"
    if not config_path.is_file():
        msg = f"Profile config not found: {config_path}"
        raise FileNotFoundError(msg)

    merged = _resolve_inherited_config(config_path, cfg_dir)
    _resolve_paths_in_config(root, merged)
    merged["profile"] = profile

    return CrashlabConfig(
        profile=profile,
        repo_root=root,
        config_path=config_path,
        raw=merged,
        digest=config_hash(merged),
    )
