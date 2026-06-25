"""OpenDataSoft acquisition with preseed short-circuit and adaptive year selection."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from crashlab.config import CrashlabConfig
from crashlab.data.manifest import build_acquisition_manifest, write_acquisition_manifest
from crashlab.data.schema import (
    CANONICAL_FIELDS,
    PRESEED_SHA256,
    REQUIRED_FIELDS,
    normalize_column_name,
)
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("data.acquire")

MIN_ACCEPT_BYTES = 100 * 1024  # 100 KiB sanity floor per PROJECT_OVERVIEW
USER_AGENT = "crashlab/0.1.0 (Brisbane Crash ML Lab; educational research)"
PRESEED_BASENAME = "brisbane_crashes_2015_2023.csv"
FIXTURE_RAW_BASENAME = "fixture_smoke.csv"


class AcquisitionError(RuntimeError):
    """Raised when raw data cannot be acquired or verified."""


def network_allowed() -> bool:
    """Return False when CRASHLAB_ALLOW_NETWORK=0 (offline / CI)."""
    value = os.environ.get("CRASHLAB_ALLOW_NETWORK", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def default_raw_path(paths: CrashlabPaths, config: CrashlabConfig) -> Path:
    """Canonical preseed / standard acquisition destination."""
    preseed = config.raw.get("preseed_raw")
    if isinstance(preseed, dict) and isinstance(preseed.get("path"), str):
        return Path(preseed["path"])
    return paths.raw_dir / PRESEED_BASENAME


def fixture_acquire_path(paths: CrashlabPaths, config: CrashlabConfig) -> Path:
    """Dedicated destination for smoke/fixture acquisition (never the preseed file)."""
    configured = config.fixture_raw_path
    if isinstance(configured, str):
        path = Path(configured)
        if not path.is_absolute():
            path = config.repo_root / configured
        return path.resolve()
    return (paths.raw_dir / FIXTURE_RAW_BASENAME).resolve()


def acquisition_target_path(paths: CrashlabPaths, config: CrashlabConfig) -> Path:
    """Path written by ``run_acquire`` for the active profile."""
    if config.use_fixture:
        return fixture_acquire_path(paths, config)
    return default_raw_path(paths, config)


def _expected_preseed_sha(config: CrashlabConfig) -> str:
    preseed = config.raw.get("preseed_raw")
    if isinstance(preseed, dict) and isinstance(preseed.get("sha256"), str):
        return str(preseed["sha256"]).upper()
    return PRESEED_SHA256


def _guard_fixture_destination(config: CrashlabConfig, destination: Path) -> None:
    """Refuse to clobber the preseed CSV with fixture content."""
    if destination.name == PRESEED_BASENAME:
        msg = (
            f"Fixture acquisition must not write to preseed path {destination}; "
            f"use fixture_raw_path (e.g. {FIXTURE_RAW_BASENAME})"
        )
        raise AcquisitionError(msg)
    if destination.is_file():
        digest = sha256_file(destination)
        if digest == _expected_preseed_sha(config):
            msg = (
                f"Refusing to overwrite preseed file at {destination} "
                f"(SHA256 matches configured preseed)"
            )
            raise AcquisitionError(msg)


def acquisition_manifest_path(paths: CrashlabPaths, profile: str) -> Path:
    return paths.manifests_dir / f"acquisition_{profile}.json"


def resolve_raw_input_path(config: CrashlabConfig, paths: CrashlabPaths) -> Path:
    """Path used by validate/prepare stages after acquisition."""
    if config.use_fixture:
        fixture_dest = fixture_acquire_path(paths, config)
        if fixture_dest.is_file():
            return fixture_dest
        if config.fixture_path:
            fixture = Path(config.fixture_path)
            if fixture.is_file():
                return fixture
        return fixture_dest

    raw = default_raw_path(paths, config)
    if raw.is_file():
        return raw
    return raw


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        total = sum(1 for _ in handle)
    return max(total - 1, 0)


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    reraise=True,
)
def _http_get(url: str, *, stream: bool = False, timeout: int = 120) -> requests.Response:
    response = requests.get(
        url,
        stream=stream,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response


def _ods_base(config: CrashlabConfig) -> str:
    ods = config.data.get("opendatasoft", {})
    if isinstance(ods, dict) and isinstance(ods.get("base_url"), str):
        return str(ods["base_url"]).rstrip("/")
    return "https://queensland.opendatasoft.com/api/explore/v2.1"


def _dataset_id(config: CrashlabConfig) -> str:
    ods = config.data.get("opendatasoft", {})
    if isinstance(ods, dict) and isinstance(ods.get("dataset_id"), str):
        return str(ods["dataset_id"])
    return "road-crash-locations-queensland"


def fetch_dataset_metadata(config: CrashlabConfig) -> dict[str, Any]:
    base = _ods_base(config)
    dataset_id = _dataset_id(config)
    url = f"{base}/catalog/datasets/{dataset_id}"
    response = _http_get(url)
    payload = response.json()
    if not isinstance(payload, dict):
        msg = f"Unexpected metadata payload type from {url}"
        raise AcquisitionError(msg)
    return payload


def discover_remote_fields(metadata: dict[str, Any]) -> list[str]:
    fields_section = metadata.get("fields")
    if isinstance(fields_section, list):
        names: list[str] = []
        for item in fields_section:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        if names:
            return names
    # Fallback: use canonical names when introspection is sparse (tests / mocks).
    return list(CANONICAL_FIELDS)


def build_field_mapping(remote_fields: list[str]) -> dict[str, str]:
    """Map remote field identifiers to canonical project columns."""
    remote_lower = {name.lower(): name for name in remote_fields}
    mapping: dict[str, str] = {}
    for canonical in CANONICAL_FIELDS:
        if canonical in remote_fields:
            mapping[canonical] = canonical
            continue
        alias_hits = [
            remote_lower[key] for key in remote_lower if normalize_column_name(key) == canonical
        ]
        if alias_hits:
            mapping[canonical] = alias_hits[0]
    return mapping


def verify_required_fields_resolvable(field_mapping: dict[str, str]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in field_mapping]
    if missing:
        msg = f"Required fields cannot be resolved from remote schema: {missing}"
        raise AcquisitionError(msg)


def build_where_clause(
    config: CrashlabConfig,
    *,
    year_start: int,
    year_end: int,
) -> str:
    lga = str(config.data.get("lga", "Brisbane City"))
    return (
        f"loc_local_government_area = '{lga}' "
        f"AND crash_year >= date'{year_start}' "
        f"AND crash_year <= date'{year_end}'"
    )


def build_export_url(
    config: CrashlabConfig,
    *,
    year_start: int,
    year_end: int,
    field_mapping: dict[str, str],
) -> str:
    base = _ods_base(config)
    dataset_id = _dataset_id(config)
    where = build_where_clause(config, year_start=year_start, year_end=year_end)
    select_fields = [field_mapping[field] for field in CANONICAL_FIELDS if field in field_mapping]
    params: dict[str, str] = {"where": where, "delimiter": ","}
    if select_fields:
        params["select"] = ",".join(select_fields)
    return f"{base}/catalog/datasets/{dataset_id}/exports/csv?{urlencode(params)}"


def probe_record_count(config: CrashlabConfig, *, year_start: int, year_end: int) -> int:
    base = _ods_base(config)
    dataset_id = _dataset_id(config)
    where = build_where_clause(config, year_start=year_start, year_end=year_end)
    url = f"{base}/catalog/datasets/{dataset_id}/records?{urlencode({'where': where, 'limit': 0})}"
    response = _http_get(url)
    payload = response.json()
    total = payload.get("total_count")
    if isinstance(total, int):
        return total
    return 0


def _atomic_write_bytes(destination: Path, write_fn) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        write_fn(temp_path)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def stream_export_to_file(
    export_url: str,
    destination: Path,
    *,
    max_bytes: int,
) -> tuple[int, dict[str, str]]:
    """Stream CSV export with byte ceiling; return bytes written and response headers."""
    response = _http_get(export_url, stream=True)
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        msg = "Export response looks like HTML, not CSV"
        raise AcquisitionError(msg)

    headers = {key: value for key, value in response.headers.items()}
    written = 0

    def _write(temp_path: Path) -> None:
        nonlocal written
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    msg = f"Export exceeded byte ceiling ({max_bytes} bytes)"
                    raise AcquisitionError(msg)
                handle.write(chunk)

    _atomic_write_bytes(destination, _write)
    return written, headers


def _copy_fixture(config: CrashlabConfig, destination: Path) -> dict[str, Any]:
    _guard_fixture_destination(config, destination)
    fixture = Path(config.fixture_path) if config.fixture_path else None
    if fixture is None or not fixture.is_file():
        msg = f"Fixture path missing or not a file: {fixture}"
        raise AcquisitionError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture, destination)
    digest = sha256_file(destination)
    rows = _count_csv_rows(destination)
    return {
        "source": "fixture",
        "fixture_path": str(fixture),
        "sha256": digest,
        "byte_size": destination.stat().st_size,
        "row_count": rows,
    }


def _try_preseed_reuse(config: CrashlabConfig, raw_path: Path) -> dict[str, Any] | None:
    preseed = config.raw.get("preseed_raw")
    if not isinstance(preseed, dict):
        return None
    expected_hash = str(preseed.get("sha256", PRESEED_SHA256)).upper()
    candidate = raw_path if raw_path.is_file() else Path(str(preseed.get("path", raw_path)))
    if not candidate.is_file():
        return None
    digest = sha256_file(candidate)
    if digest != expected_hash:
        return None
    if candidate.resolve() != raw_path.resolve():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, raw_path)
    size = raw_path.stat().st_size
    max_bytes = int(config.data.get("max_raw_bytes", 52_428_800))
    if size > max_bytes:
        return None
    rows = _count_csv_rows(raw_path)
    logger.info("Reusing preseeded raw file at %s (%d rows, %d bytes)", raw_path, rows, size)
    return {
        "source": "preseed",
        "sha256": digest,
        "byte_size": size,
        "row_count": rows,
        "preseed_path": str(candidate),
    }


def _download_with_adaptive_years(config: CrashlabConfig, paths: CrashlabPaths) -> dict[str, Any]:
    year_end = int(config.data.get("year_end", 2023))
    year_start = int(config.data.get("year_start", 2015))
    year_min = int(config.data.get("year_min_expand", 2011))
    min_bytes = int(config.data.get("min_raw_bytes", 8_388_608))
    min_rows = int(config.data.get("min_rows", 20_000))
    max_bytes = int(config.data.get("max_raw_bytes", 52_428_800))

    metadata = fetch_dataset_metadata(config)
    remote_fields = discover_remote_fields(metadata)
    field_mapping = build_field_mapping(remote_fields)
    verify_required_fields_resolvable(field_mapping)

    temp_path = paths.raw_dir / ".acquire_probe.csv"
    accepted_start = year_start
    accepted_end = year_end
    export_url = ""
    response_headers: dict[str, str] = {}
    byte_size = 0
    row_count = 0

    for _attempt in range(32):
        export_url = build_export_url(
            config,
            year_start=accepted_start,
            year_end=accepted_end,
            field_mapping=field_mapping,
        )
        logger.info(
            "Probing export for years %d–%d: %s",
            accepted_start,
            accepted_end,
            export_url[:120],
        )
        temp_path.unlink(missing_ok=True)
        byte_size, response_headers = stream_export_to_file(
            export_url,
            temp_path,
            max_bytes=max_bytes + 1,
        )
        row_count = _count_csv_rows(temp_path)

        too_large = byte_size > max_bytes
        too_small = byte_size < min_bytes or row_count < min_rows
        if too_large and accepted_start < accepted_end:
            accepted_start += 1
            continue
        if too_small and accepted_start > year_min:
            accepted_start -= 1
            continue
        if too_large or too_small:
            msg = (
                f"Could not satisfy size/row constraints: {byte_size} bytes, "
                f"{row_count} rows for years {accepted_start}–{accepted_end}"
            )
            raise AcquisitionError(msg)
        break
    else:
        msg = "Adaptive year selection exceeded retry budget"
        raise AcquisitionError(msg)

    raw_path = default_raw_path(paths, config)
    shutil.move(str(temp_path), str(raw_path))
    digest = sha256_file(raw_path)

    return {
        "metadata": metadata,
        "remote_fields": remote_fields,
        "field_mapping": field_mapping,
        "export_url": export_url,
        "response_headers": response_headers,
        "sha256": digest,
        "byte_size": byte_size,
        "row_count": row_count,
        "year_start": accepted_start,
        "year_end": accepted_end,
    }


def run_acquire(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Acquire raw Brisbane crash CSV (network, preseed, or fixture)."""
    started = time.perf_counter()
    raw_path = acquisition_target_path(paths, config)
    manifest_path = acquisition_manifest_path(paths, config.profile)

    if not force and manifest_path.is_file() and raw_path.is_file():
        logger.info("Acquisition manifest exists; skipping (use --force to re-run)")
        return {"status": "skipped", "raw_path": str(raw_path), "manifest_path": str(manifest_path)}

    extra: dict[str, Any] = {}
    source_url = "local"

    if config.use_fixture:
        logger.info("Fixture mode: copying %s -> %s", config.fixture_path, raw_path)
        fixture_info = _copy_fixture(config, raw_path)
        extra.update(fixture_info)
        source_url = str(fixture_info.get("fixture_path", "fixture"))
        year_start = int(config.data.get("year_start", 2015))
        year_end = int(config.data.get("year_end", 2023))
        filters = {"lga": config.data.get("lga"), "year_start": year_start, "year_end": year_end}
        digest = str(fixture_info["sha256"])
        byte_size = int(fixture_info["byte_size"])
        row_count = int(fixture_info["row_count"])
    else:
        preseed_info = _try_preseed_reuse(config, raw_path)
        if preseed_info is not None:
            extra.update(preseed_info)
            source_url = "preseed"
            year_start = int(config.data.get("year_start", 2015))
            year_end = int(config.data.get("year_end", 2023))
            digest = str(preseed_info["sha256"])
            byte_size = int(preseed_info["byte_size"])
            row_count = int(preseed_info["row_count"])
            filters = {
                "lga": config.data.get("lga"),
                "year_start": year_start,
                "year_end": year_end,
                "mode": "preseed_reuse",
            }
        elif not network_allowed():
            msg = (
                "Network acquisition disabled (CRASHLAB_ALLOW_NETWORK=0) and no "
                f"reusable preseed at {raw_path}"
            )
            raise AcquisitionError(msg)
        else:
            if config.data.get("allow_large_download"):
                msg = "ALLOW_LARGE_DOWNLOAD must remain disabled"
                raise AcquisitionError(msg)
            download_info = _download_with_adaptive_years(config, paths)
            year_start = int(download_info["year_start"])
            year_end = int(download_info["year_end"])
            extra.update(download_info)
            source_url = str(download_info.get("export_url", "opendatasoft"))
            digest = str(download_info["sha256"])
            byte_size = int(download_info["byte_size"])
            row_count = int(download_info["row_count"])
            filters = {
                "lga": config.data.get("lga"),
                "year_start": year_start,
                "year_end": year_end,
                "where": build_where_clause(config, year_start=year_start, year_end=year_end),
            }

    if byte_size < MIN_ACCEPT_BYTES and not config.use_fixture and extra.get("source") != "preseed":
        msg = f"Raw file too small ({byte_size} bytes)"
        raise AcquisitionError(msg)

    elapsed = time.perf_counter() - started
    manifest = build_acquisition_manifest(
        config=config,
        source_url=source_url,
        raw_path=raw_path,
        byte_size=byte_size,
        row_count=row_count,
        sha256=digest,
        filters=filters,
        selected_fields=list(CANONICAL_FIELDS),
        timings={"acquire_seconds": elapsed},
        extra=extra,
    )
    write_acquisition_manifest(manifest_path, manifest)
    logger.info(
        "Acquisition complete: %s (%d rows, %d bytes) in %.2fs",
        raw_path,
        row_count,
        byte_size,
        elapsed,
    )
    return {
        "status": "completed",
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "row_count": row_count,
        "byte_size": byte_size,
        "sha256": digest,
        "timings": {"acquire_seconds": elapsed},
    }
