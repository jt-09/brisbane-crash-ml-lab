"""Static HTML report generation from precomputed pipeline artifacts."""

from __future__ import annotations

import html
import json
import time
from typing import Any

from crashlab.config import CrashlabConfig
from crashlab.data.clean import quality_summary_path
from crashlab.data.manifest import utc_now_iso
from crashlab.evaluation.artifact_loader import (
    list_figure_paths,
    load_task_metrics,
    quality_summary_markdown,
)
from crashlab.evaluation.eda import run_eda
from crashlab.evaluation.explanation import run_explanation
from crashlab.logging import get_logger
from crashlab.paths import CrashlabPaths

logger = get_logger("evaluation.reports")

REPORT_SECTIONS: tuple[str, ...] = (
    "provenance",
    "data_quality",
    "eda",
    "experiments",
    "runtimes",
    "leakage_demo",
    "error_analysis",
    "limitations",
    "model_card",
)

LIMITATIONS = [
    "Models predict associations on reported Brisbane injury crashes; they do not establish causation.",
    "Exposure (vehicle kilometres, population) is not modelled — counts and rates are raw.",
    "Recent-year records may be preliminary per Queensland metadata.",
    "Small subgroups (fatal crashes, rare suburbs) have high metric variance.",
    "Property-damage-only records are excluded from severity modelling.",
]


def _esc(text: object) -> str:
    return html.escape(str(text))


def _table_from_rows(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_esc(c)}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _figure_embed(paths: CrashlabPaths, name: str) -> str:
    fig = paths.figures_dir / name
    if not fig.is_file():
        return f"<p class='muted'>Figure not available: {_esc(name)}</p>"
    rel = f"figures/{name}"
    return (
        f"<figure><img src='{_esc(rel)}' alt='{_esc(name)}' loading='lazy'/>"
        f"<figcaption>{_esc(name)}</figcaption></figure>"
    )


def _leaderboard_html(binary_metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for moment, payload in binary_metrics.items():
        if not payload:
            continue
        leaderboard = payload.get("leaderboard", [])
        if not leaderboard:
            continue
        rows = [
            [
                r.get("model_name"),
                f"{r.get('val_pr_auc'):.4f}" if r.get("val_pr_auc") is not None else "—",
                f"{r.get('val_brier'):.4f}" if r.get("val_brier") is not None else "—",
                "yes" if r.get("is_champion") else "",
                f"{r.get('fit_seconds', 0):.2f}s" if r.get("fit_seconds") else "—",
            ]
            for r in leaderboard
        ]
        parts.append(f"<h3>Binary — {_esc(moment)}</h3>")
        parts.append(
            _table_from_rows(
                ["Model", "Val PR-AUC", "Val Brier", "Champion", "Fit time"],
                rows,
            )
        )
    return "\n".join(parts) if parts else "<p class='muted'>No binary leaderboards found.</p>"


def _other_tasks_html(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for task in ("multiclass", "ordinal"):
        task_data = metrics.get(task) or {}
        for moment, payload in task_data.items():
            if not payload:
                continue
            champion = payload.get("champion") or {}
            parts.append(
                f"<p><strong>{_esc(task)} / {_esc(moment)}</strong>: "
                f"champion={_esc(champion.get('model_name', 'n/a'))}</p>"
            )
    for task in ("anomalies", "hotspots", "counts"):
        payload = metrics.get(task)
        if not payload:
            parts.append(f"<p class='muted'>{_esc(task)}: no metrics artifact.</p>")
            continue
        parts.append(
            f"<p><strong>{_esc(task)}</strong>: "
            f"status=completed, keys={_esc(', '.join(sorted(payload.keys())[:6]))}…</p>"
        )
    return "\n".join(parts)


def _explanation_html(explanation: dict[str, Any] | None) -> str:
    if not explanation:
        return "<p class='muted'>Explanation analysis not run.</p>"
    parts = [
        f"<p class='note'>{_esc(explanation.get('predictive_association_note', ''))}</p>",
    ]
    imp = explanation.get("permutation_importance", {})
    top = imp.get("top_features", [])[:10]
    if top:
        rows = [[f["feature"], f"{f['importance_mean']:.4f}"] for f in top]
        parts.append("<h3>Permutation importance (held-out)</h3>")
        parts.append(_table_from_rows(["Feature", "Mean AP drop"], rows))
    fp_fn = (explanation.get("error_analysis") or {}).get("fp_fn_tables", {})
    parts.append(
        f"<p>False positives: {fp_fn.get('n_false_positives', 'n/a')}; "
        f"false negatives: {fp_fn.get('n_false_negatives', 'n/a')}</p>"
    )
    boot = explanation.get("bootstrap_cis", {})
    if boot and not boot.get("skipped"):
        pr = boot.get("pr_auc", {})
        parts.append(
            f"<p>Test PR-AUC bootstrap CI: "
            f"[{pr.get('low')}, {pr.get('high')}] (median {pr.get('median')})</p>"
        )
    cal = explanation.get("calibration_comparison", {})
    if cal:
        rows = [
            [m.get("method"), f"{m.get('brier'):.4f}" if m.get("brier") is not None else "—"]
            for m in cal.get("methods", [])
        ]
        parts.append("<h3>Calibration comparison</h3>")
        parts.append(_table_from_rows(["Method", "Brier (eval)"], rows))
    return "\n".join(parts)


def _model_card_html(metrics: dict[str, Any], config: CrashlabConfig) -> str:
    binary = metrics.get("binary", {}).get("context", {})
    champion = (binary or {}).get("champion") or {}
    test = champion.get("test_metrics", {})
    return f"""
    <ul>
      <li><strong>Intended use:</strong> exploratory severity-risk scoring on Brisbane injury crashes.</li>
      <li><strong>Out of scope:</strong> operational road-safety decisions, causal inference, deployment without review.</li>
      <li><strong>Champion (context):</strong> {_esc(champion.get("model_name", "n/a"))}</li>
      <li><strong>Test PR-AUC:</strong> {_esc(test.get("pr_auc", "n/a"))}</li>
      <li><strong>Profile:</strong> {_esc(config.profile)} | <strong>Seed:</strong> {_esc(config.seed)}</li>
      <li><strong>Validation:</strong> time-based whole-year holdout (see config splits).</li>
    </ul>
    """


def build_html_report(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    metrics: dict[str, Any],
) -> str:
    """Assemble the static HTML report string."""
    data_cfg = config.data
    official = data_cfg.get("official", {}) if isinstance(data_cfg.get("official"), dict) else {}
    quality_md = quality_summary_markdown(config, paths) or ""
    leakage = metrics.get("binary", {}).get("leakage_demo", {})
    run_all = metrics.get("run_all") or {}
    timings = run_all.get("timings", {})

    eda_figures = "".join(_figure_embed(paths, p.name) for p in list_figure_paths(paths)[:6])

    limitations_list = "".join(f"<li>{_esc(item)}</li>" for item in LIMITATIONS)

    sections = {
        "provenance": f"""
        <p>Data: Queensland road crash locations (Brisbane City subset).</p>
        <ul>
          <li>Official dataset: <a href="{_esc(official.get("dataset_url", ""))}">Queensland Open Data</a></li>
          <li>License: {_esc(official.get("license", "CC BY 4.0"))}</li>
          <li>Config hash: <code>{_esc(config.digest[:16])}…</code></li>
          <li>Generated: {_esc(utc_now_iso())}</li>
        </ul>
        """,
        "data_quality": f"<pre>{_esc(quality_md[:4000])}</pre>"
        if quality_md
        else "<p class='muted'>Quality summary not found — run prepare first.</p>",
        "eda": eda_figures
        or "<p class='muted'>EDA figures not found — report runs EDA when missing.</p>",
        "experiments": _leaderboard_html(metrics.get("binary", {})) + _other_tasks_html(metrics),
        "runtimes": _table_from_rows(
            ["Stage", "Seconds"],
            [[k, f"{v:.2f}"] for k, v in sorted(timings.items())],
        )
        if timings
        else "<p class='muted'>Pipeline timings manifest not found.</p>",
        "leakage_demo": (
            "<p>The <code>leakage_demo</code> moment deliberately includes denylisted casualty fields "
            "to show inflated metrics when post-outcome information leaks in. "
            "It is excluded from leaderboards and must not be used operationally.</p>"
            + (
                f"<p>Leakage demo champion: {_esc((leakage.get('champion') or {}).get('model_name', 'none (excluded)'))}</p>"
                if leakage
                else ""
            )
        ),
        "error_analysis": _explanation_html(metrics.get("explanation")),
        "limitations": f"<ul>{limitations_list}</ul>",
        "model_card": _model_card_html(metrics, config),
    }

    nav = "".join(
        f"<a href='#{sid}'>{_esc(sid.replace('_', ' ').title())}</a>" for sid in REPORT_SECTIONS
    )
    body = "".join(
        f'<section id="{sid}"><h2>{_esc(sid.replace("_", " ").title())}</h2>{sections[sid]}</section>'
        for sid in REPORT_SECTIONS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Brisbane Crash ML Lab — Report ({_esc(config.profile)})</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; line-height: 1.5; color: #1a1a1a; }}
    body {{ max-width: 960px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }}
    nav {{ display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 2rem; padding: .75rem; background: #f4f6f8; border-radius: 6px; }}
    nav a {{ color: #0b57d0; text-decoration: none; }}
    section {{ margin-bottom: 2.5rem; padding-top: .5rem; border-top: 1px solid #e0e0e0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
    th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; }}
    th {{ background: #f0f0f0; }}
    figure img {{ max-width: 100%; height: auto; }}
    .muted {{ color: #666; }}
    .note {{ background: #fff8e1; padding: .75rem; border-radius: 4px; font-size: .9rem; }}
    code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <header>
    <h1>Brisbane Crash ML Lab</h1>
    <p>Static evaluation report — predictive associations only; not causal or operational guidance.</p>
  </header>
  <nav>{nav}</nav>
  <main>{body}</main>
  <footer><p class='muted'>crashlab profile={_esc(config.profile)}</p></footer>
</body>
</html>
"""


def run_report(
    config: CrashlabConfig,
    paths: CrashlabPaths,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Generate HTML report and supporting assets without retraining models."""
    started = time.perf_counter()
    index_path = paths.reports_dir / "index.html"
    manifest_path = paths.manifests_dir / f"report_{config.profile}.json"

    if not force and index_path.is_file() and manifest_path.is_file():
        logger.info("Report exists; skipping (use --force)")
        return {"status": "skipped", "index_html": str(index_path)}

    run_eda(config, paths, force=force)
    run_explanation(config, paths, force=force)

    metrics = load_task_metrics(config, paths)
    html_content = build_html_report(config, paths, metrics)

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(html_content, encoding="utf-8")

    manifest = {
        "schema_version": "1",
        "manifest_type": "report",
        "timestamp_utc": utc_now_iso(),
        "profile": config.profile,
        "index_html": str(index_path),
        "sections": list(REPORT_SECTIONS),
        "quality_summary": str(quality_summary_path(paths, config.profile)),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    elapsed = time.perf_counter() - started
    logger.info("Report written to %s in %.2fs", index_path, elapsed)
    return {
        "status": "completed",
        "index_html": str(index_path),
        "manifest": str(manifest_path),
        "timings": {"report_seconds": elapsed},
    }
