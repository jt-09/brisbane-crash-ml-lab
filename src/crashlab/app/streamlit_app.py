"""Streamlit explorer — reads precomputed artifacts; does not retrain on load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]
import streamlit as st

from crashlab.config import VALID_PROFILES, find_repo_root, load_config
from crashlab.data.artifacts import processed_path
from crashlab.evaluation.artifact_loader import load_task_metrics
from crashlab.paths import CrashlabPaths, ensure_dirs

PAGES = (
    "Dataset overview",
    "Severity comparison",
    "Calibration & threshold",
    "Anomaly explorer",
    "Hotspot map",
    "Count models",
    "Provenance & limitations",
)


def _default_profile() -> str:
    return "smoke"


@st.cache_data(show_spinner=False)
def _load_metrics(profile: str, repo_root: str) -> dict:
    config = load_config(profile, repo_root=Path(repo_root))
    paths = CrashlabPaths.from_config(config)
    return load_task_metrics(config, paths)


@st.cache_data(show_spinner=False)
def _load_processed(profile: str, repo_root: str) -> pd.DataFrame | None:
    config = load_config(profile, repo_root=Path(repo_root))
    paths = ensure_dirs(config)
    parquet = processed_path(paths)
    if not parquet.is_file():
        return None
    return pd.read_parquet(parquet)


def _empty(msg: str) -> None:
    st.info(msg)


def page_dataset_overview(df: pd.DataFrame | None, metrics: dict) -> None:
    st.header("Dataset overview")
    if df is None:
        _empty("Processed parquet not found. Run `crashlab prepare` first.")
        return
    st.metric("Rows (cleaned)", len(df))
    if "crash_year" in df.columns:
        st.bar_chart(df["crash_year"].value_counts().sort_index())
    quality = metrics.get("data_quality")
    if quality:
        st.subheader("Data quality")
        st.json({k: quality[k] for k in list(quality.keys())[:12]})


def page_severity_comparison(metrics: dict) -> None:
    st.header("Severity model comparison")
    binary = metrics.get("binary") or {}
    if not binary:
        _empty("Binary metrics not found. Run `crashlab train-binary`.")
        return
    rows = []
    for moment, payload in binary.items():
        if not payload:
            continue
        for entry in payload.get("leaderboard", []):
            rows.append(
                {
                    "moment": moment,
                    "model": entry.get("model_name"),
                    "val_pr_auc": entry.get("val_pr_auc"),
                    "champion": entry.get("is_champion"),
                }
            )
    if not rows:
        _empty("No leaderboard rows available.")
        return
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True)
    chart_df = table.dropna(subset=["val_pr_auc"])
    if len(chart_df):
        fig = px.bar(chart_df, x="model", y="val_pr_auc", color="moment", barmode="group")
        st.plotly_chart(fig, use_container_width=True)


def page_calibration(metrics: dict) -> None:
    st.header("Calibration and threshold explorer")
    explanation = metrics.get("explanation")
    if not explanation:
        _empty("Explanation artifacts not found. Run `crashlab report`.")
        return
    cal = explanation.get("calibration_comparison", {})
    methods = cal.get("methods", [])
    if methods:
        rows = [{"method": m.get("method"), "brier": m.get("brier")} for m in methods]
        st.dataframe(pd.DataFrame(rows))
    threshold = (explanation.get("champion") or {}).get("threshold", 0.5)
    st.slider(
        "Decision threshold (display only — uses champion threshold)",
        0.05,
        0.95,
        float(threshold),
        disabled=True,
    )
    boot = explanation.get("bootstrap_cis", {})
    if boot and not boot.get("skipped"):
        st.write("Bootstrap CIs (test split):", boot)


def page_anomalies(metrics: dict, paths: CrashlabPaths, profile: str) -> None:
    st.header("Anomaly explorer")
    payload = metrics.get("anomalies")
    if not payload:
        _empty("Anomaly metrics not found. Run `crashlab detect-anomalies`.")
        return
    st.json({k: payload[k] for k in ("methods", "n_rows", "seed") if k in payload})
    review = paths.tables_dir / f"anomaly_review_{profile}.csv"
    if review.is_file():
        st.dataframe(pd.read_csv(review).head(200), use_container_width=True)
    else:
        _empty("Anomaly review table not found.")


def page_hotspot_map(df: pd.DataFrame | None, metrics: dict) -> None:
    st.header("Hotspot map")
    if df is None:
        _empty("Processed data required for map.")
        return
    if "loc_latitude" not in df.columns or "loc_longitude" not in df.columns:
        _empty("Coordinates missing from processed data.")
        return
    subset = df.dropna(subset=["loc_latitude", "loc_longitude"]).head(5000)
    if subset.empty:
        _empty("No coordinate-valid rows to plot.")
        return
    fig = px.scatter_mapbox(
        subset,
        lat="loc_latitude",
        lon="loc_longitude",
        color="crash_severity" if "crash_severity" in subset.columns else None,
        zoom=9,
        height=500,
        opacity=0.5,
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    hotspots = metrics.get("hotspots")
    if hotspots:
        st.caption(f"Hotspot methods: {hotspots.get('methods', [])}")


def page_counts(metrics: dict) -> None:
    st.header("Suburb-month count models")
    payload = metrics.get("counts")
    if not payload:
        _empty("Count model metrics not found. Run `crashlab train-counts`.")
        return
    st.write(payload.get("exposure_note", ""))
    candidates = payload.get("candidates", [])
    if candidates:
        rows = [
            {
                "model": c.get("model_name"),
                "val_mae": (c.get("val_metrics") or {}).get("mae"),
                "test_mae": (c.get("test_metrics") or {}).get("mae"),
            }
            for c in candidates
            if c.get("valid", True)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.json(payload.get("mean_variance", {}))


def page_provenance(config_profile: str, metrics: dict) -> None:
    st.header("Provenance and limitations")
    st.markdown(
        """
        - **Source:** Queensland Government road crash locations (Brisbane City subset).
        - **License:** CC BY 4.0 — attribution required.
        - **Interpretation:** Predictive associations only; not causal or
          operational road-safety guidance.
        - **Limits:** No exposure adjustment; preliminary recent years; reporting bias.
        """
    )
    st.subheader("Run manifest")
    run_all = metrics.get("run_all")
    if run_all:
        st.json(run_all.get("timings", {}))
    else:
        _empty("Pipeline manifest not found.")
    st.caption(f"Active profile: {config_profile}")


def main() -> None:
    st.set_page_config(page_title="Brisbane Crash ML Lab", layout="wide")
    st.sidebar.title("Crashlab")
    profile = st.sidebar.selectbox("Profile", sorted(VALID_PROFILES), index=0)
    page = st.sidebar.radio("Page", PAGES)
    repo_root = str(find_repo_root())

    try:
        metrics = _load_metrics(profile, repo_root)
        config = load_config(profile, repo_root=Path(repo_root))
        paths = ensure_dirs(config)
        df = _load_processed(profile, repo_root)
    except Exception as exc:  # noqa: BLE001 — show graceful empty state
        st.error(f"Failed to load artifacts: {exc}")
        return

    if page == PAGES[0]:
        page_dataset_overview(df, metrics)
    elif page == PAGES[1]:
        page_severity_comparison(metrics)
    elif page == PAGES[2]:
        page_calibration(metrics)
    elif page == PAGES[3]:
        page_anomalies(metrics, paths, profile)
    elif page == PAGES[4]:
        page_hotspot_map(df, metrics)
    elif page == PAGES[5]:
        page_counts(metrics)
    else:
        page_provenance(profile, metrics)


if __name__ == "__main__":
    main()
