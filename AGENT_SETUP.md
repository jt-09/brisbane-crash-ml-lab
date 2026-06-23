# Agent setup handoff

This file records what was **pre-provisioned** so the autonomous one-shot agent can start implementation immediately without repeating environment work.

**Do not treat this repository as complete.** Only bootstrap assets exist. Full pipelines, experiments, report, app, and `v0.1.0` are still to be built per `AUTONOMOUS_AGENT_PROMPT.md` + `PROJECT_OVERVIEW.md`.

---

## Machine snapshot (setup time)

| Item | Value |
|---|---|
| OS | Windows 10/11 (`win32`) |
| CPU logical processors | 12 |
| RAM | ~30.7 GB |
| Python for project | **3.12.10** via `uv` (3.13 also installed system-wide; do not use as project default) |
| `uv` | 0.11.28 at `%USERPROFILE%\.local\bin` (add to PATH in new shells) |
| Git | 2.50.1 |
| GitHub CLI | authenticated as **`jt-09`** (`repo`, `workflow` scopes) |

---

## Already done

### Repository / GitHub

- Local git initialised on `main` in this working directory (`road-crash-dataset-mll` folder name is fine; package/project name is `brisbane-crash-ml-lab` / `crashlab`).
- Remote: **`https://github.com/jt-09/brisbane-crash-ml-lab`** (default branch `main`).
- PR template, issue templates, and offline CI workflow scaffold present under `.github/`.
- Setup landed on `main` as commit `chore: provision environment…`; PR flow verified via [#1](https://github.com/jt-09/brisbane-crash-ml-lab/pull/1) (`START_HERE.md`).
- Summary: `docs/prs/00-environment-setup.md`.

### Python environment

```text
uv sync --python 3.12
.venv/          # local, gitignored
uv.lock         # committed
pyproject.toml  # deps pinned for the planned stack
```

Smoke check:

```powershell
$env:Path = "C:\Users\jayth\.local\bin;$env:Path"
uv run crashlab version
```

### Data (pre-seeded; gitignored)

| Field | Value |
|---|---|
| Path | `data/raw/brisbane_crashes_2015_2023.csv` |
| LGA | Brisbane City |
| Years | 2015–2023 inclusive |
| Rows | 31046 |
| Bytes | 9907354 (~9.45 MiB) |
| SHA-256 | `CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22` |
| Source | OpenDataSoft export API (filtered columns) |
| Manifest | `artifacts/manifests/preseed_acquisition.json` |
| Remote metadata copy | `data/external/opendatasoft_dataset_metadata.json` |

**Acquisition policy:** If the file exists and the SHA-256 matches, reuse it. Still implement full `acquire` with schema introspection, adaptive years, 50 MiB cap, mocked HTTP tests, and provenance rewriting into the project's canonical manifest format.

**Field mapping note:** Mirror uses `count_casualty_medicallytreated` and `count_casualty_minorinjury` (map to canonical underscored names in cleaning).

**Year policy:** Mirror description still flags the most recent 12 months as preliminary. Principal benchmark uses 2015–2023; do not add 2024+ to the main leaderboard unless policy explicitly allows.

### Fixture / configs / stubs

- Synthetic fixture: `data/samples/fixture.csv` (committed)
- Profiles: `configs/{base,smoke,standard,extended}.yaml`
- Package stub only: `src/crashlab/{__init__,cli}.py` — replace during Phase A+
- Directory tree for data/docs/notebooks/reports/artifacts/tests present
- Cursor skill: `.cursor/skills/brisbane-crash-build/`
- Cursor rule: `.cursor/rules/brisbane-crash-ml-lab.mdc`

---

## Explicitly NOT done (agent must implement)

- Full CLI (`acquire`, `validate`, `prepare`, train_*, anomalies, spatial, counts, `report`, `all`)
- Real acquisition module (beyond consuming preseed)
- Validation/cleaning/Parquet
- EDA generators + notebooks
- Features, leakage tests, temporal splits
- All model families and experiment registry
- HTML report + Streamlit app
- Complete docs (architecture, dictionary, provenance, model card, release checklist)
- Phase branches / PR series through release
- Tag `v0.1.0`

---

## How to start the autonomous build

1. Open this repo in Cursor.
2. Paste the body of `AUTONOMOUS_AGENT_PROMPT.md` (or say: follow that prompt and the `brisbane-crash-build` skill).
3. The agent should read this file first, then execute all phases without asking questions.

### Suggested first agent commands

```powershell
$env:Path = "C:\Users\jayth\.local\bin;$env:Path"
cd <repo>
git status
gh auth status
uv sync
Get-FileHash data/raw/brisbane_crashes_2015_2023.csv -Algorithm SHA256
```

Then create `chore/bootstrap` from `main` and begin Phase A implementation (expanding beyond this setup scaffold).
