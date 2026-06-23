---
name: brisbane-crash-build
description: >-
  Executes the Brisbane Crash ML Lab autonomous one-shot build. Use when the user
  pastes AUTONOMOUS_AGENT_PROMPT.md, asks to finish/build brisbane-crash-ml-lab,
  or requests the full crash ML pipeline, release, or v0.1.0 delivery.
---

# Brisbane Crash ML Lab — autonomous build skill

## Binding inputs

1. Read [`AGENT_SETUP.md`](../../../AGENT_SETUP.md) first — environment is **pre-provisioned**.
2. Read [`PROJECT_OVERVIEW.md`](../../../PROJECT_OVERVIEW.md) completely — binding spec.
3. Follow [`AUTONOMOUS_AGENT_PROMPT.md`](../../../AUTONOMOUS_AGENT_PROMPT.md) — execution contract.
4. Follow [developmental-commit-timeline](../developmental-commit-timeline/SKILL.md) for all commits — window **2026-06-23 → 2026-07-12 (+10:00)**.

## Orchestrator vs workers

- **Parent chat model:** Grok 4.5 (orchestrator only — plan, delegate, verify).
- **All code/docs/tests/git execution:** Composer 2.5 via `.cursor/agents/` (`implementer`, `tester`, `docs-writer`, `explorer`, `git-historian`).
- Task tool calls must pass `model: "composer-2.5"` when not using a named agent.

## Do not re-do setup

Reuse unless broken:

| Asset | Location / note |
|---|---|
| Git remote | `origin` → `jt-09/brisbane-crash-ml-lab` (GitHub; `gh` authenticated as `jt-09`) |
| Python / uv | Python 3.12 via `uv`; `.venv` already created; `uv.lock` present |
| Pre-seeded raw data | `data/raw/brisbane_crashes_2015_2023.csv` (gitignored) |
| Preseed manifest | `artifacts/manifests/preseed_acquisition.json` |
| Fixture | `data/samples/fixture.csv` |
| Configs | `configs/{base,smoke,standard,extended}.yaml` |
| Package stub | `src/crashlab` CLI stub only — **replace with full implementation** |

Verify raw file before re-downloading:

```powershell
Get-FileHash data/raw/brisbane_crashes_2015_2023.csv -Algorithm SHA256
# expect CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22
```

If hash matches and size ≤ 50 MiB, **do not re-download**. Wire acquisition to accept this file (idempotent) and still implement full acquire/validate logic + mocked HTTP tests.

## Hard constraints (never violate)

- CPU only; no CUDA/GPU frameworks as defaults
- Raw download ≤ 50 MiB; never silent full ~200 MiB QLD CSV
- `ALLOW_LARGE_DOWNLOAD=1` must stay off during the build
- Smoke: fixture only, no network, < 5 min
- Standard: ≤ 25 min end-to-end target
- No leakage fields in severity predictors; isolate `leakage_demo`
- Whole-year temporal splits; fit encoders on train only
- Never commit raw data, secrets, model binaries, or caches
- Association/prediction only — no causal road-safety claims

## Execution order

Use planned branches and merge via GitHub PRs when `gh` works (preferred). Date public history with developmental-commit-timeline inside **2026-06-23 → 2026-07-12**:

1. `chore/bootstrap` (~23–25 Jun)
2. `feat/data-acquisition` (~25–28 Jun)
3. `feat/data-validation` (~28–30 Jun)
4. `feat/eda-features` (~30 Jun–2 Jul)
5. `feat/classification` (~2–5 Jul)
6. `feat/anomaly-spatial` (~5–7 Jul)
7. `feat/count-models` (~7–9 Jul)
8. `feat/reporting-release` (~9–12 Jul; tag `v0.1.0` by 12 Jul)

For each: `implementer` → `tester` → conventional backdated commits via `git-historian` → `docs/prs/NN-*.md` via `docs-writer` → `gh pr create` → merge after checks → continue. Rebuild scaffolding Jul-14 commits into the Jun–Jul window per the autonomous prompt.

Keep `artifacts/manifests/build_log.md` updated throughout.

## OpenDataSoft field quirks (already observed)

- Dataset id: `road-crash-locations-queensland`
- Remote fields are mostly snake_case already
- Casualty fields on mirror: `count_casualty_medicallytreated`, `count_casualty_minorinjury` → map to canonical `count_casualty_medically_treated`, `count_casualty_minor_injury`
- Year filter ODSQL example that works: `crash_year >= date'2015' AND crash_year <= date'2023'`
- Brisbane 2015–2023 count observed at setup: **31046** rows, **~9.45 MiB**
- Mirror metadata: most recent 12 months may be preliminary; exclude 2024+ from principal benchmark by default

## Definition of done

Satisfy PROJECT_OVERVIEW §17 and AUTONOMOUS_AGENT_PROMPT final response requirements. Tag `v0.1.0` only after verified checks. Do not ask the user questions — decide and document deviations.

## More detail

See [reference.md](reference.md) for branch/commit checklist and preseed verification commands.
