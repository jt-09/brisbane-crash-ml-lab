# Brisbane Crash build — reference

## Preseed verification

```powershell
$env:Path = "C:\Users\jayth\.local\bin;$env:Path"
uv --version
uv run python --version   # expect 3.12.x
Test-Path .venv
Test-Path data/raw/brisbane_crashes_2015_2023.csv
(Get-Item data/raw/brisbane_crashes_2015_2023.csv).Length  # 9907354
(Get-FileHash data/raw/brisbane_crashes_2015_2023.csv -Algorithm SHA256).Hash
# CE2A0435366E0870F238295CB9A2A700C12F6FFB25815B8D51ACA2A506CEFF22
gh auth status
git remote -v
```

## Developmental commit window

- Skill: `.cursor/skills/developmental-commit-timeline/SKILL.md`
- Start: `2026-06-23` / End: `2026-07-12` / TZ: `+10:00`
- Subagent: `git-historian` (Composer 2.5)
- Uneven milestone gaps; set `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`

## Preferred PR titles

1. `chore: bootstrap cpu-first python project`
2. `feat(data): add filtered crash data acquisition`
3. `feat(data): add schema validation and cleaning pipeline`
4. `feat(features): add temporal splits and leakage-safe features`
5. `feat(models): add classification experiments`
6. `feat(models): add anomaly detection and hotspot clustering`
7. `feat(models): add suburb-month count models`
8. `feat(report): add evaluation report and streamlit explorer` (+ release docs/tag)

## Merge policy

- Squash-merge hosted PRs when checks pass
- Delete remote branches after merge
- If `gh`/`origin` fails: local `git merge --no-ff` + keep `docs/prs/` summaries
- Never invent a remote URL

## Runtime budget reminder (standard)

| Stage | Budget |
|---|---:|
| Acquisition/verify | 2 min (skip re-download if preseed valid) |
| Clean/validate | 2 min |
| EDA | 2 min |
| Features | 2 min |
| Binary | 7 min |
| Multiclass/ordinal | 4 min |
| Anomaly/spatial | 3 min |
| Counts | 2 min |
| Report | 1 min |
| **Total** | **~20–25 min** |

## CLI surface to implement

```text
crashlab acquire|validate|prepare|train-binary|train-multiclass|train-ordinal|
         detect-anomalies|cluster-hotspots|train-counts|report|all
```

Profiles: `--profile smoke|standard|extended`
