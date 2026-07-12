# Release checklist (v0.1.0)

## Pre-release

- [x] `uv run pytest` passes offline
- [x] `uv run ruff check src tests` clean
- [ ] `uv run mypy src` acceptable
- [x] `uv run crashlab all --profile smoke --force` completes in &lt;5 min
- [ ] `uv run crashlab all --profile standard` within 25 min budget (local)
- [x] No raw government CSV, secrets, or model binaries committed
- [x] `reports/index.html` generated and opens locally
- [x] Streamlit app loads: `make app`

## Documentation

- [x] README quickstart with exact commands
- [x] `docs/data_provenance.md` attribution present
- [x] `docs/model_card.md` limitations prominent
- [ ] Changelog entry for v0.1.0

## Tag

- [ ] Git tag `v0.1.0` on `feat/reporting-release` merge commit
- [ ] Manifest records config hash, data hash, git SHA

## Known gaps (document if unfixed)

- SHAP optional — not required for v0.1.0
- Extended profile may exceed standard runtime budget
