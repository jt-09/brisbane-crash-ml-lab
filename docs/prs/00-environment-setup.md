# PR 00 — Environment bootstrap setup (pre-implementation)

## Summary

Provisions the GitHub repository, locked Python environment, directory scaffold, Cursor skill/rule, synthetic fixture, config stubs, and offline CI skeleton so a later autonomous agent can execute `AUTONOMOUS_AGENT_PROMPT.md` in one shot.

## Scope

- **In:** tooling, docs handoff, fixture, configs, stub package, CI scaffold
- **Out:** full acquisition/clean/model/report implementation (next agent)

## Changes

- `uv` + Python 3.12 lockfile and `.venv` (local)
- Pre-seeded Brisbane 2015–2023 extract locally (gitignored); documented in `AGENT_SETUP.md`
- Cursor skill `brisbane-crash-build` + always-apply rule
- GitHub PR/issue templates and CI workflow
- Minimal `crashlab` CLI stub + version unit test

## Tests

- [x] `uv run ruff check .`
- [x] `uv run pytest` (version stub)
- [x] `uv run mypy src`
- [x] Raw data / manifests / `.venv` gitignored

## Data / provenance

- Local raw file SHA-256 and row counts recorded in `AGENT_SETUP.md`
- Source: Queensland OpenDataSoft mirror; license CC BY 4.0
- Raw CSV **not** committed

## Runtime

- Setup only; no model training

## Risks

- Stub CLI will fail real pipeline commands until Phase A+ lands
- Pre-seeded CSV exists only on the setup machine; clean clones must run `acquire` (or copy the file)

## Checklist

- [x] No secrets or raw data committed
- [x] Handoff docs for autonomous agent
- [x] Remote `origin` configured
