# Start here (autonomous build)

1. Read [AGENT_SETUP.md](AGENT_SETUP.md) — what is already provisioned.
2. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — binding specification (includes Git timeline **2026-06-23 → 2026-07-12**).
3. Open a **new Cursor Agent chat** and select **Grok 4.5** as the parent/orchestrator model.
4. Paste [AUTONOMOUS_AGENT_PROMPT.md](AUTONOMOUS_AGENT_PROMPT.md) (or ask it to follow that prompt + the `brisbane-crash-build` skill).
5. Confirm subagents exist under `.cursor/agents/` (all pinned to Composer 2.5): `implementer`, `tester`, `docs-writer`, `explorer`, `git-historian`.
6. Do **not** re-download data if `data/raw/brisbane_crashes_2015_2023.csv` SHA-256 matches the value in AGENT_SETUP.md.
7. Implement phases A→J with GitHub PRs on the planned branches; rebuild public Git history into the Jun 23–Jul 12 window via the developmental-commit-timeline skill; tag `v0.1.0` only after verified checks (tag date ≤ 2026-07-12).

Environment setup PR summary: [docs/prs/00-environment-setup.md](docs/prs/00-environment-setup.md).
