# Start here (autonomous build)

1. Read [AGENT_SETUP.md](AGENT_SETUP.md) — what is already provisioned.
2. Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — binding specification.
3. In a new Cursor agent chat, paste [AUTONOMOUS_AGENT_PROMPT.md](AUTONOMOUS_AGENT_PROMPT.md) (or ask it to follow that prompt + the `brisbane-crash-build` skill).
4. Do **not** re-download data if `data/raw/brisbane_crashes_2015_2023.csv` SHA-256 matches the value in AGENT_SETUP.md.
5. Implement phases A→J with GitHub PRs on the planned branches; tag `v0.1.0` only after verified checks.

Environment setup PR summary: [docs/prs/00-environment-setup.md](docs/prs/00-environment-setup.md).
