# Brisbane Crash ML Lab

CPU-first, reproducible ML lab on Brisbane City road-crash records (Queensland open data).

> **Status:** Environment pre-provisioned. Implementation is intentionally incomplete.
> Next step: run the autonomous build using [`AUTONOMOUS_AGENT_PROMPT.md`](AUTONOMOUS_AGENT_PROMPT.md)
> after reading [`AGENT_SETUP.md`](AGENT_SETUP.md).

## Specs

- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — binding project specification
- [`AUTONOMOUS_AGENT_PROMPT.md`](AUTONOMOUS_AGENT_PROMPT.md) — one-shot build prompt
- [`AGENT_SETUP.md`](AGENT_SETUP.md) — what is already provisioned for the build agent

## Quick environment check (pre-build)

```bash
uv sync
uv run crashlab version
```

Do not treat this README as the final portfolio README. The build agent must replace it with full quickstart, outputs, and limitations after Phase J.

## License / attribution

Code: MIT (see `LICENSE`).
Crash data: © State of Queensland / Department of Transport and Main Roads — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
