---
name: developmental-commit-timeline
description: >-
  Plan and execute realistic developmental git commit histories over a
  user-specified date range and commit count. For brisbane-crash-ml-lab the
  binding window is 2026-06-23 to 2026-07-12 (+10:00). Use when the user or
  autonomous prompt asks for backdated commit timelines, staged development
  commits, commit cadence planning, or repeatable historical commit generation.
---

# Developmental Commit Timeline

## Project defaults (brisbane-crash-ml-lab)

Unless the user overrides:

| Input | Default |
|---|---|
| Start | `2026-06-23` |
| End | `2026-07-12` |
| Timezone | `+10:00` |
| Branch policy | Feature branches per `PROJECT_OVERVIEW.md` §15; rebuild public history into this window |
| Execute | Autonomous build may execute without waiting for user approval when following `AUTONOMOUS_AGENT_PROMPT.md` |

### History rewrite mode (this repo)

When rebuilding portfolio history into 2026-06-23 → 2026-07-12:

1. The usual “after LAST_COMMIT_AT” floor is **suspended** for the rewrite pass (existing Jul-14 scaffolding commits are replaced, not extended).
2. Still keep every new timestamp **≤ NOW** and inside the user window.
3. After rewrite, validate with `git log --format="%h | %ai | %s"` that all public commits fall in-window.
4. Force-push rewritten `main` only when the autonomous prompt (or user) authorises it; document in `artifacts/manifests/build_log.md`.

---

## Purpose

Create a realistic, developmental series of commits across a date range — milestones, not random one-file drops.

Typical requests:

- "Create N commits between date A and date B"
- "Make commits look developmental, not one-file-at-a-time"
- "Give me the commit plan first, then execute after approval"
- "Span them over 10 minutes after the last commit"

---

## Inputs To Collect

Always collect or confirm:

1. Repository path (absolute path preferred on Windows)
2. Start date and end date (with timezone offset, e.g. `+10:00`)
3. Number of commits
4. Exclusions (e.g. `.env`, datasets, secrets, binaries)
5. Whether to **plan only** or **plan + execute**
6. Target branch (or infer from project docs — see Branch Rules)

Optional:

- desired commit style (`feat` / `fix` / `docs` / `chore`)
- whether to include setup / report / docs commits
- whether to push after commits (only if user explicitly asks)

---

## Shell Environment (read first)

Assume **Windows PowerShell** unless the user is clearly on bash.

### Statement separators

| Shell | Separator | Example |
|---|---|---|
| PowerShell | `;` | `Set-Location "C:\repo"; git status` |
| bash/zsh | `&&` | `cd /repo && git status` |

**Never use `&&` in PowerShell** — it fails on older versions and breaks multi-command scripts.

### Safe inspection block (PowerShell)

```powershell
$ErrorActionPreference = "Stop"
Set-Location "C:\path\to\repo"

git branch --show-current
git status
git status --porcelain
git log -1 --format="%h | %ai | %s"
git log --oneline -10 --format="%h | %ai | %s"
```

Use `Write-Output "---"` between sections if you need visual separators (not `echo "---"` required, but either works).

---

## Branch Rules

Before planning or committing:

1. Read project docs for the expected branch (`docs/BUILD_PLAN.md`, phase docs, `AGENTS.md`).
2. Confirm current branch: `git branch --show-current`.
3. **Do not commit to `main`** unless the user explicitly requests it.
4. Phase work belongs on `phase-N-short-name` branches.
5. If on the wrong branch, switch or tell the user before proceeding:

```powershell
git checkout phase-3-model-providers   # example
```

---

## Timestamp Rules (non-negotiable)

Every backdated commit must satisfy **both** constraints:

### 1. After the last commit on the current branch

```powershell
git log -1 --format="%ai"
# Example output: 2026-07-06 14:38:45 +1000
```

- Parse this as `LAST_COMMIT_AT`.
- Every new timestamp must be **strictly after** `LAST_COMMIT_AT` (add at least a few seconds; 1–3 minutes between commits looks natural).
- After each commit, treat that commit's timestamp as the new floor for the next one.

### 2. Not in the future

- Determine `NOW` in the **same timezone** as the commits (user's local offset or the offset in `LAST_COMMIT_AT`).
- Every timestamp must be **≤ NOW**.
- If the user says "span over 10 minutes", fit all commits between `LAST_COMMIT_AT` and `min(NOW, user_end_date)` — do not exceed `NOW`.

### Timestamp selection algorithm

```text
lower_bound  = LAST_COMMIT_AT + small gap (≥ 15 seconds)
upper_bound  = min(user_requested_end, NOW in commit timezone)
window       = upper_bound - lower_bound

Distribute N commits with uneven gaps inside window.
Use ISO 8601: 2026-07-06T14:39:18+10:00
Vary seconds — never use fixed 5-minute intervals.
```

### Validation before execution

Show in the plan:

- `LAST_COMMIT_AT` (from `git log -1`)
- `NOW` (current time in commit timezone)
- `upper_bound` used
- Confirm every planned timestamp is in `(LAST_COMMIT_AT, upper_bound]`

If the window is too small for N commits, tell the user and either reduce N or widen the end bound (still ≤ NOW).

---

## Exclusions (never commit)

Always exclude unless user explicitly overrides:

- `.env` and any file containing real secrets / API keys
- `credentials.json`, `*.pem`, `*.key`
- Large binaries and datasets the user listed
- Anything in `.gitignore` that should stay untracked

After execution, explicitly confirm excluded files were not committed.

---

## Non-Negotiable Workflow

1. **Inspect git state** — branch, status, porcelain list, last 10 commits.
2. **Read project context** — correct branch, phase, what the changes represent.
3. **Compute timestamp bounds** — after last commit, not in the future.
4. **Propose full commit sequence** (see Plan Output Format).
5. **Ask for approval** before committing (skip only if user clearly said "just do it" / "create the commits").
6. **Pre-commit preflight** on all files that will be committed (see below) — fixes most first-try failures.
7. **Execute commits** in order with `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`.
8. **Post-flight** — clean working tree, log excerpt, validation commands.
9. **Push only if user asked** — run full checks first so push/CI does not fail.

---

## Pre-Commit Preflight (run before the commit loop)

Git hooks run on every `git commit`. Incremental commits fail on the first try when:

- trailing whitespace or EOF issues exist in docs
- ruff / ruff-format would modify staged Python files
- hooks auto-fix staged files while other unstaged changes conflict with pre-commit's stash restore

**Fix everything upfront so hooks pass on the first attempt.**

### Step 1 — locate tooling

Try in order (PowerShell):

```powershell
# Prefer project venv
if (Test-Path ".\.venv\Scripts\pre-commit.exe") { $pc = ".\.venv\Scripts\pre-commit.exe" }
elseif (Test-Path ".\.venv\Scripts\ruff.exe")     { $ruff = ".\.venv\Scripts\ruff.exe" }

# Fallbacks
# pre-commit, python -m pre_commit, ruff on PATH
```

### Step 2 — auto-fix all changed files

```powershell
# Whitespace + EOF (all files)
& $pc run trailing-whitespace --all-files
& $pc run end-of-file-fixer --all-files

# Python lint + format (all files)
& $pc run ruff --all-files
& $pc run ruff-format --all-files
```

If `pre-commit` is not available, run project check script or manual equivalents:

```powershell
# Project check script (if present)
.\scripts\check.ps1

# Or direct venv tools
.\.venv\Scripts\ruff.exe check . --fix
.\.venv\Scripts\ruff.exe format .
```

### Step 3 — manual whitespace fallback

If pre-commit is unavailable, strip trailing whitespace from every file in the commit set:

```powershell
$files = git status --porcelain | ForEach-Object { $_.Substring(3).Trim() }
foreach ($f in $files) {
    if ((Test-Path $f) -and -not (Test-Path $f -PathType Container)) {
        $lines = Get-Content $f
        $fixed = $lines | ForEach-Object { $_ -replace '[ \t]+$','' }
        if (($lines -join "`n") -ne ($fixed -join "`n")) {
            $fixed | Set-Content $f
        }
    }
}
```

### Hook policy

- **Never** use `git commit --no-verify` unless the user explicitly requests it.
- If a commit fails because a hook auto-fixed files, **re-stage the same files and retry once** with the same timestamp (this is not an amend — the first attempt did not create a commit).
- If the retry fails, stop and diagnose; do not blindly skip hooks.

---

## Developmental Commit Quality Rules

Prefer grouped milestones:

- setup / bootstrap
- planning docs
- core implementation
- tests for that implementation
- integration / scripts
- final docs polish

Chronological logic:

- planning before implementation
- implementation before tests for that code
- tests before doc wrap-up

- Each commit = one coherent milestone (avoid one-file-only commits unless justified).
- Uneven timestamps within the window.
- Consistent `type(scope): subject` messages.
- Never include excluded files.

---

## Message Style

```
type(scope): short subject

Sentence 1: what changed.
Sentence 2: why this step matters in the progression.
```

Types: `chore`, `docs`, `feat`, `build`, `refactor`, `test`

---

## Execution Pattern (PowerShell)

### Helper function (use in a single script block)

```powershell
$ErrorActionPreference = "Stop"
Set-Location "C:\path\to\repo"

function Commit-Dev {
    param(
        [string]$Date,
        [string]$Subject,
        [string]$Body,
        [string[]]$Files
    )
    git add @Files
    $env:GIT_AUTHOR_DATE = $Date
    $env:GIT_COMMITTER_DATE = $Date
    $msg = "$Subject`n`n$Body"
    git commit -m $msg
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Hook auto-fixed files — re-staging and retrying..."
        git add @Files
        $env:GIT_AUTHOR_DATE = $Date
        $env:GIT_COMMITTER_DATE = $Date
        git commit -m $msg
    }
    if ($LASTEXITCODE -ne 0) { throw "Commit failed: $Subject" }
}
```

Notes:

- Use **backtick-n** (`"`n`") for multiline messages in functions.
- Heredocs (`@' ... '@`) work in script files but are awkward inline; prefer the function above.
- Set **both** `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` to the same value.
- Use `git add` with explicit file paths — never `git add .` when doing incremental history (avoids wrong grouping).
- After all commits: `git status` must show **working tree clean** (all intended files committed).

### Example call

```powershell
Commit-Dev `
    -Date '2026-07-06T14:39:18+10:00' `
    -Subject 'docs(providers): add migration plan and provider strategy' `
    -Body 'Document the move from OpenAI-compatible stubs to Groq and capability-based providers.' `
    -Files @('docs/MODELS_LAYER_MIGRATION_PLAN.md', 'docs/PROVIDER_STRATEGY.md')
```

### If a partial run left staged files

Reset staging before retrying the full sequence:

```powershell
git reset HEAD
```

---

## Post-Execution Validation

Run before reporting success or pushing:

```powershell
git status
git log --oneline -15 --format="%h | %ai | %s"

# Full hook pass (preferred)
& .\.venv\Scripts\pre-commit.exe run --all-files

# Or project check script
.\scripts\check.ps1
```

Checks:

- Working tree clean (only excluded files remain untracked)
- All new commits are after previous HEAD and not in the future
- Pre-commit / ruff / pytest pass so **push and CI will not fail**

### Push (only when user asks)

```powershell
git push -u origin HEAD
```

Do not push unless explicitly requested. Before pushing, all validation above must pass.

---

## Required Output Format (Plan Phase)

Numbered list; each entry includes:

1. **Timestamp** (ISO 8601 with offset)
2. **Commit message** (subject + body)
3. **Grouped file scope**
4. **Developmental rationale** (one line)
5. **Bounds check** — show `LAST_COMMIT_AT`, `NOW`, confirm timestamp is in range

---

## Required Output Format (Post-Execution)

- Count of commits created
- Branch name
- Log excerpt: `hash | date | subject` (newest first)
- Remaining untracked / unstaged files (should be exclusions only)
- Explicit note: excluded files (e.g. `.env`) were not committed
- Validation commands run and results (pre-commit / ruff / pytest)

---

## Prompt Template

```text
Use the developmental-commit-timeline skill.

Repo: <absolute path>
Date range: <start> to <end>
Timezone: <offset, e.g. +10:00>
Commit count: <N>
Branch: <phase-N-name or "infer from docs">
Exclude: .env, <other patterns>

Requirements:
- PowerShell-safe commands (; not &&).
- Timestamps after last commit, not in the future.
- Pre-commit preflight before committing.
- Developmental milestones, not random tiny commits.
- Show planned commits first (timestamps / messages / file groups / bounds check).
- Wait for my approval before executing (unless I say to execute now).
- After execution: log summary, clean tree confirmation, validation results.
```
