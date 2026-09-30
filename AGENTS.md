# BlindAssist agent map

## Project

BlindAssist is an Android research demo and thesis project, not a certified safety product.
Choose methods by measured benefit, stability and cost; novelty claims need evidence.

Keep ownership stable: `:app` shell/assets, `:feature:assist` runtime, `:core:assist` risk,
`:core:vision` detection, `:core:device` adapters, `:core:ui` UI.

## Context routing

1. Read [project state](docs/PROJECT_STATE.md) for priorities or ownership;
   start a self-contained fix at its known file. Reuse context already read.
2. For research work, read `docs/CURRENT_DECISION.md` and the affected route
   `CURRENT.md`; skip route loading for unrelated code or documentation changes.
3. If baseline, inheritance, or failure context is missing, use
   `python tools/knowledge.py context --route <route> --limit 4 --query <question>`.
   Do not repeat unchanged lookups.
4. Open the detailed route `README.md`, code, test, or contract only as needed.
5. Check `git status --short` before editing/staging; use
   `scripts/show_worktree_scope.ps1` only when ownership is unclear.

Historical gates retain their tested scope. Explain changed mechanisms and useful
checks; preserve route authority and consumed evidence.

## Execution policy

Default to a small working implementation and a task-effect check, not a review chain.
The user decides new questions, budget expansions and changed frozen criteria.
Within an authorized question/budget, routine mechanism comparisons, Development
iteration, recovery and delivery need no new approval. Preserve explicit stop rules;
a failed experiment ends that run, not every different mechanism in the same question.

- **Fast lane (default):** diagnostics, engineering and pilots. No registration,
  separate protocol or independent review by default. Reuse eligible Development
  for iteration and synthetic-source repair; no one-shot/cohort churn by default.
  Respect existing frozen rules. Repeat identity audits only for changed inputs or
  concrete integrity evidence. Short existing result records suffice; tables optional.
- **Formal lane:** mainline/baseline promotion or confirmatory paper claims. Freeze
  criteria, register, check identity and reproduce with independence suited to the
  claim. Follow [research workflow](research/WORKFLOW.md), keeping train/calib/eval
  separate. Consumed diagnostics cannot become fresh confirmation retrospectively.
- An exploratory next-step decision or a reversible engineering fix does not by
  itself require formal reproduction. When uncertain, start in the fast lane.
- Add a precheck, review, test or abstraction only for an observed defect, explicit
  acceptance requirement, consequential action or decision-changing evidence gap.
  Name that reason briefly when expanding work; do not create a new approval gate.
- Start with one meaningful falsifier or focused check; stop when covered. Repeat
  or broaden only for changed inputs, defects or material integration/evidence gaps.
- Implement the current need directly. Do not add speculative fallback paths,
  configuration layers or frameworks; keep necessary error/data-integrity handling.
- From **v1.3**, pilot G2 quota shortfalls do not stop downstream; G0 identity/axes
  and G1 duplicates still stop. All gates stay hard at scale-up; **v1.2 is unchanged**.
- Lead with effect, cost and next decision; consolidate limits once. Keep currents
  near 3KB; update for route decisions, archive prior text, keep history in results.
Assess independent units/power before inferential collection; small pilots do not
establish generalization. Development reuse is not fresh confirmation. Use the benchmark;
supplements stay within authorized scope. Public access grants no extra data rights.
Use `FINAL` before protected blind/final access or claim-critical paper numbers and
[research governance](docs/formal/RESEARCH_GOVERNANCE.md). Use `EXTERNAL` for release,
deployment, credentials, privacy, destructive external actions or real-user safety
claims. Missing deployment evidence limits claims, not authorized reversible work.

## Integrity and evidence

- Never fabricate measurements, provenance, labels, licenses, credentials,
  consent, user decisions, authorization, or objective truth.
- Keep public goal identity, evaluator-only truth, proposal, selection, and
  handoff/persistence as separate authorities.
- `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence.
- Name synthetic, replay, pseudo-labeled, model-reviewed, device, and natural
  evidence accurately; curated Development is not universal product or safety
  performance.
- Preserve failed/consumed terminals. Reuse permits diagnostics, regression, or
  disclosed Development, never fresh confirmation authority.
- Formal inheritance applies to terminals changing mainline, baseline or reuse
  decisions; ordinary exploration needs a clear conclusion, not a new terminal.
  Preserve existing records and use supported tooling; roles and obligations are
  defined in [research workflow](research/WORKFLOW.md).
- Do not leak protected outcomes, silently change denominators, hide collapsed
  coverage, or read evaluator truth from observations.

## Task routing

| Task | Read next |
| --- | --- |
| Algorithm/model/data experiment | One route current, then its owning code/result |
| Protected blind/final claim | [Research governance](docs/formal/RESEARCH_GOVERNANCE.md) |
| Android/CameraX/UI/module code | [Code map](docs/CODE_MAP.md) |
| Device/ADB/latency/stability | [Device regression](docs/DEVICE_REGRESSION.md) |
| Release/APK/archive | [Release and verification](docs/RELEASE_AND_VERIFICATION.md) |
| Hardware/glasses/ESP32/network | [Hardware route](docs/GLASSES_HARDWARE_ROUTE.md) |
| Documentation/layout/artifacts | [Document governance](docs/DOCUMENT_GOVERNANCE.md) |
| Long/remote compute | [Host research compute](docs/HOST_RESEARCH_COMPUTE.md) |
| SkyDiscover search | [SkyDiscover playbook](docs/SKYDISCOVER_PLAYBOOK.md) plus the owning route |

## Tools and compute

- Prefer Exa for external search, literature discovery, and multi-source research when available.
- Run Android/Gradle through `pwsh -NoProfile -File scripts/run_android_gradle.ps1 <tasks...>`.
- Register formal experiments with `python tools/knowledge.py register-experiment`; never append `experiments/index.jsonl` manually.
- Use `python tools/knowledge.py set-terminal-inheritance`; archived registration
  links `--decision-id` to a terminal with complete inheritance.
- Use `pwsh -NoProfile -File tools/ba.ps1 doctor <profile>` for an affected prerequisite or failure, not as a per-task gate.
- Validate changed surfaces: `git diff --check`, structure for layout, docs index for hot links.
- Use [host compute](docs/HOST_RESEARCH_COMPUTE.md) for GPU-first placement, CPU exceptions and backend evidence.

## Ownership and delivery

- Pre-existing/concurrent changes are user-owned; edit/stage only task-owned paths or hunks and never revert/reclassify unrelated work.
- Uncommitted files and untracked candidates are WIP, not route authority.
- Payloads and generated outputs stay under ignored `artifacts.local/`; on managed Windows it remains the canonical junction in [local artifacts](docs/LOCAL_ARTIFACTS.md). Never create a physical workspace-drive bypass; run hygiene after storage-path changes.
- Never rewrite history, force-push, delete branches, change remotes, or perform destructive actions without explicit authorization.
- Deliver routine research directly to the default branch unless requested otherwise; verify remote parity and never absorb unrelated changes.

Report results first in plain Chinese: what improved, its cost and the decision;
then give labeled metrics with denominators/units. Avoid cryptic compressed counts;
tables are optional. Completion requires the outcome, scoped checks or stated gaps,
applicable inheritance, reviewed diff and release of task-owned resources.
