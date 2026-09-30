# Research workflow

Updated: 2026-10-01. Route currents own priorities and evidence; `AGENTS.md` owns
execution, integrity and delivery rules. This workflow adds no approval gate.

## Choose useful work

Name the capability gap/opportunity, credible baseline and decision the result
will change. Choose by task effect, reliability and cost; mature methods, better
data, integration and new algorithms are all valid. Prefer the simpler approach
when effects are comparable. Search literature/history to resolve a concrete gap.
Keep each comparison interpretable; broader exploration may examine different
hypotheses. Add an ablation when it changes the contribution judgment, and revisit
repeated local patches when a common mechanism could explain them.

## Work phases

| Phase | Smallest useful work | Decision |
| --- | --- | --- |
| Explore | Implement and inspect a paired task-effect check on Development. | Iterate with recorded changes; no registration, protocol, independent audit or terminal by default. |
| Engineering | Fix an execution, source, integration or recovery problem; inspect a representative smoke/replay. | Correct execution is not proof of algorithm gain. |
| Confirm | Test a fixed method and comparison against a stated claim. | Register, fix criteria/access/retries, and choose independence suited to the claim. |

An exploratory next-step decision or reversible fix is not formal promotion.
Protected access and claim-critical numbers use [formal governance](../docs/formal/RESEARCH_GOVERNANCE.md).

## Data and run records

Synthetic or simulated data is eligible Development unless explicitly marked protected or final.
This eligibility does not reopen an explicitly stopped run or change frozen criteria.
Existing real Development retains its documented access and use terms. During
exploration, training/validation inputs can guide method and parameter selection;
record their role/identity once and disclose selection. Use the fixed benchmark by
default; supplements must fit the authorized question and budget. New seeds/pixels
alone do not establish unseen-structure transfer or fresh confirmation.

Each active route appends one row per run to `research/active/<route>/RUNS.md`:
date, code version, configuration changes, metrics with denominators/units, conclusion.
Current logs: [forward perception](active/dtr-r0/RUNS.md) and
[hardware support](active/hardware-bringup/RUNS.md). Paused routes create their log
when resumed. Link payloads or a necessary detailed report from the row; ordinary
attempts need no separate result document. Record a not-run metric as `NOT_RUN`.

For synthetic-source defects, repair and regenerate/replay the same Development
scenes/seeds, preserve old outputs, and append the repair result to the route log.
No forced fresh cohort or one-shot run; without an evaluable algorithm run, a source
failure gives no algorithm verdict. Use hashes for immutable identity, transfer
verification or a concrete integrity issue.

## Governed asset reuse

For existing governed UE/nearfield bundles, search assets before capture and use
`tools/ba.ps1 run research-ue -RunSpec <run-spec.json>`. The spec declares `reuse.mode`,
`reuse.query`, exact input subpaths and roles; the runtime checks [input contracts](../data/ue-reuse-policy.json)
and records lineage. See [UE reuse](../docs/asset-management/UE_REUSE.md) for adapters
and Core regression. A mixed reserved bundle needs an admitted subset; `split=train`
cannot unlock protected files. Synthetic debugging outside these bundles follows
the engineering loop above, without this asset-admission gate.

## Finish the decision

Retain and integrate useful gains. For no gain, distinguish hypothesis failure,
implementation defect, insufficient opportunity and an unsuitable comparison;
revise, simplify or stop. For not-evaluable work, report the missing evidence and
continue independent authorized tasks. Report effects and costs before gate labels;
zero correct and zero wrong commits do not establish perfect identification.

Formal runs use `python tools/knowledge.py register-experiment`; never append
`experiments/index.jsonl` manually. Mainline/baseline or governed reuse decisions
use `python tools/knowledge.py set-terminal-inheritance` under formal governance;
archived registrations link a `--decision-id` with complete inheritance.

## Knowledge maintenance

For template-only configuration changes, retain previous bytes and run
`python tools/refresh_decision_templates.py --previous-config PATH --check`.
Apply without `--check` only after eligibility passes; this preserves cached outcomes,
not ledger validation. Source/retrieval drift or an already stale cache needs a full
rebuild. Install the knowledge hook once with
`pwsh -NoProfile -File scripts/refresh_knowledge.ps1 -InstallHook`.
