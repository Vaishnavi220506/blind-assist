# SkySynth Auditor component pilot

## Frozen question and scope

Does the official SkySynth Auditor role prompt find more reproducible metric
defects than direct Codex, with the same source, specification, model and call
ceiling? This is an EXPLORE engineering mutation pilot, not a full SkySynth
workflow, protected confirmation, or a claim that production code has these bugs.

The component is a frozen copy of `ba_camera_corridor_metrics.evaluate_rows`.
Eight evaluator-owned, single-fault mutants and one combined candidate cover
UNKNOWN truth/prediction, frame/timestamp validation, pre-entry detection,
sampled false-alert duration, event splitting, and boundary accounting.
The original reference passes eight manually specified probes; each corresponding
single-fault mutant fails its probe before either model is dispatched.
The combined candidate and complete contract are identical between arms. The
reference, seed identities, and scoring probes are not in either model prompt.

## Budget and fidelity

- Direct Codex versus the official Auditor brief at SkyDiscover `269c9e6`.
- `gpt-5.6-sol`, medium reasoning, one fresh CLI invocation per arm, 360 seconds
  each; order direct then auditor; no tool use, retry, repair, or score feedback.
- Both return at most 12 findings, at most two declarative executable probes each.
- Same call/time ceilings, not equal consumed tokens. Record actual CLI usage
  and elapsed time; do not invent monetary cost or exact causal speed differences.
- This isolates the **prompt component**. The full SkySynth knowledge base,
  iterative challenge, independent completeness pass, production reviews and
  test-authoring agents do not run. Parent executes the declarative tests after
  both submissions. No full SkySynth or `production-ready` claim is permitted.
- Journal before dispatch. A dispatched slot is consumed even on failure;
  interrupted uncertain calls are `in_doubt`, never automatically rerun.
  No iteration-level resume; at most one 360-second call can be lost at a time.

## Adjudication

A submitted finding is accepted only if all its probes pass the unchanged
reference and at least one fails the combined candidate. Separately measure the
union of individual seeded mutants killed by those validated probes (out of 8).
Report invalid findings, duplicate coverage, usage and wall time. A failed probe
is not automatically proof that the underlying textual concern is false.

Keep the Auditor component as promising only if it catches at least two extra
distinct seeded faults with no extra invalid findings. A tie favors the simpler
direct prompt. Any forbidden tool call invalidates the comparison. A one-component,
one-sample pilot cannot rank the full systems or establish statistical superiority.

All payloads and receipts: `artifacts.local/evidence/skysynth-audit-20260922/`.
Harness: `scripts/research/skysynth_audit_pilot.py`. CPU-only scalar evaluation
is TASK_NOT_GPU_SUITABLE. No remote worker, training, original metric change,
alert recipe change, or Android promotion is part of this task.

## Result: incremental value not established

| Measurement | Direct Codex | Official Auditor prompt |
| --- | ---: | ---: |
| Model invocations / forbidden tool calls | 1 / 0 | 1 / 0 |
| Submitted findings | 9 | 8 |
| Findings whose complete probe bundle passes reference and fails combined candidate | 8 | 8 |
| Distinct individually seeded faults caught by those bundles | 6/8 | 7/8 |
| Probe bundles failing reference | 1 | 0 |
| CLI elapsed seconds | 125.02 | 182.82 |
| Input tokens | 19,032 | 22,759 |
| Output tokens (CLI field, including reported reasoning) | 3,317 | 4,920 |
| Reported reasoning-output tokens (subset, not added again) | 2,070 | 3,549 |

The frozen decision is `INCREMENTAL_VALUE_NOT_ESTABLISHED`: +1 distinct fault,
below the required +2. The Auditor used approximately 24% more input+output tokens
and 46% more CLI time in this single sequential sample. Neither is a stable cost
or speed estimate; monetary billing is unavailable. Both invocations completed
and their processes exited; no model retry or repair was performed.

The extra coverage is M04 (timestamp equality). Both arms describe M03 (missing
frame-index validation), but choose probes whose timestamps also mismatch the
enumerated row position. Consequently the single M03 mutant still rejects the
input through its intact timestamp check. A prose finding is not a discriminating
regression test. Direct F01/F02 also duplicate coverage of M01; findings are not
the same as unique defects.

Direct F09 combines a useful numeric timestamp probe with a string timestamp
requiring ValueError. The frozen reference raises TypeError on that string.
Thus the full bundle is rejected by the predeclared reference-pass rule. This
is a **contract/reference ambiguity, not an established false-positive bug
claim**: the common prose says bad timestamps raise ValueError but does not
explicitly bound timestamps to numeric inputs. Do not describe the 1/0 failed
bundles as a clean false-positive-rate comparison. If the numeric probe is
examined alone, it catches M04 too; that is posthoc diagnostic evidence only,
not a replacement for the frozen 6/8 versus 7/8 result. Both arms' narrative
findings cover the eight seeded mechanisms; the observed difference concerns
submitted test quality and grouping, not discovering a mechanism the other missed.

## Validation, retention and boundaries

- Before dispatch: all 8 hand-specified probes pass the reference and fail their
  respective single mutant. Original metric unit tests: 7 passed. Experiment
  evidence-gate tests: 5 passed.
- All prompts, source copies, individual mutants, expected probes, CLI JSONL,
  token usage, source hashes and terminal receipts are retained in the artifact
  root. Models saw only the common source/contract and their arm's instructions.
- Official `validate_test.py` verifies seven submitted bundles in each arm;
  these cover six distinct mutants for direct and seven for Auditor. Both
  frame-index bundles are rejected for catching no isolated mutant. Direct's
  mixed timestamp bundle cannot validate because its reference fails. The
  initial direct boundary probe unexpectedly returned no mutant kill; one
  focused mechanical recheck of the unchanged probe/reference/M08 passed.
  Its original receipt and discrepancy-check log are retained; the initial
  shell run does not expose a cause, so none is asserted. The in-process frozen
  adjudication and final focused official check agree. No model rerun occurred.
  The Windows adapter explicitly selects Git Bash because bare
  `bash` resolves to an unavailable WSL shell before PATH. UTF-8 and shell
  resolution failures are retained; no model or metric inputs were changed.
- Global registration was attempted through `tools/knowledge.py`; it was blocked
  by the existing `experiments/index.jsonl:303` fingerprint mismatch. The local
  registration receipt records that gap. No global ledger or scientific-route
  terminal was modified or inherited around the error.
- Retain this as an engineering diagnostic COMPONENT: runnable mutation corpus,
  probes and adjudication. It does not promote the Auditor prompt to the default.
  The native SkySynth iterative workflow remains **not evaluated**. There is no
  rejection of that wider workflow, no automatic second cohort, and no change
  to near-field algorithms or the original metrics module.
- Final integrity check verified original source and both output/JSONL hashes
  against frozen receipts; owned model/validation processes are absent. Durable
  corpus, logs and failed validation receipts remain for reproducibility; no
  temporary service, remote allocation or paid worker remains active.

The useful practice is to require both a passing reference and a failing isolated
mutant before accepting a regression test. The pilot also exposed the need to
specify malformed-input behavior before comparing audit methods. Preserve the
original frozen result rather than fixing the specification after seeing answers.
