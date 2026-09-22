# Fixed-entry conditional hold: one selection rescue, none on evaluation

2026-09-21. **Negative result for this frozen decoder contract.** Selecting only
an original-score continuation threshold adds one Boundary positive in selection
and zero in evaluation. Evaluation exactly equals original_current, including
all alert identities: Core82/11/2, Boundary16/0/68. Keep the demo/baseline and stop
this recipe; do not lower entry thresholds, train, choose another head or retry.

[Protocol](CONDITIONAL_HOLD_PROTOCOL_20260921.md),
[runner](run_conditional_hold.py), [tests](test_conditional_hold.py).
The user separately authorized the previously untested state-conditioned decoder
question. Earlier current-only, readout and RGB negative results are unchanged;
they were not themselves tests of this recurrence.

## Fixed inputs and one selected threshold

Consumed same-simulator Development, SELECTION g00/g01 and EVALUATION g02/g03:
eight complete layouts,24 clips and576 frames per role. No original FIT content
or reserved-test access. Original saved scores only; no fitting or inference.
Clip starts are observable reset boundaries, not evaluator event boundaries.

The original entry cutoff remains15.769264221191408. A_current may also start or
reassert the state; no A/current trigger can be removed. At each frame:

    trigger = A_current OR original_score >= tau_on
    alert = trigger OR (previous_alert AND original_score >= tau_keep)

This is recursive continuation, not the old nonrecursive one-frame extension.
UNKNOWN is copied unchanged, including while an alert is active. Thresholds are
logits, not probabilities. Complete evaluation predictions were sealed before
evaluation truth/native joins; hashing label files for provenance is disclosed.

The552 canonical selection thresholds enumerate every distinct >= behavior on
selection, including tau_on as the no-continuation control. Five are feasible
under zero added negative flags. Maximizing Boundary TP, then Core TP, then the
highest threshold selects **tau_keep=15.535846710205078**. It recovers only
f0679, an internal positive at1.4s in body_protruding_plane_g01 with30 observed
target-corridor samples. No evaluation threshold curve was computed.

The lowest feasible canonical threshold is15.52305793762207. The next lower
canonical value15.453608512878418 adds Core exit FP f0665 without recovering
another positive. With no FP constraint, the selection curve can recover26
Boundary positives; these are not eligible gains at the required operating point.

## Task effects

Counts below are TP/FP/FN. Core includes complete INSIDE and OUTSIDE clips;
Boundary is separate. Evaluation has84 Core positives/300 negatives and84
Boundary positives/108 negatives. Durations are sample count times0.2s.

| Evaluation policy | Core | Boundary | Boundary events | Core / Boundary false segments |
| --- | --- | --- | --- | --- |
| A_current | 82/11/2 | 4/0/80 | 2/8 | 10 / 0 |
| A + original, no hold | 82/11/2 | 16/0/68 | 3/8 | 10 / 0 |
| A + original, conditional | 82/11/2 | 16/0/68 | 3/8 | 10 / 0 |
| A + original, old hold | 84/19/0 | 18/3/66 | 3/8 | 13 / 3 |
| A, old hold | 84/19/0 | 6/2/78 | 2/8 | 13 / 2 |

Conditional Core precision/FPR=88.17%/3.67%; Boundary=100%/0%, recall19.05%.
Compared with original_current, evaluation adds/losses zero TP and zero FP,
preserves all FP identities, all event timing and contributors. Core false
duration stays2.2s and Boundary0s. Core retains one internal silent frame and
one initial silent frame; Boundary retains16 internal silent frames. Core
post-exit carryover remains five samples from the current baseline; Boundary
remains zero. Thus no *additional* tail is not absence of all existing tails.

Selection Core remains89/6/0; original Boundary10/1/79 becomes11/1/78,
with events3/8 unchanged and internal silence26->25. This one-frame gain does
not transfer. The12-frame evaluation gain over A_current comes entirely from
the pre-existing original-score supplement, not the conditional decoder.

Core current retention is not held-Core retention: the two positives f0221 and
f0227 recovered by old A_hold remain missed, as do held Boundary positives
f0751 and f1111. UNKNOWN remains selection Core324/Boundary192 and evaluation
Core332/Boundary192. Unknown silence is not certified free space. IoU N/A.

## Two distinct limitations

**Startup opportunity is limited.** In evaluation, five Boundary clips have no
current trigger anywhere, accounting for51 positive frames. One additional
positive precedes the first trigger in body_protruding_plane_g02. Of68 remaining
current misses,52 therefore cannot be recovered by any causal continuation of
these fixed triggers. Only16 are reachable after a trigger, eight in each of
body_protruding_plane_g02 and body_suspended_solid_g03. The third detected event,
head_hanging_plane_g02, already has12/12 current coverage. Even never releasing
would cover only32/84 Boundary positives, ignoring all false-tail costs.

Selection likewise has five unseeded Boundary clips (53 positive frames) and
26 reachable additional positives. Its body_suspended_solid_g00 first trigger
is a *negative* f0892; the opportunity record explicitly marks that false seed.
The never-release suffix is a diagnostic ceiling, not a proposed policy.

**One continuation threshold cannot satisfy the needed score inequalities.**
In the same selection Boundary event:

| Frame | Role in event | Original score | Required continuation threshold |
| --- | --- | ---: | --- |
| f0680 | Internal positive after an active prefix | 11.615569 | tau_keep <=11.615569 to retain |
| f0689 | First exit negative after current trigger f0688 | 14.757340 | tau_keep >14.757340 to release |
| f0665 | Core exit negative after current trigger f0664 | 15.453609 | tau_keep >15.453609 to avoid added Core FP |

The first two constraints already conflict within one event; cross-clip Core
preservation makes the global constraint tighter. Conditional state prevents a
weak OFF-state score from starting an alert, but does not reverse score ordering
between an internal positive and an exit negative when both follow ON states.
This explains why a narrowly lower threshold can rescue f0679 but not the main
weak interval. It is evidence against this same-score/global-keep recipe, not
all causal decoders, other representations or time information in general.

The fixed evaluation candidate criterion (>=3 additional Boundary frames across
>=2 events/layouts, no extra FP and current preservation) fails with zero gain.
It fails independently of the older4/8-layout gate; no gate revision is needed.

## Verification, disposition and evidence

Four focused tests pass: weak-gap preservation with immediate release,
recursive multi-frame continuation and equality, no spontaneous/cross-clip
start, and authoritative current-trigger preservation. [Independent saved-output
verification](verify_conditional_hold.py) passes:57 hashes/six seals, all552
selection candidates, optimal tie-break, both roles' path-based predictions,
independent full event metrics, native counts and opportunity decomposition.
The verifier reuses the independent current-only audit's metric implementation,
not the runner's production metrics. Scalar standard-library CPU replay is
TASK_NOT_GPU_SUITABLE; no model, GPU, paid allocation or persistent worker.

Evidence in `artifacts.local/work/ba-conditional-hold-20260921/` retains the
frozen protocol, all552 selection records, chosen threshold, prediction/evaluation
seals, complete frame/event metrics, opportunity decomposition and native changes.
Local inheritance role is NEGATIVE_CONTROL for fixed-entry original-score
single-threshold continuation at the no-added-FP budget. Do not present consumed
layout results as fresh confirmation, continuous-stream or hardware performance.
Supported global registration fails at the existing
`experiments/index.jsonl:303 input_fingerprint` mismatch; inheritance returns
`unknown terminal id: ba-conditional-hold-20260921`. Receipts and a structured
local disposition are retained; global metadata remains pending without bypass.
