# Support-bound one-step event inheritance

2026-09-22. User-authorized EXPLORE; one candidate, no fit, capture or threshold
selection. This protocol is fixed before this run opens observation payloads.

## Question and mechanism

Can an alert's original observable support qualify a short continuation more
usefully than a blind one-frame hold? The old DTR distinction between starting
an event and inheriting an event is reused as a mechanism, not as evidence that
the current ToF stream has target identity. This is not the closed 2026-09-21
conditional-hold recipe: it does not select another original-score keep cutoff,
recurse, or use a learned logit. Birth remains the exact A current decision.

All arms use unchanged ToF score/decision code, threshold
`0.4071309640537889`, definite-support bypass and current-frame UNKNOWN.

1. `A_current`: unchanged current decision.
2. `A_hold`: current OR the immediately preceding current decision in this clip.
3. `support_inherit`: current OR a qualified one-step inheritance described below.

A genuine current trigger stores its frame/time and all decisive original zones:
definite zones, or possible zones whose joint score reaches the unchanged A
threshold. Only the next adjacent sample may inherit, within 200,000,000 ns of
that capture time. A held alert never refreshes the seed. To inherit, at least
one current valid return must be in the SAME original decisive zone, its complete
support interval must overlap the original support interval with positive length,
its zone support must still be possible under the unchanged camera-forward query,
and its current measured range centre must lie in the fixed `[0.3, 3.0] m` slab.
No current range prediction, range extrapolation, threshold fit, neighboring-zone
association or invisible return is used. Missing/invalid return, nonadjacent
sample, interval mismatch, all applicable centres outside the slab, or expiry
ends inheritance. Every A current alert is retained unconditionally.

The centre-in-slab predicate is a fixed continuation qualifier, not proof that
the whole support is inside the corridor or certified free space outside it.
Same-zone interval compatibility is observable consistency, NOT an object ID:
two different objects with indistinguishable returns remain unresolvable. It
does not merge different zones into the old event. Clip boundaries are observable
recording resets; evaluator positive-event boundaries never reset inference.

## Inputs, separation and complete clips

Reuse all 1,728 consumed query-occupancy frames: 48 layout groups, 144 clips,
12 samples per clip at 0.2 s. Preserve original train/dev/evaluation roles solely
for reporting. The original 576-frame/16-layout evaluation role is the primary
effect check; 864 train and 288 dev frames are disclosed secondary diagnostics,
not selection data. No role, family, relation, truth or native contributor enters
the candidate. Runtime input is only frame/clip identity, capture time, fixed zone
footprints and observed ToF ranges. The prepared float32 range normalization by
8 is exactly inverted; zone boundaries are checked against `boxes45()`.

Existing UE contracts admit exact prepared observation subpaths,
`materialization.json` as configuration and three label NPZs as evaluator data.
Execute through `tools/ba.ps1 run research-ue -RunSpec <saved-spec>`; no policy
change is needed. Bind input/source hashes, seal all 1,728 predictions before
opening label arrays, then join centre BODY/HEAD query truth exactly as the
original evaluator. Verify A parity against stored baseline after the seal.
No RGB/native depth, model weights, protected data or prior learned output is read.

## Fixed decision and reporting

Retain this candidate only if, on the primary evaluation role, it retains at
least 75% of the extra TP supplied by A_hold and removes at least 50% of A_hold's
extra FP relative to A_current. Require at least 8 added TP across 4 distinct
layout groups, no A_current true decision removed, no event detection lost or
onset delayed, and no increase in false segments against A_hold. A zero gain
denominator is not a pass. These are local usefulness criteria, not safety gates.
Otherwise preserve the exact recipe as NEGATIVE_CONTROL; report partial tradeoffs
and whether insufficient observable continuation explains the result. No retry,
parameter adjustment, alternative candidate or automatic successor follows.

Report full frame TP/FP/FN, precision/recall/FPR with unknown-negative abstentions,
per-event onset/coverage/internal and terminal gaps, pre-entry alerts, exit tails,
false segments/duration, every paired gain/loss ID, source/family/layer/relation
strata, per-frame support decisions and UNKNOWN parity. Durations are sample
counts, not continuous walking or measured warning latency. IoU is not applicable
to this alert-only decoder. CPU scalar scoring is TASK_NOT_GPU_SUITABLE; record
actual backend and runtime. No continued worker or paid allocation is needed.

Focused analytic controls cover reappearance after a missed sample, same/different
zone second targets, interval mismatch, time expiry, clip reset, nonrecursive
hold and preservation of fresh current triggers. They verify decoder mechanics,
not algorithm effect. Independent audit reconstructs each decision from public
return supports and recalculates counts, event timing and gates from saved outputs.
The historical 7-frame bbox trichotomy is NOT_EVALUABLE here because no real
detector tracks are available; evaluator object IDs must not substitute for them.

Evidence remains same-generator controlled-object simulation Development.
No natural-scene, physical ToF, user-benefit, Android or safety claim is supported.
Mechanical corrections may rerun only a failed task-owned stage with its original
logs and hashes retained, without changing candidate, data or scientific rules.
