# Positive-priority selector: implemented tradeoff on consumed synthetic data

2026-09-21. **The public selector is now executable and its single consumed
replay realizes the audited tradeoff:** on273 initially ambiguous cases it gives
56 positive answers,62 correct negative identifications and155 UNKNOWN. Fixed
scanning gives47/57/169; old adaptive46/68/159. It retains every positive answer
of both comparators, but gives up six negative answers of old adaptive. All32
out-of-prior cases still abstain. Retain a narrow diagnostic component, not a
hardware or general-purpose obstacle policy.

[Frozen brief](ACTIVE_VIEW_POSITIVE_PROTOCOL_20260921.md),
[preceding posthoc audit](ACTIVE_VIEW_OPPORTUNITY_AUDIT_20260921.md),
[selector](../../../../scripts/research/nearfield_opportunities/active_view_positive.py),
[tests](../../../../scripts/research/nearfield_opportunities/test_active_view_positive.py).

## What was newly run

The audit had already exposed this prior's achievable answer before implementation.
This is explicitly **posthoc consumed Development**, not a blind comparison,
independent confirmation or a newly discovered information ceiling. One new
selector replay processes all653 existing cases; fixed and old adaptive use
their authenticated original saved outputs. No old code, protocol, data, action
or label changed, and no second new-policy run occurred.

The unique rule minimizes positive unresolved hypothesis count, then total
unresolved count, then the existing fixed +X/-X/+Z/-Z order. There is no pair-cost
tie-break, weight, fitted parameter or truth-aware exception. Each candidate
action uses only the declared prior's predicted measurements; measured future
views are unavailable during selection. The selector accepts only initial range
bins and the public finite bank, not case ID, true scene or evaluator truth.

All653 choices are serialized and sealed **before the simulator parses true
source geometry**. It then generates only the chosen actual second observation
per case. All predictions are sealed before evaluator truth and old-policy
outcomes are joined. Stage receipts record this order. This demonstrates the
public-computable implementation, rather than copying oracle class decisions.

All policies receive two measurements and one0.12m translation. The corridor
remains fixed in the original camera frame, with full obstacle-extent labels.
The ideal eight-ray radial sensor, exact poses, quantization,621-state finite
prior and source geometries are unchanged. Negative answers remain finite-prior
nonintersection predictions, not observed or certified free space.

## Complete accounting

`FN no-IN` includes every positive without INTERSECTS, including UNKNOWN.
Explicit false negative commitments are zero for all three policies; the
positive UNKNOWN column accounts for these FN. Correct negative answers are
model-conditional identifications, not alert true-negative safety claims.

| Initial ambiguity273 (113+/160-) | TP | FP | FN no-IN | Correct negative | UNKNOWN + / - | Total UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| Fixed +X |47|0|66|57|66 / 103|169|
| Old adaptive pair split |46|0|67|68|67 / 92|159|
| Positive-priority |56|0|57|62|57 / 98|155|

Positive-answer coverage on the113 positives is41.59%,40.71%,49.56%, respectively.
The candidate's correct decisive count is118 versus104/114. Precision is100%
and FPR0/160 on this constructed in-prior subset; these are largely structural
consequences of including the noiseless true scene in the same model bank.

| All653 (360+/293-) | TP | FP | FN no-IN | Correct negative | UNKNOWN + / - | Correct decisive |
|---|---:|---:|---:|---:|---:|---:|
| Fixed +X |278|0|82|174|82 / 119|452|
| Old adaptive |277|0|83|185|83 / 108|462|
| Positive-priority |287|0|73|179|73 / 114|466|

| Candidate stratum | N (+/-) | TP / FP / FN no-IN | Correct negative | UNKNOWN + / - | Final no-bank-match |
|---|---:|---:|---:|---:|---:|
| In-prior |621 (344/277)|287 / 0 / 57|179|57 / 98|0|
| Off-grid |24 (12/12)|0 / 0 / 12|0|12 / 12|19|
| Changed wall |8 (4/4)|0 / 0 / 4|0|4 / 4|8|

Both out-of-prior strata remain identical to the comparators:32 UNKNOWN, no
correct decisive answer, no wrong decisive answer. This is zero coverage, not
successful robustness. The exact-match finite bank remains the main limitation.

## ID retention and cost

Against fixed, all278 original positive IDs remain, nine are added. Six negative
identifications are added, one becomes UNKNOWN: `in_0608`. Thus +9TP/+5 negative
answers is aggregate improvement but not retention of every negative answer.
The nine new positive IDs are `in_0044,in_0103,in_0142,in_0466,in_0507,in_0531,
in_0548,in_0564,in_0576`.

Against old adaptive, all277 positive IDs remain and ten are added. No negative
answer is added; six become UNKNOWN: `in_0008,in_0063,in_0102,in_0502,in_0543,
in_0608`. This is the audited positive/negative tradeoff, not lossless dominance.
Complete gained/lost IDs for every stratum and both comparators are in summary.
All gains/losses occur in the original273-case ambiguity subset.

The new policy chooses +X610 times, -X25, +Z0, -Z18 over653 cases. Every action
has0.12m displacement and one measurement; no privileged actual action sweep is
performed. Bank forecast enumeration is public model computation, not additional
observed sensor data. Physical motion, settling, tracking and acquisition costs
are not measured. No event detection or warning-delay claim is possible here.

## Decision and verification

Local terminal `RETAIN_CONSUMED_PUBLIC_SELECTOR_COMPONENT`. The component passes
the frozen implementation gate: public-only choice, choice-before-future and
prediction-before-truth sealing, audited actual-view parity, more positive and
total answers than fixed, retained fixed-positive IDs, zero additional wrong
answers, and full mismatch/cost reporting. This does not reopen the old pilot,
promote an Android policy, repair prior coverage or establish independent benefit.

Five focused tests pass: positive mass differs from pair splitting; future/truth
APIs are unavailable during selection; deterministic ties/no-match UNKNOWN;
unsealed or tampered choices prevent true-source parsing; FN/UNKNOWN accounting.
A same-agent independent-equation saved-output audit passes37 hash links,
653 choice-cost reconstructions,653 actual selected-observation ray reconstructions,
653 public posteriors,653 fixed-query truth labels and15 complete metric groups.
It uses face intersections instead of the original slab algorithm and imports
neither predictor. All18 original artifact files retain their original hashes.
No scored rerun or new view was used for verification.

The root agent's independent read-only
[review](../../../../artifacts.local/work/ba-active-view-positive-20260921/root-readonly-review.json)
also passes:653 public-prior choices,653 saved posteriors, all three primary
273-case count tables and new seal hashes. It uses no new observations or model
inference and reports no material issue.

The full source-to-accounting stage log spans0.1715s on the host CPU, excluding
startup/final serialization. It is `TASK_NOT_GPU_SUITABLE` scalar arithmetic,
not measured device movement/acquisition latency. No persistent worker or paid
allocation remains. One new653-case replay ends here; no successor is launched.

Artifacts: `artifacts.local/work/ba-active-view-positive-20260921/run-v1/`
contains copied source/tests/protocol, consumed-input hashes, choices/observations/
predictions/evaluation/summary, stage order, local disposition and seals.
The parent folder retains `saved-output-audit.json`, focused test receipts and
supported registration/inheritance command receipts. Global metadata admission
remains incomplete: registration reports the existing ledger303 input-fingerprint
mismatch, and inheritance reports unknown terminal `ba-active-view-positive-20260921`.
The original command outputs, arguments and exit codes are retained without a
ledger rewrite or bypass. The sealed local disposition retains its pre-attempt
pending field; later receipts record these attempted assignments.

```powershell
python -B -m unittest discover -s scripts/research/nearfield_opportunities -p test_active_view_positive.py -v
python -B scripts/research/nearfield_opportunities/active_view_positive.py --source artifacts.local/work/ba-active-view-20260921/run-v1 --protocol research/active/dtr-r0/nearfield/ACTIVE_VIEW_POSITIVE_PROTOCOL_20260921.md --output artifacts.local/work/ba-active-view-positive-20260921/run-v1
```

The executed path is immutable; the runner refuses to overwrite it. A later
authorized reproduction requires a fresh child path and remains consumed-data
evidence. Known exact poses, a complete in-prior bank, uniform hypothesis counts,
noise-free radial quantization and simple geometry remain substantive assumptions.
