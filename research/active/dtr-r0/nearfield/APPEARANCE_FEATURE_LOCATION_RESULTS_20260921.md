# Spatial input mediation: apparent stability can include compensating effects

2026-09-21. One newly authorized diagnostic reuses the consumed appearance cache
and frozen B/N heads. **N's smaller whole-input flip count is not sufficient
evidence of learned appearance invariance.** Complementary spatial features can
restore some decisions changed by central features while creating other changes.
Retain this input-location diagnostic; no spatial suppression or alert policy
is selected.

[Protocol](APPEARANCE_FEATURE_LOCATION_PROTOCOL_20260921.md),
[implementation](appearance_feature_location.py),
[three focused tests](test_appearance_feature_location.py).
The [original source](APPEARANCE_RESULTS_20260921.md) remains NOT_EVALUABLE for
strict pure-material causality. This diagnostic performs no recapture, image
editing, feature re-encoding, model fitting or cutoff change.

## Fixed intervention and complete results

The public 8x8 model input has216 RGB-derived and4 sensor channels. Central
columns2:6 and complementary outer columns0:2,6:8 were fixed before execution.
For every288-pose repeat/target/background comparison, retain reference sensor
channels and insert either central or outer RGB feature entries from the changed
condition. Full reference/changed input replays exactly: B/N maximum logit error
is0.0 and every threshold decision matches the original1152 predictions.

These are **feature-grid locations**, not cropped pixels, foreground masks,
object ownership or exact physical regions. Encoder receptive fields overlap;
hybrids can be outside the natural feature distribution. No online device is
assumed to possess a same-scene reference appearance.

| Model / condition, each288 pairs | Full current flips | Central-only | Outer-only | Mean absolute allocated logit: center / outer / interaction |
| --- | ---: | ---: | ---: | --- |
| B / repeat |1|1|0|0.351 /0.144 /0.007|
| B / target |13|13|0|3.210 /0.673 /0.161|
| B / background |20|22|4|4.371 /1.738 /0.743|
| N / repeat |0|0|0|0.445 /0.244 /0.008|
| N / target |8|19|1|3.545 /1.639 /0.391|
| N / background |12|19|2|3.609 /5.362 /0.641|

All counts compare original A OR frozen supplement against its reference output;
they are changes, not improvements. The exact two-part allocation shares the
nonlinear interaction symmetrically and sums to total logit change per pair.
Absolute allocations need not sum to the absolute total because signed effects
can oppose. No independence, statistical significance or calibrated causal
effect is inferred from these correlated frames.

N target central-only changes19 IDs; full changes8. Six IDs overlap: adding the
outer features restores13 of the19 reference decisions and changes2 other IDs.
N background central-only changes19 versus full12, with10 shared IDs: nine are
restored and two newly changed. Center/outer signed allocations oppose on168/288
target and187/288 background pairs. In comparison B target has12 shared changes,
one central-only and one full-only; B background has13 shared, nine central-only
and seven full-only. Equal totals can hide different changed IDs.

Family breakdown also rejects a universal location story. For N background,
mean absolute center/outer allocations are3.274/7.188 for BODY plane,
4.133/8.717 for BODY solid,2.915/3.023 for HEAD hanging plane, and4.116/2.523
for HEAD horizontal. The corresponding central-only/full current flips are
0/0,8/9,0/1 and11/2. Central-only is therefore not a generally less sensitive
readout. B is more center-dominated on mean absolute allocation across these
four families, but nonlinear interactions still change decision membership.

## What this changes

The useful hypothesis is more specific than "remove background" or "penalize
all appearance drift": determine which interactions preserve intrusion evidence
across nuisance changes and which merely compensate on the consumed source.
That is a proposed research question, not a trained disentanglement method.
This result does not prove where relevant physical pixels originate, that outer
context caused historical errors, or that suppressing it would improve alerts.

Earlier [HEAD paired consistency](HEAD_PAIRED_RESULTS_20260908.md) already found
lighting-related damage without an original-EVAL improvement from its fixed
consistency recipe. Earlier [background invariance](corridor_fusion_v1/BG_INVARIANCE_RESULTS_20260918.md)
reduced logit drift while preserving identical held alert outputs. Those closed
recipes are not retried here. Their source-control work also documents disabling
background mesh world-position offset; this is a concrete source-repair lead
for future authorization, not proof that it explains this run's51 changed pixels.

All23 failed native comparisons and all288 poses per condition remain included;
three tiny original camera-coordinate discrepancies are preserved. No selected
native-equal subset is upgraded to confirmation. The new terminal is
`INPUT_LOCATION_DIAGNOSTIC_ONLY`, COMPONENT mode; original NOT_EVALUABLE and all
frozen baseline/negative-control dispositions remain unchanged.

## Validation and evidence

Three tests cover disjoint/exhaustive feature partition, sensor/input preservation,
nonlinear allocation and identity. Original code/input/checkpoint/feature hashes
and all original seals authenticate before inference. New1728 pair/model records
are sealed before evaluator family tags are joined. A separate standard-library
saved-output calculation checks allocation arithmetic, six aggregate summaries,
flip counts and seal order without invoking a model. No claim of whole-system
alert gain follows from these feature interventions.
An independent agent additionally checks1728 paired records, original logit
parity, all flags/cutoffs, six overall and24 family summaries, complementary
RGB-only slices and the stated restoration/new-flip ID counts; no defect found.

Actual CUDA on the same RTX3060 Laptop/torch2.9.1+cu128 is verified. Existing
measured batch64 backend choices are reused for the same heads/runtime. The
diagnostic takes4.651s including source/cache checks; child lifetime5.493s.
The child exits0 under a120s subprocess timeout; no UE, background worker,
persistent allocation or paid service is started. Previous raw evidence is
unchanged.

Local artifacts: `artifacts.local/work/ba-appearance-feature-location-20260921/`;
`remote-results/run-v1/` holds protocol, predictions, summary and seals;
`saved-arithmetic-audit.json` records the independent arithmetic check.
Worker retains the small task-owned snapshot/results under
`G:/DevWorkspace/BlindAssist/artifacts/work/ba-appearance-feature-location-20260921/`.
Original feature/RGB/native evidence remains in the original appearance task.
These are durable diagnostic evidence; release storage only via separately
authorized cleanup of the exact task trees, never shared assets/environments.

Supported registration/inheritance receipts remain local when the existing
ledger303 / unknown-terminal blocker prevents global metadata admission. No
manual ledger edit or automatic training successor follows this diagnostic.
