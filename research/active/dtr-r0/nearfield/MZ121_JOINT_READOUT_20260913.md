# MZ121: development joint feasibility exists; spatial tradeoff remains

**The original MZ120 development curve has31jointly feasible grid points,
0.30through0.60at0.01spacing.** Threshold0.71fails the rod-cell criterion,
but that failure does not establish that this model has no feasible development
operating point. Keep MZ116 as baseline and preserve MZ120's original terminal,
model and threshold unchanged.

This is the user's requested zero-training check, using only consumed
development data to judge feasibility. It does not test a new architecture,
retune on MZ119, reopen its evaluation, or establish a new-scene result.

## Checked criteria and selection correction

[Joint selector and executable diagnostic](mz121_joint_readout.py),
[focused tests](test_mz121_joint_readout.py).

Apply all original MZ120 retention requirements **before** selecting a point:
frame recall within2percentage points of MZ116; at least30% reduction in both
FP frames and false seconds; all previously detected complete events retained;
maximum first-alert delay at most baseline+.25s; HEAD and rod occupied-cell
recall each at least90%. Minimize FP within that joint set, breaking ties with
the higher threshold. Missing critical-class opportunities are not perfect
recall, and an empty feasible set returns no selected point.

The new independent Development readout candidate is **0.60**. No MZ119 curve
or source is opened by this diagnostic. It only extracts the old development
baseline from MZ120's authenticated summary, which also contains already-known
historical transfer results. No transfer outcome enters the selection.

| Development metric | MZ116 | Original MZ120 .71 | Joint readout .60 |
|---|---:|---:|---:|
| TP / FP / FN / TN |82/40/2/20|83/14/1/46|84/16/0/44|
| Frame recall |97.62%|98.81%|100%|
| False-alert seconds |10.0|3.5|4.0|
| Complete events detected |8/8|8/8|8/8|
| Maximum first-alert delay |.25s|0s|0s|
| HEAD occupied cells |not comparable|133/142|142/142|
| Rod occupied cells |not comparable|102/126|114/126|
| Whole-grid cell TP / FP / FN |not comparable|700/899/262|764/1130/198|
| Whole-grid precision |not comparable|43.78%|40.34%|
| Whole-grid recall |not comparable|72.77%|79.42%|
| Whole-grid IoU |not comparable|.3761|.3652|

The .60 readout retains60% fewer false frames/seconds than MZ116 and meets the
old development gates. Relative to.71, it restores12rod cells and9HEAD cells
while adding231wrong grid cells (+25.70%). There are no baseline occupancy
probabilities, so MZ116 does not receive invented cell-level scores.

The lower feasible boundary is the nuisance requirement: at.29,29FP exceed
the allowed28; at.30there are26FP. The upper boundary is rod recall:
.60gives114/126(90.48%);.61gives112/126(88.89%). This establishes the feasible
**sampled grid**; it does not infer every unsampled real-valued threshold.

As an additional disclosed diagnostic, imposing no increase over.71's899wrong
grid cells leaves **zero** jointly feasible grid points. That was not an
original acceptance requirement and is not retroactively added as one. It
shows why a readout correction is not evidence of better spatial precision.
The full45-cell score includes off-corridor, above-head and far cells; all
original region-specific metrics remain available in `joint_curve.json`.

## Training data, inspected after development selection

These numbers diagnose all336training frames at development-selected thresholds;
training metrics do not choose the threshold.

| Training metric | .71 | .60 |
|---|---:|---:|
| Alert TP / FP / FN |176/23/29|197/64/8|
| Frame recall |85.85%|96.10%|
| Missed complete events |2|0|
| HEAD occupied cells |247/310|274/310|
| Rod occupied cells |219/250|231/250|
| Wrong grid cells |2303|2949|
| Whole-grid precision |41.05%|37.51%|
| Whole-grid IoU |.3705|.3558|

Thus the original training recall deficit is partly threshold-dependent, but
the tradeoff remains: substantially more false alerts/cells accompany recovery.
This does not isolate capacity, optimization, local sampling or measurement
ambiguity, and does not prove adequate spatial fitting.

## Interpretation and next decision

Retain the joint selector and.60profile as a **Development component/challenger**,
not as a replacement for MZ116 or a rewrite of MZ120's negative terminal.
MZ120 has initial alert-discrimination benefit; its critical spatial and HEAD
continuity findings remain scoped as reported. No new transfer score at.60is
claimed. This bounded branch stops after the now-positive feasibility diagnosis
and selector correction; no training, data expansion, CNH or suppressor follows.

The code-backed early-pooling hypothesis remains eligible but untested here.
`sampled.mean(-1)` and `sampled.amax(-1)` discard the ordering of49sampled
feature vectors before ToF/Radar fusion. CNN features and the query's projected
sampling footprint still carry contextual information, so this is not proof
that all position information is absent. A future comparison should preserve
the current encoder, inputs, split and loss, fuse position-indexed local samples
with spatially related ToF returns before pooling, and compare **both** arms
under this same joint readout and spatial precision/recall accounting. Enlarging
data or adding CNH is not established as necessary by this result.

## Validation and retained evidence

Authenticated MZ120 model/summary/split hashes and both source receipts. Rebuilt
all480×45native-bound labels and all101development operating points exactly;
baseline prediction payload hashes match the original freeze. Read-only source
audit uses raw/evaluator/spec files; spec authenticity is checked against the
receipt's top-level `spec_sha256`, after correcting an initial lookup that
incorrectly expected it in the per-payload hash map. No scoring output existed
at that initial mechanical failure.

Four focused tests pass: reject a lower-FP point that fails spatial retention,
return no fallback when infeasible, reject missing HEAD opportunities, and
include event/delay constraints before selection. Scalar cached scoring runs on
CPU (`TASK_NOT_GPU_SUITABLE`); no model inference, GPU allocation, capture,
service or paid resource is started.

Durable evidence: `artifacts.local/work/mz121-joint-readout-20260913/analysis-v1/`
contains the complete annotated curve, validation, summary, source/output hashes
and a model-hash-bound independent `readout-candidate.json`. Original MZ120
payloads remain unchanged. Structured registration/inheritance is pending the
pre-existing ledger303input-fingerprint mismatch; successful knowledge-engine
acceptance is not claimed. The attempted command/error is retained separately.
