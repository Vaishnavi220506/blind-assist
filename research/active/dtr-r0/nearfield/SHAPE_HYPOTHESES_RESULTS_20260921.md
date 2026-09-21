# Multiple shape hypotheses retain ambiguity; incomplete priors still give wrong OUT

2026-09-21, one fixed analytic synthetic EXPLORE run. **Retain a narrow
set-valued ambiguity component, not an alert replacement or clearance method.**
Within the 150-shape closed world, consensus changes three erroneous single-fit
OUT predictions into UNKNOWN. It does not add any true alert. On 24 out-of-prior
cases, four true intrusions receive unanimous but wrong OUT predictions. This
directly falsifies an unqualified claim that agreement among plausible shapes
certifies absence of an obstacle.

[Pre-execution protocol](SHAPE_HYPOTHESES_PROTOCOL_20260921.md),
[runner](../../../../scripts/research/nearfield_opportunities/shape_hypotheses.py),
[focused tests](../../../../scripts/research/nearfield_opportunities/test_shape_hypotheses.py).
No training, threshold sweep, old-data re-evaluation, UE/CUDA launch, or second
scientific run occurred. Existing Calibration/A/B/N and frozen results are unchanged.

## Actual comparison and complete counts

Inputs are an **ideal 64x48 binary silhouette** and hypothetical 8x8 regional
radial means, each from 3x3 rays. The silhouette is supplied by this analytic
source, not obtained by a demonstrated RGB segmentation algorithm. True shapes
are not predictor inputs. The explicit prior contains 150 axis-aligned boxes;
the closed-world cohort includes those same geometries once each. That tests
prior-inclusion feasibility, not learned or heldout generalization.

All 174 static cases are retained: 67 positives and 107 negatives. These are
designed correlated cases, not independent natural trials or temporal events.
`OUT` below always means **OUT_MODEL_PRIOR**, never sensor-certified free space.
`FN alert` includes every positive without an IN output, including UNKNOWN.
The six disjoint categories TP, FP, false OUT, correct OUT, UNKNOWN-positive and
UNKNOWN-negative sum to N; FN alert is derived and should not be added again.

| All 174 cases | TP | FP | FN alert | False OUT | Correct OUT | UNKNOWN + / - | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single best consistent shape |40|0|27|7|61|20 / 46|62.07%|
| Consistent-shape consensus |40|0|27|4|58|23 / 49|58.62%|
| Public zone-centre point proxy |42|0|25|0|0|25 / 107|24.14%|

All three have 100% alert precision and 0/107 FPR on this selected source;
recall is 59.70%, 59.70%, and 62.69%, respectively. These high precision values
do not establish safe or complete detection. The point proxy never outputs OUT,
so its classification coverage and wrong-OUT count are not evidence that it
solves negative classification. It projects a zonal aggregate along the public
zone-centre ray; it does not receive private returned-point coordinates.

| Consensus stratum | N (+ / -) | TP / FP / FN alert | False / correct OUT | UNKNOWN + / - | Coverage |
|---|---:|---:|---:|---:|---:|
| Closed world |150 (49 / 101)|40 / 0 / 9|0 / 58|9 / 43|65.33%|
| Closed, near .30m lower window |30 (7 / 23)|2 / 0 / 5|0 / 0|5 / 23|6.67%|
| Closed, other distances |120 (42 / 78)|38 / 0 / 4|0 / 58|4 / 20|80.00%|
| Out-of-prior thin appendage |6 (6 / 0)|0 / 0 / 6|4 / 0|2 / 0|66.67%|
| Out-of-prior visible appendage |6 (6 / 0)|0 / 0 / 6|0 / 0|6 / 0|0%|
| Out-of-prior off-grid box |12 (6 / 6)|0 / 0 / 6|0 / 0|6 / 6|0%|

The two closed-world subrows partition the 150-row group and must not be counted
again. FPR is undefined in appendage-only positive strata. Complete per-policy
subgroup metrics are retained in `results.json`.

## What the multiple explanations change

Six range-supported cases have feasible shapes on both sides of the corridor
decision. There are two exact observation-equivalence classes:
`case_060/061` and `case_070/071/080/081`. Each has both positive and negative
truth. Their front surfaces are at Z=.25m; a thin box ends before the .30m
evaluation window while a thick box continues into it behind the same visible
front. Both the silhouette and regional ranges are exactly identical, not merely
within the 1cm consistency tolerance.

The fixed single-best tie order chooses the thin explanation. It therefore
produces three wrong OUT and three correct OUT. Consensus abstains on all six.
Closed-world wrong commitments fall 3 to 0, with coverage 69.33% to 65.33%.
TP remains40, FP0, and FN alert9 for both. This is a useful representation of
ambiguity, **not improved alert recall**. All of this particular disagreement
benefit occurs near the .30m lower-distance boundary; other closed-world cases
have identical single/set decisions. There is no measured broad ambiguity gain.

Across all 174 predictions there are 40 unanimous IN, 62 unanimous OUT, six
disagreements, 14 no-consistent-prior outputs and 52 no-observed-range outputs.
The latter two causes produce UNKNOWN. A predicted bank shape cannot supply
metric range when no observed return exists. Feasible-set sizes are 0/1/2/4/38
on 14/72/40/4/44 cases; a large set can still agree or have no valid range.

Against the point proxy, explicit shape extent adds six closed-world true alerts
without a false alert (40 versus34 TP). This benefit belongs to the shape prior,
not specifically to keeping multiple hypotheses: single-best also gets those six.
Conversely, requiring a compatible prior loses eight point-proxy true alerts on
out-of-prior cases: two visible-appendage and six off-grid positives. Overall the
shape candidate has two fewer TP. Raw observations are retained, but the tested
shape-only policy does not preserve every warning from the point comparator.

## The prior-completeness falsifier succeeds

`case_152/154/158/160` contain a 4mm-high inward appendage on an otherwise outside
box at front Z1.5m or2.5m. The appendage crosses X=.30m but falls between this
source's camera/ToF ray samples. Each complete observation is identical to an
outside bank shape (`case_024/026/144/146`, respectively). All matching bank
shapes say OUT, yet full-solid evaluator truth says IN. Neither consensus nor
single-best notices the absent shape family.

These four cases are an intentionally declared out-of-model thin-structure
challenge, not measured frequencies of realistic obstacles. Their role is to
falsify a universal inference guarantee. Higher-resolution images, different
sampling, or a broader prior could change them, but none was tried after results.
The 50mm-high appendages instead lead to UNKNOWN here, exposing mismatch without
solving it. The 12 off-grid shapes all abstain; generalization coverage is absent.

Seven complete observation-equivalence classes contain opposite truth: two
range-supported closed-world classes above, four appendage/bank collisions,
and one large blank/no-return class. The last is handled by the no-range guard
and is not credited as set-valued metric-shape identification.

## Decision, validation, and reproduction

Local terminal: `COMPONENT_SYNTHETIC_AMBIGUITY_RETAINED_PRIOR_COVERAGE_REQUIRED`.
Retain only the finite-set/UNKNOWN mechanism as a synthetic diagnostic component.
The complete shape-only alert policy has no overall advantage over the point
proxy; an unconditional consensus-to-clearance claim is **FALSIFIED**. No fitted
model, runtime candidate, new threshold, or follow-on experiment is selected.

Five focused tests pass: analytic radial raycasting/parallel axes, complete-solid
contact, exact hidden-depth ambiguity, observation-field isolation, and UNKNOWN
confusion accounting. A separate saved-output scalar audit reconstructs truth
intersection and all three complete confusion tables without importing the
predictor or rerunning inference; source/protocol/input/prediction hashes match.
Predictions were serialized and sealed before evaluator geometry was reloaded.
An independent first-level temporal agent also completed a read-only source and
saved-output cross-review: all174 cases, seven opposite-truth observation classes,
the four wrong-OUT IDs, every complete policy count and the prediction boundary
match, with no material findings. Its separate receipt is retained at
`artifacts.local/work/ba-active-view-20260921/shape-readonly-review.json`.

Actual runtime is Python3.11/NumPy2.4.4 on CPU with
`TASK_NOT_GPU_SUITABLE`: total0.4005s, prediction-only0.0668s for174cases and a
150-shape bank. This excludes any real segmentation/sensor acquisition and is
not target-device latency. No persistent task process or allocation remains.

Evidence root: `artifacts.local/work/ba-shape-hypotheses-20260921/run-v1/`.
It retains the frozen source/protocol/config, prior bank, source truth, complete
observations, sealed predictions, results, local disposition, completion and
saved-output audits. Supported registration was attempted and fails at the
pre-existing `experiments/index.jsonl:303 input_fingerprint` mismatch; supported
inheritance assignment reports an unknown terminal. Both command receipts are
retained in `registration.log` and `inheritance.log`. The local disposition does
not claim global acceptance, and no ledger edit or bypass occurred.

Executed command (the runner refuses to overwrite an existing directory):

```powershell
& E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe scripts/research/nearfield_opportunities/shape_hypotheses.py --output artifacts.local/work/ba-shape-hypotheses-20260921/run-v1
```

Reproduce only under an explicitly authorized new child output directory, with
the frozen source/config. Focused checks:

```powershell
& E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe -B -m unittest discover -s scripts/research/nearfield_opportunities -p test_shape_hypotheses.py -v
```

Protocol SHA256: `ba26068f2153c8d2623071e2558c6f18ad5298b0dc9ce00e831cf00117eae524`.
Prediction SHA256: `9adecc2cf73f8f0fa6c784514567dea90a51ebad563611adf604ff080e56f2da`.
Perfect segmentation, closed-world inclusion, noiseless calibration, fixed ray
sampling and axis-aligned boxes are substantive assumptions. This experiment
supplies no natural-distribution, hardware, safety or current Calibration-A gain.
