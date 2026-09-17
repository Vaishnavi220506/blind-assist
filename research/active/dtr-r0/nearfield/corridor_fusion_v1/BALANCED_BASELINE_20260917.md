# A: frozen balanced research baseline

2026-09-17. The user adopts E1 arm A as the balanced research baseline for the
next bounded conditional-veto diagnostic. This updates its research role from
the alternative working point recorded in [E1 results](E1_RESULTS_20260917.md).
It does not change the App, deployed sensor logic, or historical conclusions.

## Exact inherited implementation

- Model: `artifacts.local/work/corridor-depth-e1-20260917/run-v2/A-model.pkl`.
- SHA-256: `d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef`.
  The current file was checked against that run's `selection-seal.json`.
- Input: unchanged 2485 public sensor/geometry features, in the original order.
- Decision: `predict_proba(features)[:, 1] >= 0.3917890013717321`.
  This is a frame decision; it does not apply MZ145's two-threshold causal rule.
  A nonalert means UNKNOWN, not certified clear space.
- HGB: 150 iterations, 7 maximum leaves, learning rate .05, minimum leaf 8,
  L2=1, no early stopping, random state177017. The first of three prespecified
  candidates won dev F1, with AP and candidate index as tie-breakers.
- Training: original MZ136 TRAIN192, scenes0–3; selection: dev48, scene4.
  Dev F1 and dev-recall>=95% selected the same threshold:24TP/1FP/0FN.
  Original test48 was excluded. Models and thresholds were sealed before this
  run's MZ170 report-label access.

The entire MZ170288 A score vector is bit-identical to the original static
MZ145 HGB score vector, as verified in `run-v2/delivery-audit.json`. A's change
is the selected threshold/readout and operating tradeoff. It is not new learned
representation capability. Ranking PR-AUC=.9614845378 and AP=.9614903270 are
therefore unchanged from that static reference.

## Same-cohort evidence and tradeoffs

All results below use the same consumed MZ170288 frames,144positive/144negative,
from24 configurations of the same procedural simulator. They are Development
evidence, not fresh confirmation, natural-scene, phone, wearer-benefit or safety
evidence. Reference models have different historical calibration and causal
readouts; this table compares their complete frozen operating points.

| Frozen operating point | TP/FP/FN | Precision | Recall | F1 | False segments | False bins, seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MZ129 |136/74/8|64.76%|94.44%|76.84%|28|18.50|
| MZ145 causal |143/69/1|67.45%|99.31%|80.34%|24|17.25|
| Four-expert causal mean |140/36/4|79.55%|97.22%|87.50%|19|9.00|
| **A balanced research baseline** |**137/28/7**|**83.03%**|**95.14%**|**88.67%**|**18**|**7.00**|

A has116TN and123UNKNOWN across the complete288-frame denominator. Compared
with mean it trades8fewer FP for3more FN; versus MZ145 it trades41fewer FP for
6more FN. Relative to MZ129 the aggregate changes are46fewer FP and one fewer
FN, but this does not establish preservation of each old true alert. The mean
retains the stronger ranking curve, AP=.9795844730. MZ129 has no comparable
continuous ranking score.

| Family | A TP/FP/FN | Original mean TP/FP/FN |
| --- | ---: | ---: |
| Near rod / far wall |36/11/0|36/13/0|
| Boundary pressure |30/6/6|32/16/4|
| Substantial BODY |36/1/0|36/1/0|
| Suspended HEAD |35/10/1|36/6/0|

All30 positive events are eventually detected by each reference. A has four
0.25s onset delays: mean delay0.03333s, maximum0.25s. The original mean has
maximum0.50s but smaller mean delay0.025s; MZ145's mean is0.00833s and MZ129's
0.01667s. No claim of uniformly earlier or retained warning times follows.
Among six observable within-episode true-to-false transitions, A releases
immediately on five and after0.25s on one; other episode endings do not provide
observed clearance transitions. False duration sums sampled0.25s bins, not
real-user interruption duration.

The native ToF audit finds five A nonalerts carrying57 returned corridor
contributor samples. They include four boundary frames and one HEAD frame,
all previously alerted by MZ129. Input preservation does not preserve every
native-supported warning. Per-return native Radar lineage is NOT_EVALUABLE.

Measured warm A decode+base frontend+head latency on24 predefined first-episode
frames is p50=78.20ms/p95=88.94ms in E1. Its later intermediate diagnostic
replay measured81.57/103.86ms on the same selection, illustrating run variation.
These are measured host timings without capture, transport or event waiting;
they are not phone latency. No new inference was run to adopt this baseline.

## Conditional specialist diagnostic

The user's family-oracle example takes the completed intermediate C on the
near-rod/far-wall family only, and A on every other family. C contributes
36TP/0FP/0FN there instead of A's36/11/0. The resulting arithmetic is
**137TP/17FP/7FN**,127TN: precision=`137/154`=**88.96%**, recall=`137/144`=
**95.14%**, F1=`274/298`=**91.95%**.

This is a conditional diagnostic upper bound for the proposed rod-only
specialist routing opportunity under perfect evaluator-family knowledge.
It is not a deployed router, an observable family classifier, a global optimum,
or a proven upper bound on every possible policy. An imperfect observable
router may lose true alerts or invoke C outside the intended stratum. Its
aggregate count does not establish event timing, native-support retention or
runtime. The separately authorized tiny conditional C veto must be evaluated
as its own bounded diagnostic against this exact A baseline; family truth must
not enter its inference. No model, threshold or App change is implied by the
oracle calculation.

## Evidence pointers

- E1: `artifacts.local/work/corridor-depth-e1-20260917/run-v2/`
  `selection-seal.json`, `summary.json`, `native.json`, `delivery-audit.json`,
  and `predictions.csv`.
- Cached reference parity:
  `artifacts.local/work/corridor-depth-e1-20260917/e0-reference-replay.json`.
- Intermediate C: `artifacts.local/work/corridor-depth-intermediate-20260917/`
  `summary.json` and `selection-seal.json`; see
  [intermediate results](INTERMEDIATE_RESULTS_20260917.md).
