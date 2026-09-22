# Frozen model responds to lateral motion; wrong-way ranking is not the main failure

2026-09-21. One saved-output diagnostic completed on the consumed16-layout
transfer cohort. **BOUNDARY scores exceed matched OUTSIDE scores in173/173
positive-depth pairs, including119 pairs with unchanged geometric ToF zone sets.**
All17 previously added OUTSIDE false positives have lower scores than their
matched BOUNDARY frames, but still exceed the frozen high cutoff. These findings
do not support the proposed explanation that the model generally ignores motion
within a zone. They establish response direction, not correct absolute decisions.

[Protocol](LATERAL_PAIR_DIAGNOSTIC_PROTOCOL_20260921.md),
[reproducible diagnostic](audit_lateral_pair_response.py),
[disposition](LATERAL_PAIR_DIAGNOSTIC_DISPOSITION_20260921.json).
Evidence: `artifacts.local/work/ba-lateral-pair-diagnostic-20260921/`.

## Comparable pairs and fixed readout

384 complete triplets pair INSIDE/BOUNDARY/OUTSIDE at identical group/frame index.
173 have positive INSIDE and BOUNDARY truth and negative OUTSIDE truth;211 are
negative-depth controls. Source intent and actual rendered camera, backdrop,
target size/material/height/forward position were verified. The maximum actual
nonlateral numeric mismatch is1.11e-16m. Only target lateral position changes.
Equal axial depth does not mean equal radial range; shadow/occlusion changes and
conditional sensor noise remain consequences/confounds of the intervention.

The original B logits and high cutoff7.6612162590026855 are reused. No new model
execution, input manipulation, training, capture, cutoff search or held-policy
change occurs. Source, observation, prediction and evaluation seals pass; actual
geometry independently reproduces all1152 truth values. CPU is appropriate for
these small saved-array/geometry reductions.

## Paired scores versus fixed-cutoff decisions

Counts below refer to173 positive-depth pairs across16 groups. A decrease means
the positive frame scores higher than its matched OUTSIDE frame; it is not a
recall statistic. "Both low" includes real positives missed by B, whether or not
the geometry baseline A detects them.

| Contrast | Score decreases | Median logit decrease | Positive high / outside low | Both high | Both low |
| --- | ---: | ---: | ---: | ---: | ---: |
| INSIDE to OUTSIDE |172/173|22.021|138|13|22|
| BOUNDARY to OUTSIDE |173/173|17.476|110|13|50|

No primary pair crosses the threshold in the wrong direction (positive low,
OUTSIDE high). The one INSIDE ranking inversion is below the cutoff at both
endpoints. All16 groups have correct BOUNDARY-versus-OUTSIDE ordering throughout
their positive-depth intervals. Frames are clustered, with repeated dwell poses;
these are not173 independent layout confirmations.

Across all384 times, INSIDE scores decrease300 times and BOUNDARY323 times.
Negative-depth controls are not positive
versus negative discrimination tests and are not silently removed from the run.

## Same-zone coverage does not explain the errors

The full cuboid geometric footprint uses ray/volume intersection on the original
point-sampled lattice, then aggregates occupied rays into the existing64 zones.
It is not a target-center test or a claim of visible/native-return ownership.

| BOUNDARY to OUTSIDE geometric zone sets | Pairs | Score decreases | Fixed pair separated | Both high (outside FP) | Both low |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identical nonempty sets |119|119|88|3|28|
| Changed sets |54|54|22|10|22|

Each stratum spans16 groups. The smaller changed-set stratum contains10 of the13
positive-depth OUTSIDE false positives. This is opposite to concentration in the
unchanged-set stratum; it does not prove that crossing cells causes errors.
The supplemental front-face inner-edge column is unchanged in141 pairs, all
with decreasing scores (102 separated,8 both high,31 both low);32 cross columns,
also all decreasing. All173 INSIDE-to-OUTSIDE pairs change footprint and column,
so that contrast cannot test same-zone behavior.

Even identical geometric zone sets do not imply identical observations. The
median common-valid range MAE is0.000265m for unchanged sets and0.099043m for
changed sets; validity and per-zone values are retained in every pair record.
RGB has nine ordered feature samples per zone. Neither input equality, RGB-only
causality, discarded features nor physical nonidentifiability follows from these
coarse geometric strata.

An independent supplemental equality check finds0/173 primary BOUNDARY/OUTSIDE
pairs with identical full64-zone ToF arrays (NaNs compared as missing). The same
is true for INSIDE/OUTSIDE. Consequently primary score changes cannot be
specifically attributed to RGB by holding ToF input constant. This check did
not alter the protocol, pairs, model or primary results.

## What the17 known OUTSIDE mistakes actually look like

All17 score lower than their paired BOUNDARY frames. Thirteen occur at positive
depth and four while even the INSIDE/BOUNDARY variants are outside the depth
window. They remain concentrated in the same three layouts. The positive-depth
subset is shown below, preserving each group's full denominator:

| Layout | Pairs | Decreases | Both high | Both low | Separated |
| --- | ---: | ---: | ---: | ---: | ---: |
| BODY protruding plane g01 |12|12|4|0|8|
| BODY suspended solid g02 |10|10|3|0|7|
| HEAD hanging plane g02 |12|12|6|0|6|

For example, BODY suspended solid f1038/f1062 scores11.755/10.950: moving outside
lowers the score by0.805 but both exceed7.661. Conversely, the already missed
HEAD hanging plane g01 has9/9 correctly ordered pairs yet9/9 below cutoff at
both endpoints. Thus within-pair ranking is insufficient for cross-layout
decisions. This is a descriptive margin/score-level problem; the causal origin
and whether a different representation or training recipe fixes it are unknown.
The two examples do not authorize another cutoff or a calibration sweep.

The118 recovered Boundary frames remain intact:91 are in unchanged-footprint
pairs (73 with returned native corridor contributors),27 in changed-footprint
pairs (15 supported). Discarding all unchanged-footprint situations would discard
useful predictions as well. Native contributor truth is evaluation-only.

## Original task costs remain unchanged

TP/FP/FN, with all original denominators and UNKNOWN retained:

| Stratum | A current | C current | A hold | C hold |
| --- | --- | --- | --- | --- |
| Core,173 positive/595 negative |171/17/2|173/36/0|173/33/0|173/58/0|
| Boundary,173 positive/211 negative |10/1/163|128/10/45|14/5/159|140/19/33|

Core current precision90.96% to82.78%, FPR2.86% to6.05%; Boundary precision90.91%
to92.75%, FPR0.47% to4.74%. Original event readout remains Core16/16 and Boundary
4/16 to15/16, with the same five delayed Boundary detections and one entirely
missed event. UNKNOWN remains656/768 Core and384/384 Boundary. No new masks or
localization predictions exist, so no IoU gain or first-alert improvement is claimed.

## Delivery and decision

Retain the matched score table as COMPONENT diagnostic evidence. The general
"same-zone motion is invisible to this model" explanation is unsupported on
these pairs. Do not initiate a finer-grid architecture, occupancy-bound system,
new ranking loss, or threshold adjustment based on that explanation. Any separately
authorized successor must address absolute positive/negative separation across
layouts while preserving the demonstrated Boundary rescues; this audit does not
select a validated repair.

Three focused geometry tests pass, covering side-face entry, zero ray components,
mirroring and behind-camera geometry. Both generated figures were visually
inspected;12 selected RGB hashes match their original receipts. The image grid
uses the protocol's earliest-error rule, so two failed-group examples are
negative-depth frames, explicitly labelled truth=0. Images and all16 score curves
are retained beside the pair table; no manual best-case selection occurred.

Independent review PASS reconstructs all1152 geometric zone footprints and
recounts every paired row/cutoff decision, with source/code/protocol/output hashes
unchanged. The full receipt is `independent-review.json` beside the results.

The existing global ledger line303 fingerprint mismatch blocks supported run
registration; the supported inheritance command reports unknown terminal id.
Local structured disposition and command receipts preserve that metadata gap.
No old records were edited or bypassed. Original reserved test remains untouched;
no model/simulator process, GPU allocation or temporary payload requires release.
