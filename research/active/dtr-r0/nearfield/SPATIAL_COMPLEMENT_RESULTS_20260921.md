# Frozen A/B complementarity: useful current rescue, remaining FP and hold cost

2026-09-21. One separately authorized saved-output diagnostic is complete.
**A plus high-scoring B recovers useful missed obstacles on consumed development,
but no tested condition establishes a complete zero-added-FP alert upgrade.**
Preserve this as a diagnostic COMPONENT, not a deployed policy. The earlier
spatial BCE full-replacement recipe remains NEGATIVE_CONTROL in its original scope.

[Protocol](SPATIAL_COMPLEMENT_PROTOCOL_20260921.md),
[runner](spatial_complement_diagnostic.py),
[independent audit](audit_spatial_complement.py).
No model fit, image inference, capture, original-weight change, test evaluation,
runtime change or automatic successor occurred. All results below reuse the SAME
eight consumed development groups/576 frames: Core384=86P/298N, Boundary192=86P/106N.

## What can separate useful supplementation from extra false alerts?

The original A-silent/B>=0 pool contains95 frames:77 true rescues and18 new FP.
Unrestricted OR is costly: current Core86/19/0 and Boundary81/6/5, versus A85/4/1
and5/3/81. Keeping A guarantees retention; it does not itself control B's added FP.

We fixed two observable screens before this diagnostic: B score alone, and B
score plus the original Calibration current decision (T0=.007085703945147101).
The second screen is completely redundant here: **all95 B>=0 disagreement frames
already satisfy Calibration.** Both fixed-high and grouped versions therefore
have identical flags, recovered IDs, FP and event metrics with or without it.
This particular geometric screen supplies no additional discrimination.

## Current-frame complementarity, with selection provenance retained

Counts are TP/FP/FN. The high cutoff7.6612162590026855 was already disclosed in
the prior consumed-dev diagnosis. It was NOT selected on new independent data.

| Current readout | Core | Boundary | Core precision / FPR | Boundary precision / FPR |
| --- | --- | --- | --- | --- |
| Frozen A |85/4/1|5/3/81|95.51% /1.34%|62.50% /2.83%|
| A OR fixed-high B |86/4/0|71/3/15|95.56% /1.34%|95.95% /2.83%|
| A OR grouped-cutoff B |86/5/0|71/3/15|94.51% /1.68%|95.95% /2.83%|

Fixed-high supplementation adds1 CoreTP and66 BoundaryTP without new current FP.
Boundary recall rises5.81% to82.56%; events2/8 to7/8. All A flags and first in-event
times remain; the one formerly delayed Core HEAD onset improves0.2s to0s, and
all8 Core events attain full positive-frame coverage. One Boundary event still
has zero coverage; another first reports0.2s late. This is not complete Boundary
detection, and recovered frames are repeated samples within only eight groups.

The two screenings each receive the same leave-one-base-group-out diagnostic.
For each held group, its entire72 frames and all lateral variants are excluded.
Using only the other seven groups' eligible A-negative negative frames, the
cutoff is immediately above the highest negative B logit, with fixed lower
bound0. Scores/cutoffs/comparisons all use float64, including nextafter ties.
Every calibration partition has zero added RAW FP by construction.

Seven held groups use7.633724212646485; holding out BODY protruding-plane g02
uses6.338444232940675. In that group f1588, an INSIDE pre-entry frame at0.8s,
has logit7.633724212646484 and becomes an additional FP. It is the highest-scoring
negative available when calibrating on all eight groups, so withholding its
group exposes the optimistic aspect of the full-dev zero-FP cutoff. It also
passes Calibration. The other seven held groups add no current FP.

The grouped result preserves all67 fixed-high true rescues but adds this one FP.
The method/screen design was already informed by consumed outcomes; this is an
internal stability diagnostic, NOT an unbiased out-of-fold generalization estimate
or fresh confirmation. Eight differing fold cutoffs are not one deployed policy.

| Held base group | New Boundary current TP | New Core current TP | New current FP |
| --- | ---: | ---: | ---: |
| BODY protruding-plane g00 |8|0|0|
| BODY protruding-plane g02 |12|0|1|
| BODY suspended-solid g01 |7|0|0|
| BODY suspended-solid g05 |10|0|0|
| HEAD hanging-plane g00 |10|0|0|
| HEAD hanging-plane g06 |8|0|0|
| HEAD horizontal g05 |0|1|0|
| HEAD horizontal g06 |11|0|0|

Thus Boundary gains occur in7/8 groups rather than one favourable layout, but
the strict zero-additional-FP condition fails the grouped check. No different
cutoff or screen was chosen to remove f1588 after seeing it.

## Unchanged hold adds a separate release cost

| Same nonrecursive one-frame hold | Core | Boundary | Core FP segments | Boundary FP segments |
| --- | --- | --- | ---: | ---: |
| A |85/12/1|7/5/79|10|3|
| A OR fixed-high B |86/12/0|75/8/11|10|6|
| A OR grouped-cutoff B |86/13/0|75/8/11|11|6|

The fixed-high current output has no new FP, but the original hold extends three
genuine final-positive B alerts into the first exit-negative sample:

| Frame | Layout | Time | B logit on negative | Calibration on negative |
| --- | --- | ---: | ---: | --- |
| f0760 |HEAD hanging-plane g00 Boundary|3.2s|-1.0971|false|
| f1625 |BODY protruding-plane g02 Boundary|3.4s|5.1915|true|
| f2560 |BODY suspended-solid g05 Boundary|3.2s|-16.2284|false|

Each adds0.2s, totaling0.6s and three false segments. Current B is below its high
cutoff on all three. This directly locates the added FP in holding, not an extra
model-positive on those frames. We did not change release/hold policy to repair
it. Existing A release tails and earlier FP remain in all denominators.

## Interpretation and closure

Useful complementarity exists in these observations and saved model scores.
However the new Calibration gate contributes nothing, grouped score calibration
adds1 CoreFP, and unchanged hold adds3 BoundaryFP. No tested arm passes the
complete zero-added-FP/segment condition. Fixed-high current alone passes the
local current-frame criterion but is explicitly consumed-dev-derived.

All67 fixed-high current rescues remain model classifications:43 have native
target/corridor contributors in an observed winning return and24 do not. Native
support was joined only after predictions; it never entered the screen or cutoff.
UNKNOWN remains328/384 Core and192/192 Boundary. Calibration eligibility does
not establish foreground ownership or exact measured range. No masks/IoU,
hardware performance, natural-scene or product-safety claim is made.

Four focused tests pass for float64 atomic cutoffs, exclusion of the full held
group, admissible calibration negatives, OR retention and nonrecursive/reset hold.
Independent audit replays folds, current/held metrics, identity sets, source hashes,
event timing and test non-activation without primary runner selection/metrics.
Input/code seals and all per-group predictions and false-frame identities are
retained under `artifacts.local/work/ba-spatial-complement-diagnostic-20260921/`.
CPU scalar replay is TASK_NOT_GPU_SUITABLE; no GPU/model/UE allocation was started.

Supported global registration still fails at existing ledger303; inheritance
assignment receipt and [local disposition](SPATIAL_COMPLEMENT_DISPOSITION_20260921.json)
preserve that metadata gap. The original A/B source files remain byte-identical.
This diagnostic ends here, with no threshold retry, trained gate, test access,
release-policy change, new cohort or automatic successor.
