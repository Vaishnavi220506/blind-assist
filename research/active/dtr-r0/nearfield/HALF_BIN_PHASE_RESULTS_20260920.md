# One half-bin phase diagnostic: local sensitivity, no onset repair

2026-09-20 EXPLORE. One frozen global 5 cm shift of the 10 cm histogram
origin was evaluated on the consumed 36-clip/432-frame transfer cohort.
It did not change either delayed HEAD onset's critical returns or scores.
Other returns did change: one new HEAD strong-evidence loss propagated into
a later lost held alert. Keep the original proxy, Calibration and scoped hold
result. This diagnostic ends here; there is no selected replacement phase.

## Paired experiment and evidence boundary

[Protocol](HALF_BIN_PHASE_PROTOCOL_20260920.md) and
[implementation](half_bin_phase_20260920.py) fix both phases, all source frames,
eligibility, inverse-square bin weights, dropout, noise rule and readouts.
Both phases use the same original seeds, dropout draws and standardized Gaussian
variates. Noise sigma remains a function of the selected mean: this measures
paired full-proxy sensitivity, not a noiseless binning effect. Edge bins are
truncated by the unchanged [0.1,8) m domain. No phase/threshold search occurred.

Original phase0 reproduces all 432 stored 64-zone observations, missingness,
contributors and readouts. Observation vectors were sealed before prediction;
predictions were sealed before evaluator labels/native ownership were joined.
Ownership uses the existing native bounds proxy (including its 2 cm tolerance),
and is diagnostic only. No evaluator ownership enters predictions.

Initial construction stopped at f0000/z56 before any observation or prediction
output: NumPy float32 promotion made the recorded `noise/sigma` fail a 1e-14
pairing assertion. The recording expression was changed to
`float(noise)/float(sigma)`; actual noise, RNG calls and observations were not
changed. Prior code/start/protocol and a mechanical-fix receipt are retained;
the corrected code was frozen before successful construction. This was a
telemetry correction, not another scientific variant.

## Complete Core comparison

Core comprises 78 positive and 210 negative frames. FP time is sampled duration
at 0.2 s per frame, not interpolated continuous duration.

| Frozen readout | Original TP/FP/FN | Shifted TP/FP/FN | FP segments original/shifted | FP seconds original/shifted |
| --- | --- | --- | --- | --- |
| Calibration | 77/128/1 | 77/129/1 | 26/26 | 25.6/25.8 |
| Strong T=0.4071309640537889 | 73/8/5 | 72/8/6 | 7/7 | 1.6/1.6 |
| Strong + nonrecursive one-frame hold | 76/9/2 | 75/9/3 | 7/7 | 1.8/1.8 |

| Readout | Precision original/shifted | Recall original/shifted | FPR original/shifted |
| --- | --- | --- | --- |
| Calibration | 37.56%/37.38% | 98.72%/98.72% | 60.95%/61.43% |
| Strong | 90.12%/90.00% | 93.59%/92.31% | 3.81%/3.81% |
| Hold | 89.41%/89.29% | 97.44%/96.15% | 4.29%/4.29% |

All arms retain 12/12 Core events with unchanged first in-event alert times.
Both high-threshold HEAD onset delays remain 0.2 s. BODY strong/hold remains
36/7/0. HEAD strong changes 37/1/5 to 36/1/6; HEAD hold changes 40/2/2 to
39/2/3. Calibration BODY changes 36/67/0 to 36/68/0; HEAD stays 41/61/1.

OUTSIDE Calibration stays 90 FP/13 segments/18.0 s; strong/hold stay zero.
INSIDE negative Calibration changes 38 to 39 FP (13 segments, 7.6 to 7.8 s);
strong stays 8 FP/7 segments/1.6 s and hold 9 FP/7 segments/1.8 s.
Core UNKNOWN remains 246/288 under both phases; it was recomputed, not fixed.

Boundary flags are identical across phases: Calibration 77/16/1 with 12/12
events, strong 6/1/72 with 3/12, hold 9/1/69 with 3/12. Complete432 counts are
Calibration 154/144/2 to 154/145/2; strong 79/9/77 to 78/9/78;
hold 85/10/71 to 84/10/72. Complete UNKNOWN stays 390/432.

## What actually changed

Across all 27,648 zone-frames, candidate contributor sets change in 4,420;
observed contributor sets change in 4,405. Of 23,586 zone-frames observed under
both phases, 4,404 change range and 75 change by more than 1 m (41 frames).
Median absolute range change is zero; maximum is 4.640140 m. Candidate mean
maximum change is 4.596292 m. These are contributor identities, not bin-number
renaming. Changes occur somewhere in every frame, but almost all alert decisions
remain unchanged; raw sensitivity is not equivalent to alert instability.

Only f0243/z49 changes missingness: OBSERVED to NOISY_RANGE_OUTSIDE_LIMIT,
after the selected mean changes 5.736282 to 7.697028 m. Dropout is identical.
Winning edge-bin counts before output validity are 595 original and 594 shifted.
Observed ownership changes include TARGET to BACKGROUND in 6 zone-frames and
BACKGROUND to TARGET in 1; actual target-corridor contributor presence is gained
in 8 and lost in 9. Complete transition counts and frame-level records are saved.

The exhaustive changed alert IDs over all432 are:

| Readout | Change | Explanation |
| --- | --- | --- |
| Calibration | New FP f0182 | Negative score 0.002794697 to 0.009865495 crosses T0; selected target points remain outside the actual corridor. |
| Strong | Lost TP f0007 | z35 changes from 28 corridor-target points at mean 2.737659 m to 137 background points at 6.351005 m; frame score 0.949254 to 0.099420. |
| Hold | Lost TP f0008 | Its current score is unchanged; f0007 no longer supplies genuine strong evidence to hold from. |

There are no recovered TPs or removed FPs. Hold still covers f0007 from f0006,
but cannot recursively extend into f0008. The affected seven-frame HEAD event's
strong coverage falls 4/7 to 3/7 and maximum silence 0.2 to 0.4 s; held coverage
falls 6/7 to 5/7 with maximum silence still 0.2 s. Other11 Core event details are
unchanged. Per-event gaps, clip-first alerts and background strata are saved.

## Original onset hypotheses did not improve

f0004/5/6 scores remain 0.284675 / 0.026336 / 0.794798 under both phases.
At f0005/z35, both phases retain exactly the same competing contributor groups:
24 target points at 2.965004 m, weight 2.729988; 6 target points at 3.091987 m,
weight 0.627590; 135 background points at 6.585005 m, weight 3.113304.
Background still wins. The half-bin origin does not join these target groups.
Critical zones27/34/35 have unchanged contributor sets and output ranges.

f0292/3/4 scores remain 0.128146 / 0.244035 / 0.618651. At f0293/z36,
48 target-corridor contributors, mean 2.999005 m and reported 3.162859 m are
unchanged. Critical zones28/36/37 are unchanged in contributor identity/range;
z28's bin number changes only by renaming. This diagnostic does not resolve
either return selection at f0005 or interval scoring at the 1 mm entry f0293.

## Disposition and verification

Retain this as a COMPONENT diagnostic of measured proxy phase sensitivity.
The tested shift is not an onset repair or a promoted observation model. Local
phase sensitivity exists elsewhere, but does not explain the original onset
failures in this test. One offset is not evidence of a generally unstable proxy,
an optimal phase, real-hardware behavior, or an information ceiling of 8x8 ToF.
No contract relaxation, new threshold, phase sweep or automatic successor follows.

Evidence directory: `artifacts.local/work/ba-half-bin-phase-20260920/`.
It preserves source/protocol hashes, mechanical-fix receipts, paired records,
observation/prediction/evaluation seals, full metrics and all changed zone records.
Independent source reconstruction checks all 432x64x2 records without importing
the production sampler. The final independent receipt and its checks are linked
by [structured disposition](HALF_BIN_PHASE_DISPOSITION_20260920.json).

Registration remains blocked by pre-existing ledger303 fingerprint mismatch;
global terminal inheritance remains pending supported-CLI registration. Receipts
and local disposition preserve the gap; no ledger repair or manual bypass occurred.
