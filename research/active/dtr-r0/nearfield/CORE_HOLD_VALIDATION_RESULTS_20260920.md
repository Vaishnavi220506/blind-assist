# Frozen strong-plus-hold policy: useful Core behavior on new complete layouts

2026-09-20. The fixed simple policy passes its prospective controlled Core
tradeoff check on one new 36-layout/432-frame cohort. Core Calibration74/133/0
becomes **74/10/0** with strong plus one genuine-strong-backed held frame.
All12 Core events are alerted throughout their positive samples. False-alert
segments fall26 to8 and sampled false-alert duration26.6 to2.0s (92.48% reduction).
Retain this as a Core-specific controlled-demonstration CHALLENGER and end the
algorithm round. The full-task Calibration baseline and old failures remain intact.

## Fixed comparison, not another algorithm search

[Protocol](CORE_HOLD_VALIDATION_PROTOCOL_20260920.md),
[source specification](core_hold_validation_spec.py) and
[runner](run_core_hold_validation.py) were frozen before capture. The original
proxy phase, score, thresholds (T0=.007085703945147101, T=.4071309640537889),
definite-support bypass and current-frame UNKNOWN remain unchanged. Only genuine
strong evidence can extend one adjacent sample; hold cannot renew itself.
No rising extrapolation, RGB branch, training, phase/threshold search or scene
replacement occurred. Prior864-frame parity and reset/causality checks passed.

This source changes dimensions, heights, relative placements, paths and backdrops
against both consumed cohorts. Relative geometry signatures, not just renamed or
globally translated scenes, are disjoint. It shares the six cuboid families,
Willow map, assets, renderer and single-return simulator. It is new controlled
arrangement evidence, not hardware, natural-distribution or a protected final set.
All432 source frames pass admission. Observations and all three prediction vectors
were sealed before evaluator join; the complete-layout geometry labels are unchanged.

Core contains the full INSIDE and OUTSIDE layouts:288 frames,74P/214N.
Boundary layouts contain144 frames,74P/70N. No near-entry samples inside a Core
layout were moved to Boundary after outcomes. Time below is nominal sampled time
at0.2s, not capture wall time or interpolated continuous alert duration.

## Complete Core result

| Readout | TP/FP/FN | Precision | Recall | F1 | FPR | FP segments | FP sampled seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Calibration | 74/133/0 | 35.75% | 100% | 52.67% | 62.15% | 26 | 26.6 |
| Fixed strong | 72/10/2 | 87.80% | 97.30% | 92.31% | 4.67% | 8 | 2.0 |
| Strong + one-frame hold | **74/10/0** | **88.10%** | **100%** | **93.67%** | **4.67%** | **8** | **2.0** |

| Core group / negative phase | Calibration TP/FP/FN | Strong | Hold |
| --- | --- | --- | --- |
| BODY (36P/108N) | 36/69/0 | 36/7/0 | 36/7/0 |
| HEAD (38P/106N) | 38/64/0 | 36/3/2 | 38/3/0 |
| OUTSIDE (144N) | 0/88/0 | 0/0/0 | 0/0/0 |
| INSIDE negative (70N) | 0/45/0 | 0/10/0 | 0/10/0 |

OUTSIDE false segments14 to0; sampled duration17.6 to0s out of28.8s negative
exposure. INSIDE-negative segments12 to8; duration9.0 to2.0s out of14.0s negative
exposure. Thus the remaining10 FP all precede entry, and inside-negative FPR is
still14.29%; the lower overall4.67% FPR includes the clean OUTSIDE layouts.
Every original Core positive remains alerted,123 Core FP are removed, and no
new Core FP is added relative to Calibration. All paired IDs are saved.

Current-observation UNKNOWN stays246/288 Core frames for all arms (390/432 full).
It is not removed from denominators or changed into a clear-space conclusion.
An alert, including a held alert, may coexist with this uncertainty flag.

## Event timing and the specific contribution of hold

All three arms detect12/12 Core events at the first labelled positive sample.
The candidate's maximum sampled onset delay is0s in this cohort, so the allowed
0.2s budget was not needed. However,8/12 candidate events already have an alert
on the preceding negative frame (Calibration12/12). This is not12 accurately
timed new triggers; the pre-entry false-alert cost remains in the table above.

Every held-policy event has100% positive-sample coverage and zero silent gaps.
Strong alone has two single-sample gaps, both HEAD:

| New-cohort frame | Clip | Time | Current score | Previous genuine strong | Effect of hold |
| --- | --- | --- | --- | --- | --- |
| f0008 | b0 head_horizontal INSIDE | 1.6s | .074331524 | f0007, score .905301819 | 5/6 to6/6 coverage |
| f0082 | b0 head_protruding_edge INSIDE | 2.0s | .264055628 | f0081, definite support | 6/7 to7/7 coverage |

Both current raw alerts are true but their scores are below T. Hold restores
these two decisions without changing current measurements and adds no FP in
this cohort. Strong accounts for the large FP reduction; hold contributes the
measured continuity improvement. Strong alone also meets this cohort's bounded
timing/coverage limits, so hold is not claimed necessary to pass those limits.

At the first in-event alert, target-front axial depth spans2.887 to2.996m.
This is geometry context, not human reaction time, collision time, or a measured
warning margin. The12-frame posed approaches do not test departure or post-event
alert release. Per-event first times, gaps and clip-first times are saved in full.

The earlier consumed primary result remains76/9/2 with two0.2s HEAD onset delays.
This new result does not repair those old onsets or erase the old scalar transfer
failure. It adds one prospective new-arrangement result for the combined policy.

## Boundary and full-source costs remain substantial

| Boundary readout (74P/70N) | TP/FP/FN | Detected events | FP segments / sampled seconds |
| --- | --- | --- | --- |
| Calibration | 72/17/2 | 12/12 | 10 / 3.4 |
| Strong | 4/4/70 | 3/12 | 3 / 0.8 |
| Hold | 9/4/65 | **4/12** | 3 / 0.8 |

Hold misses8 Boundary events and65/74 Boundary-positive samples. Its reported
zero delay on detected Boundary events excludes those8 misses. Over all432,
Calibration146/150/2 becomes hold83/14/65, events24/24 to16/24. This prohibits
claiming full strict-task replacement from the Core result. Thin-object coverage
was not newly evaluated; its existing challenge results remain unchanged.

## Decision, checks and delivery

All predeclared checks pass: all12 Core events, onset no later than0.2s,
at least50% FP reduction, nonincreasing Core/negative-phase false segments and
frames, and each event at least5/6 coverage with no gap longer than0.2s.
These were Development-informed pilot criteria, not human safety limits. The
earlier lossless-onset protocol and its disposition are not retrospectively changed.

Retain **COMPONENT_OR_CHALLENGER / CHALLENGER**, scoped to low-false-alert BODY/HEAD
controlled demonstrations with the declared geometry and severe grazing limitation.
No App/default promotion, replacement of full-task Calibration, new tuning or
automatic successor. Useful effect is the contribution; no major algorithmic
novelty is claimed. This completes the authorized consolidation/validation round.

Independent standard-library recount passes all432 identities,36 timelines,
spec/native geometry consistency, three causal decisions,15 metric strata, event
coverage/timing, changed IDs, frozen checks and recursive evidence hashes. It
treats the saved scalar/raw geometry scores as inputs; it is not a second numerical
integration of the score or hardware validation. The complete Core timeline plot
was rendered and visually inspected; it shows every Core sample for each arm.

UE captured once on its configured RTX5060 Laptop GPU; capture script wall time
was312.156s. Task actors and the tracked UE process tree are released with no
survivors. CPU geometry plus three readouts averaged2.400ms/frame (p95 3.394ms);
this excludes sensing/decoding/I/O and is not target-device latency.

Evidence: `artifacts.local/work/ba-core-hold-validation-20260920/`, including
protocol/source/prediction/evaluation seals, complete native/RGB observations,
results, independent-recount.py/json, process-release receipt and
`core-alert-timeline.png`. [Structured disposition](CORE_HOLD_VALIDATION_DISPOSITION_20260920.json)
binds their hashes. Global experiment registration remains pending the existing
ledger303 fingerprint error; supported-CLI receipts and local disposition retain
the gap without manual ledger edits.
