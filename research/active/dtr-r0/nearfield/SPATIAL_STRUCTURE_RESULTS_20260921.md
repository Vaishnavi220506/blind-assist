# Ordered spatial contrast: geometry addition not supported

2026-09-21. **Both new arms fail the frozen usability criteria; the added-geometry
hypothesis is NOT_SUPPORTED.** With identical RGB structure, model, initialization,
batch schedule and compute, U and G give exactly the same decisions on all 96
HEAD-horizontal Boundary clip frames: 15/43 positive frames, 2/4 events. Neither
reaches 22 true frames or three timely events. Both also exceed transfer false-alert
budgets. Close this exact pair as `NEGATIVE_CONTROL`; no retry or successor.

[Protocol](SPATIAL_STRUCTURE_PROTOCOL_20260921.md), [runner](run_spatial_structure.py),
[model](spatial_structure_model.py), [tests](test_spatial_structure.py).
Evidence: `artifacts.local/work/ba-spatial-structure-20260921/`.

## What this contrast establishes

Both arms losslessly rearrange the same sealed frozen encoder cache into a
24-channel 24x24 ordered grid. This is nine existing samples per 8x8 zone, not
new native-image resolution. Four shared sensor/ray channels and the same
65,825-parameter convolutional head are used. G adds eight deterministic ray /
original-interval boundary-distance channels; U zeros those slots. Invalid
distance channels remain masked and neither head outputs measured depth.

Only the representation of available geometric information differs. Both use
the same initial-state hash, 1,200 batches, labels, optimizer and CUDA device.
Each arm is fitted once, with one dev-selected threshold under the same budget
contract. This is not equal realized FP, equal numerical thresholds, or an
isolated test that earlier band pooling caused the miss. Zero input slots also
do not imply equal effective feature capacity. No geometry-usefulness or
spatial-information ceiling follows from this one fixed negative contrast.

Original train24 layouts/1,728 frames; dev8/576; consumed transfer16/1,152.
Whole-layout partitions are disjoint, but transfer previously informed the
hypothesis. This is controlled reused Development, not fresh confirmation.
No protected test images/labels/inference, native-depth inference, new capture,
hardware, demo promotion or threshold retry occurred. A/UNKNOWN are retained.

## Paired task effects

A is frozen Core; B is the earlier Spatial-BCE supplement; R is the preceding
interval-band package. B/R are unchanged descriptive controls with different
historical recipes. U and G are the matched comparison. Counts are TP/FP/FN.

| Region / readout | A | B | R | U | G |
|---|---:|---:|---:|---:|---:|
| Core current | 171/17/2 | 173/36/0 | 171/20/2 | 173/24/0 | 173/25/0 |
| Core hold | 173/33/0 | 173/58/0 | 173/38/0 | 173/44/0 | 173/45/0 |
| Boundary current | 10/1/163 | 128/10/45 | 62/1/111 | 58/4/115 | 59/3/114 |
| Boundary hold | 14/5/159 | 140/19/33 | 78/6/95 | 66/11/107 | 68/9/105 |

| Hold metric | U | G |
|---|---:|---:|
| Core recall / precision / FPR | 100% / 79.72% / 7.39% | 100% / 79.36% / 7.56% |
| Core events / false segments / false sampled seconds | 16/16 / 28 / 8.8 | 16/16 / 28 / 9.0 |
| Boundary recall / precision / FPR | 38.15% / 85.71% / 5.21% | 39.31% / 88.31% / 4.27% |
| Boundary events / false segments / false sampled seconds | 9/16 / 9 / 2.2 | 10/16 / 8 / 1.8 |
| Boundary gain layouts over A | 7/16 | 7/16 |
| Horizontal TP/FP/FN; events | 15/2/28; 2/4 | 15/2/28; 2/4 |

Both horizontal current readouts are also identical. The retained hold detects
g00 (11/12 coverage, .2s first delay) and g03 (4/11, zero first delay); g01/g02
are wholly missed. Across Boundary, all detected U/G first delays are <=.2s,
but seven U and six G events are wholly missed. Good delay on the detected subset
does not represent those misses.

G gains four held positive frames and loses two versus U: net +2 Boundary TP.
It adds one Core FP and removes two Boundary FP. No horizontal decision changes.
The geometry hypothesis fails its >=5 extra horizontal TP / >=2 improved
horizontal-layout criteria as well as usability; a small overall net gain does
not support the proposed repair.

| Boundary family | Positive frames | A hold TP | B | R | U | G |
|---|---:|---:|---:|---:|---:|---:|
| BODY protruding plane | 45 | 3 | 33 | 31 | 16 | 20 |
| BODY suspended solid | 44 | 11 | 44 | 18 | 13 | 11 |
| HEAD hanging plane | 41 | 0 | 26 | 29 | 22 | 22 |
| HEAD horizontal | 43 | 0 | 37 | 0 | 15 | 15 |

This pair recovers some horizontal cases relative to R while losing coverage
elsewhere and adding FP. As several architectural/feature choices differ from R,
that observation is not an isolated causal proof about spatial ordering.

## Cost, timing and support boundaries

Added values are relative to A; fixed current/hold FP caps are 5/11 Core and
2/4 Boundary. False-segment caps are four Core and two Boundary per readout.

| Readout | U added FP / segments | G added FP / segments | FP / segment caps |
|---|---:|---:|---:|
| Core current | 7 / 4 | 8 / 4 | 5 / 4 |
| Core hold | 11 / 4 | 12 / 4 | 11 / 4 |
| Boundary current | 3 / 3 | 2 / 2 | 2 / 2 |
| Boundary hold | 6 / 4 | 4 / 3 | 4 / 2 |

Core held coverage remains 100%, all 16 onsets have zero in-event delay; existing
INSIDE pre-entry/post-exit false frames stay9/24. Added held Core FP are entirely
OUTSIDE. Boundary initial/internal/terminal silent positive frames are72/35/0
for U and63/35/7 for G. Boundary pre-entry/post-exit FP are2/9 and2/7, versus A1/4.
Every UNKNOWN and every A alert is retained, including learned-only alerts.

U/G add50/51 current true positives;15/16 of them respectively lack native
returned target-corridor contributors in the sealed source admission audit.
These are classifications, not independently measured obstacle ranges. Scalar
alerts have no object-mask IoU claim. Full BODY/HEAD/background/layout metrics,
event gaps and release records remain in the saved metrics and frame results.

## Why the negative result is useful

Both final heads have zero zero-logit classification errors on training rows.
Dev selected U15.960611343383789 and G15.010066986083984. Core dev current stays
85/4/1 for both, while Boundary becomes U45/3/41 and G44/3/42. For each model,
the already computed immediately lower dev breakpoint would add two Boundary
hold false segments against the cap of one. That explains the local selection
boundary; neither cutoff was lowered after seeing transfer.

Both selected systems then exceed transfer costs. The observed failure concerns
generalization of coverage and specificity across layouts under this readout
contract. It is not simply failure to fit the training set. The added explicit
geometry representation did not resolve that failure in this comparison.

## Execution and disposition

Three focused tests pass: exhaustive ordered feature placement, independently
calculated endpoint query geometry/U-G equality outside the ablated channels,
and missing-range behavior preserving RGB. Prepare also exactly inverts every
selected cached feature row. CPU/GPU batch64 training probes measured11.383/2.803ms;
inference3.049/.486ms. Both arms actually ran on cuda:0; fits took2.75/2.69s.
These are host batch/head measurements, not full sensor/device latency.

[Independent saved-output audit](verify_spatial_structure.py) passes551,024
assertions: all3,456 input rows,47,775,744 ordered RGB values,15,925,248 geometric
values, reproduced common initial weights and1,200x64 batch indices, five seals,
1,154 saved dev candidates and11,520 transfer decisions. Independent binary64
geometry reconstruction differs by at most2.416e-7 in normalized channel values;
the ordered RGB mapping is exact. Full event/subgroup metrics and terminal gates
agree. No encoder, trained head or training is rerun by this audit. Receipt/log:
`independent-verification.json` and `independent-verification.log`.

Structured local inheritance assigns both U/G `NEGATIVE_CONTROL` for this exact
fit/representation/selection question. A, B, R and previous negative terminals
remain unchanged; no successor or protected-test opening. The global supported
registration still fails at existing index303 fingerprint mismatch; inheritance
returns unknown terminal. Local receipts preserve the gap without editing the
global ledger. Source/code, inputs, fits, selection, predictions and evaluator
outputs are provenance-bound; all durable evidence and both checkpoints remain.
