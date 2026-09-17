# Public positive evidence implemented: learned signal, no gain at selected working points

**A real public-input return classifier is implemented and trained.** The
predeclared, scene-calibrated OR policy leaves all final alerts unchanged:
changed-domain clear **95 / 27 / 13, F1 82.61%**; old-domain clear
**107 / 22 / 1, F1 90.30%**. It therefore does not establish a selected operating
point improvement. A separate posthoc fixed-zero-logit diagnostic recovers all
8 oracle-supported changed-domain clear misses, at a cost of 6 new FP:
**103 / 33 / 5, F1 84.43%**. This diagnostic demonstrates useful public signal
with a real false-alert cost; it is not substituted for the predeclared result.

No capture, RGB/pretrained-depth execution, threshold sweep on report outcomes,
veto, or change to A occurred. This is grouped consumed Development, not fresh
confirmation or a single final deployed model.

## What was built and reused

The [fixed protocol](PUBLIC_POSITIVE_PROTOCOL_20260917.md) uses a shared
**1,537-parameter 30 -> 32 -> 16 -> 1 ReLU MLP**. Inputs are the 21 public
MZ161/MZ174 token fields plus nine explicitly computed corridor relations:
signed center, possible-envelope-overlap and full-envelope-containment margins
on each axis. These remain hypotheses from zone/range/IMU; no native point
coordinate, lineage, actor ID, family, scene or label enters inference.

[public_return_tokens.py](public_return_tokens.py) removes MZ161's unused dense
image work. All TRAIN192 tokens and validity masks match the authenticated old
cache bitwise, including 181 MERGED tokens and 287 outside/partial-RGB tokens.
The interface retains 132 x 21 tokens; this pointwise head uses only the 128 ToF
tokens. The four unsupervised Radar tokens do not acquire a learned readout;
Radar and RGB remain in the unchanged A branch. No new RGB context was tested.

MZ171's sampled-point witness labels supervise each actual usable return.
Positive means at least one resolved contributor in the nominal corridor;
negative means that return's complete resolved lineage has no such contributor.
It never means the whole corridor is empty. UNKNOWN receives zero loss. The
anchor192 targets reproduce the authenticated prior labels exactly. All 13,696
usable ToF slots in the reused 768-frame pool have known labels, including
2,386 positive slots; invalid slots and all Radar slots stay unsupervised.

The branch outputs the maximum valid-ToF logit and is ORed with frozen A.
No usable ToF return means no positive contribution, with A retained. This
preserves existing A alerts/timing by construction; added false alerts remain
possible. The public-only [inference entry](public_positive_inference.py) loads
one checkpoint and its associated threshold, accepts a public row, causal yaw
and A's alert, and never loads labels or dataset metadata. Offline fold identity
chooses a held-out model during evaluation only, not during product inference.

## Grouped training and selection

The old dev48 has zero A misses, making it uninformative for this branch's
desired benefit. Instead use six fixed outer scene-index folds across the two
consumed 288-frame cohorts, with original MZ136 TRAIN192 as fitting anchors.
Per fold: **576 fit / 96 calibration / 96 report**. Report scene index k,
calibrate index (k+1) mod 6, fit the remaining indices plus anchors. Both paired
trajectories and every frame in a scene group stay together. Every reported
frame is excluded from its own model's fit, normalization and threshold choice.
Original test48 is excluded. Shared generator templates and prior diagnostic
access mean this is not an independent-domain confirmation.

Each fold uses one seed, AdamW .001, weight decay .0001, batch16, 120 epochs,
equal total fitting mass for known positive/negative returns, and the final
checkpoint. Fit-only standardization uses a 0.01 std floor. No architecture,
checkpoint or seed selection took place. Tau maximizes calibration final-OR
clear F1, then fewer FP, then higher threshold. Report outcomes never choose it.

## Predeclared main result

All clear denominators are 216/288 (75% coverage), with 108 positives and 108
negatives. Every reported frame uses its own held-group model and calibration.

| Cohort / method | TP / FP / FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Changed A | 95 / 27 / 13 | 77.87% | 87.96% | 82.61% |
| Changed A + public positive, selected | 95 / 27 / 13 | 77.87% | 87.96% | 82.61% |
| Changed sampled-support OR oracle reference | 103 / 27 / 5 | 79.23% | 95.37% | 86.55% |
| Old A | 107 / 22 / 1 | 82.95% | 99.07% | 90.30% |
| Old A + public positive, selected | 107 / 22 / 1 | 82.95% | 99.07% | 90.30% |
| Old sampled-support OR oracle reference | 108 / 22 / 0 | 83.08% | 100% | 90.76% |

Both complete 288-frame policies also remain exactly A: old 137/28/7 and new
120/45/24. Strict events remain old 30/30, new 29/30; core events remain 18/18
in each cohort. No old TP, first-alert time or zero-return fallback changes.
No selected operating point adds any final true or false alert.

## Why calibration suppressed the useful signal

The return head fits its training partitions: final zero-logit slot precision
87.89-91.93%, recall 98.25-99.54%. Held-group zero-logit return metrics are
old 891TP/100FP/41FN/3206TN (precision 89.91%, recall 95.60%) and changed
888/128/16/5875 (precision 87.40%, recall 98.23%). These are correlated return
records, not independent scenes or final-alert results.

| Report scene index | Calibration index | Calibration clear A FN | FN with returned sampled support | Selected logit tau |
| --- | --- | ---: | ---: | ---: |
| 0 | 1 | 2 | 0 | 11.76984 |
| 1 | 2 | 0 | 0 | 14.53380 |
| 2 | 3 | 1 | 1 | 13.68990 |
| 3 | 4 | 0 | 0 | 12.64050 |
| 4 | 5 | 2 | 0 | 11.70115 |
| 5 | 0 | 9 | 8 | 5.81921 |

Five folds choose the finite threshold just above the largest calibration score,
preserving A's calibration F1. Only the fold calibrated on scene0 sees the eight
changed-domain support-backed misses and chooses a useful lower threshold there.
Its report is scene5, where A's two clear misses have no ToF return. Conversely,
report scene0 contains all eight recoverable misses, but its calibration scene1
contains no support-backed missed frame. Scene-level opportunity imbalance and
score/working-point transfer are material limitations of this small calibration
design, separate from whether the pointwise head learned discrimination.

The protocol's "inactive candidate" is **inactive on calibration samples only**:
the code uses finite max(calibration score) plus float64 nextafter, not a global
off switch. Report folds0/1/4 do activate on 6/5/6 frames, respectively; all are
already A alerts. The independent audit reproduced this behavior and found no
post-seal numerical/selection defect. No sealed method was changed to rescue it.

## Single posthoc fixed-zero-logit diagnostic

After finding no selected alert gain, replayed the saved logits once at the
standard logit0 / sigmoid0.5 point. No refit, threshold search or best-report
selection occurred. **This is explicitly posthoc**, not the predeclared selected
policy and not a new validation cohort.

| Cohort | Clear TP / FP / FN | Precision | Recall | F1 | A FN recovered | New FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Old | 108 / 23 / 0 | 82.44% | 100% | 90.38% | 1 | 1 |
| Changed | 103 / 33 / 5 | 75.74% | 95.37% | 84.43% | 8 | 6 |

Changed-domain HEAD becomes 36/8/0 from 31/8/5. BODY stays 36/6/0. Rod becomes
31/19/5 from 28/13/8: all six new clear FP are rod/far-wall, including five
correlated frames in one out-of-corridor episode and one in another episode.
All eight rescued clear positives have actual sampled corridor witnesses. The
same model thus exposes the sought public-input signal, while mislocalizing
some background/rod evidence into a positive warning. A scalar maximum amplifies
even a small number of wrongly positive returns into frame alerts.

The clear false-alert burden rises from 25% to 30.56% in the changed cohort;
precision falls despite the 1.82 pp F1 gain and improved recall. All 18 core events
remain detected. HEAD scene0 first alert changes 1.25 -> 0 s; rod scene0 changes
0.75 -> 0.25 s; no A onset is delayed. These episodes were already positive at
their first observation, so this is faster observed response, not advance warning.

Boundary pressure is substantially worse and must accompany the clear result:

| Cohort / policy | Strict TP / FP / FN | Strict F1 | Boundary TP / FP / FN |
| --- | --- | ---: | --- |
| Old A / selected | 137 / 28 / 7 | 88.67% | 30 / 6 / 6 |
| Old zero-logit diagnostic | 142 / 46 / 2 | 85.54% | 34 / 23 / 2 |
| Changed A / selected | 120 / 45 / 24 | 77.67% | 25 / 18 / 11 |
| Changed zero-logit diagnostic | 138 / 66 / 6 | 79.31% | 35 / 33 / 1 |

Zero-logit strict events become 30/30 in both cohorts. Of 18 added strict true
frames in the changed cohort, 16 have sampled witnesses and two do not; true
frame classification alone does not certify correct return localization. Do not
hide old-domain strict degradation behind the adopted clear-task denominator.

## Execution, checks and disposition

The six fits complete in 33.39 s including placement/reduction. Equivalent
batch16 forward/backward medians are CPU 0.470 ms and CUDA 1.483 ms, selecting
CPU (`CPU_FASTER_MEASURED`). The public-only per-frame token frontend plus head
replay measures mean 1.64 ms / p50 1.45 ms / p95 2.82 ms on this host, excludes
A's frontend, capture/transport and is not phone or complete-chain latency.
All 576 public-only replay logits and decisions match saved held outputs exactly.

Four token tests and four head/readout tests pass. A pre-fit test caught NumPy
float32 threshold comparison rounding; both calibration and inference scores
were promoted to float64 before fitting. No trained outcome was consumed to fix
it. Independent audit reconstructs checkpoint layers without the model helper:
all held/calibration logits match bitwise, train-only normalization/orders and
whole-group exclusion pass, calibration candidates/choice match, and A/UNKNOWN
retention plus event timing pass. Input/source/prediction seals remain intact.

Retain the public token adapter, tiny trained evidence component and honest
zero-logit diagnostic as reusable Development evidence. **This selected
calibration/readout recipe is a no-alert-gain result**, and no final method is
promoted. Public evidence is learnable here; robust final-score calibration and
rod/boundary false activation remain unresolved. The diagnostic motivates those
questions without authorizing another run or changing this experiment's result.

Artifacts: `artifacts.local/work/corridor-public-positive-20260917/` contains
authenticated public/label caches, six checkpoint/optimizer states, fit orders,
calibration and held logits, seals, all cases, primary and posthoc reports,
public-only inference audit and independent audit. No worker, paid allocation
or persistent process remains. Existing sources, A and App defaults are unchanged.

The supported registration again fails at the existing
`experiments/index.jsonl:303 input_fingerprint` mismatch; inheritance reports an
unknown terminal ID. Receipts remain in the artifact root. The intended component
disposition and registration are pending; no shared ledger was edited or bypassed.
