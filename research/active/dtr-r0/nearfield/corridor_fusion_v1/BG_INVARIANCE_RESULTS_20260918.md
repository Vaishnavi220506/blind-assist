# Texture invariance reduces score drift, but fails the alert objective

**The fixed B2 recipe does not deliver99TP with2--3FP.** On consumed clear216,
raw HGB99/5/9 becomes B2104/22/4. Texture consistency lowers held-pair score
drift relative to matched B1, but B1/B2 give identical held96 alert outputs.
Retain raw HGB and A*; close this recipe without a threshold/loss sweep, second
capture or automatic successor. This rejects the fixed loss/adapter recipe,
not all counterfactual methods or all remaining signal in the task.

## What was tested

The [frozen protocol](BG_INVARIANCE_PROTOCOL_20260918.md) uses raw2224 sensor
geometry and unassociated RGB summaries. HGB cannot optimize pair losses, so
each arm adds the same35,617-parameter2224->16->1 residual to frozen raw-HGB
log odds. This disclosed architecture change means only B0/B1/B2 isolate the
objectives; comparisons against HGB also include the adapter and extra data.
No explicit scene/intrusion latent factorization was implemented or claimed.

- B0: BCE.
- B1: BCE+.25*intrusion hinge, margin1 logit.
- B2: B1+.1*absolute logit difference on authenticated background pairs.

All use seed189018,120 full-batch AdamW steps,lr.001,weight decay.01,last
checkpoint, identical initialization and fit-only normalization/clip. Existing
1152 non-anchor OOF rows plus192 new rows train each final residual. Underlying
HGB stays fixed on original1344 rows. Six whole-scene outer folds and30 inner
HGB fits prevent training on in-fit old base scores; all texture/lateral members
of each new physical group stay together. New groups never enter HGB fitting.
All21 residual fits complete once. Crossfit-to-final base-score transfer remains
a limitation shared by all three arms.

One threshold per arm uses pooled1344 outer-held clear-F1, fewer FP, then higher
threshold. Chosen logits: B0 .5939988493919373, B1 .2746528387069702,
B2 .20751702785491943. These OOF scores are calibration, not independent results.
No report-driven adjustment occurred. Frozen references replay exactly.

## New intervention data and scope

One seed189018 UE capture contains12 physical configurations,2 lateral members,
2 context appearances and6 timestamps:288 frames. Physical indices0/1 train192;
index2 holds96 across4 configurations. Native geometry/camera/target appearance,
reflectance and sensor seeds are identical across each texture pair. Only
background texture enable/grid/seed changes: plain versus procedural grayscale
tiles. Context remains behind target and outside the body corridor. This is not
a geometry/clutter/occlusion intervention or evidence about arbitrary backgrounds.

All144 texture pairs have **exactly identical public non-RGB sensors**, native
task labels and sampled corridor-return support; every pair changes RGB. Native
objects match source geometry within1.02e-15m. All319 return files authenticate
and288 images decode. A plain/textured rod example was visually inspected.
The96held frames supply48 background and48 intrusion pairs; only4 physical
groups are independent configurations. Texture duplication does not double
independent event evidence. Target visibility admission uses fixed geometry and
depth ordering, not an independently labeled RGB visibility mask.

Source/native labels were inspected for acquisition admission before training;
held labels never enter fitting, normalization, threshold selection or checkpoint
choice. Both report prediction sets were sealed before performance-label joins.
Original MZ136 dev/test remain excluded. The old288 report is consumed Development;
new96 is a small same-generator Development transfer pilot, not protected final,
natural-distribution, device or safety evidence.

## Existing288 report: target fails

Clear216/288=75% coverage, with108 positives/108 negatives. Counts are TP/FP/FN.

| Method | Clear counts | P / R | Clear F1 | Strict counts | Strict events | Clear false episodes |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Raw HGB | 99/5/9 | 95.19 / 91.67% | 93.40% | 111/9/33 | 25/30 | 4 |
| A* | 96/3/12 | 96.97 / 88.89% | 92.75% | 107/6/37 | 24/30 | 3 |
| B0 | 100/16/8 | 86.21 / 92.59% | 89.29% | 118/20/26 | 26/30 | 7 |
| B1 | 103/21/5 | 83.06 / 95.37% | 88.79% | 122/26/22 | 27/30 | 8 |
| B2 | 104/22/4 | 82.54 / 96.30% | 88.89% | 123/28/21 | 27/30 | 9 |

Boundary72: raw12/4/24,A*11/3/25,B018/4/18,B119/5/17,B219/6/17.
Rod clear72: raw29/4/7,A*26/2/10,B031/9/5,B133/13/3,B234/14/2.
The extra rod sensitivity is accompanied by many more nuisance alerts.

B2 rescues5 raw clear FN, loses0 raw clear TP, adds18 clear FP and removes1.
It improves clear correctness in1 configuration, ties10 and worsens7. B0 loses3
raw clear TP and B1 loses1. All retain18/18 core events; B0 delays rod scene1
by.25s, while B1/B2 improve rod scene5 from.5s to0. Each residual loses1 raw
strict event; B0 gains2 and B1/B2 gain3. Net event counts cannot be called full
retention. Existing core events are left-censored, not advance warnings.

Strict native-supported true alerts: all residuals retain every raw native TP;
B0 rescues7,B1/B2 rescue8. This useful retention does not compensate for FP
growth. Zero-return and clear-native intersections are retained in audit.json;
missing returns remain UNKNOWN and no-alert does not certify free space.

Old intrusion ordering is raw124/144,A*124/144,B0131/144,B1133/144,B2130/144.
Thus the extra invariance term does not improve old paired ordering over B1.

## New held96: drift improves without an alert gain

Clear72/96=75%,36 positives/36 negatives. All methods detect6/6 core events
(three clear physical configurations times two textures).

| Method | Clear counts; F1 | Strict counts | Boundary24 | Strict events |
| --- | --- | --- | --- | ---: |
| Raw HGB | 34/0/2;97.14% | 34/4/14 | 0/4/12 | 6/10 |
| A* | 34/0/2;97.14% | 40/4/8 | 6/4/6 | 10/10 |
| B0 | 30/0/6;90.91% | 32/6/16 | 2/6/10 | 8/10 |
| B1 | 34/0/2;97.14% | 40/9/8 | 6/9/6 | 10/10 |
| B2 | 34/0/2;97.14% | 40/9/8 | 6/9/6 | 10/10 |

B1/B2 match clear totals but exchange2 raw true frames for2 rescued frames;
B0 loses4 rod true frames and delays both texture versions of that rod event
by.5s. B1/B2 retain raw event timing. A* has the same strict TP/event count as
B1/B2 with5 fewer FP. All methods have0 clear false episodes: this held subset
offers no baseline clear false-positive reduction opportunity.

| Method | Mean texture logit drift | Mean probability drift | Alert flips /48 | Intrusion order /48 |
| --- | ---: | ---: | ---: | ---: |
| Raw HGB | .145494 | .014802 | 0 | 39 |
| A* | .767649 | .043182 | 2 | 44 |
| B0 | .138224 | .009840 | 0 | 38 |
| B1 | .128174 | .007607 | 1 | 38 |
| B2 | .090825 | .006872 | 1 | 38 |

B2 reduces mean logit drift29.14% and probability drift9.66% versus B1, but
produces identical held alerts, event counts and intrusion ordering. Its boundary
probability drift actually rises.022849->.024123, while rod drift falls
.005434->.001440. Both have the same one boundary alert flip. Global stability
must not be presented as uniform per-family improvement. BCE alone already
reduces drift relative to raw, so B2-versus-raw differences are not all caused
by the invariance term. The observed matched effect is a narrow score-stability
component, not the proposed causal-learning performance breakthrough.

## Decision, checks and delivery

All three arms fail the predeclared low-FP/retention target. Mark this fixed B2
alert recipe NEGATIVE_CONTROL; preserve drift reduction as diagnostic evidence,
not a promoted algorithm. Keep raw HGB and A* with their respective tradeoffs.
Do not claim the models disentangle scene and intrusion factors, background
shortcuts caused old errors, or HGB has exhausted all available information.
No loss/threshold sweep, further capture, new factorized architecture, project
switch or App-default change followed this result.

Four source-authentication and four objective tests pass. Separate audit replays
18OOF residual models and6 final report evaluations, reselects all3 thresholds
using exact rational F1 ordering, verifies fit-only normalization/group/pair
isolation, and independently recounts384 frame predictions and event onsets.
It reuses native labels and Torch forward; not an independent physical evaluator.
Public sensor/native source parity and original baseline score identity also pass.

Real-step medians CPU4.066ms versus CUDA2.546ms select RTX5060 Laptop CUDA,
Torch2.11/cu130. Old30HGB fits take93.76s on CPU (sklearn GPU backend unavailable).
These are offline compute measurements, not online/device latency.

Artifacts: `artifacts.local/work/corridor-bg-invariance-20260918/` holds protocol/
recipe/source/code seals,30 inner HGBs,21 residual checkpoints,OOF selections,
old/new cases,predictions,audit,diagnosis and capture receipts. Models can be
replayed with `audit_bg_invariance.py`; `test_bg_invariance.py` and
`test_bg_invariance_source.py` cover the focused contracts. Stage runner is
`run_bg_invariance.py crossfit|admit|train|prediction|evaluate`; completed stages
refuse overwrite. `prepare_bg_capture.py` preserves the inherited capture code.

One worker acquisition ran00:44:41--00:59:11HKT on2026-09-18 without restart.
UE,scheduled task and Zen port20638 are verified released. Owned reproducible DDC
was removed,455,099,412 logical bytes. Worker raw remains owned by this experiment
at `G:/DevWorkspace/BlindAssist/artifacts/work/corridor-bg-invariance-20260918/capture-v1`;
controller copy is `source/returned-v1/capture-v1`. These are durable paired-data
evidence, retained for reproduction; no process or paid allocation remains.

Registration remains pending the existing ledger303 input-fingerprint mismatch;
supported inheritance assignment reports unknown terminal. Logs retain both
failures without rewriting the ledger. Intended role is NEGATIVE_CONTROL for
the fixed alert recipe, with no blanket rejection of counterfactual learning.
