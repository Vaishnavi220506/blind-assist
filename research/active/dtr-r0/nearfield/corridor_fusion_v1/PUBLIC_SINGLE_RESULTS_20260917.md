# Single public positive version: frozen configuration confirmation

**The single model is runnable, but this new cohort contains no clear A miss
with returned corridor support: clear A and A+head both score97/25/11.** The
head adds one boundary TP and one boundary FP. Its supported-miss recall benefit
is therefore untested here, not disproven. No new clear false alert appears.

**The same-data retrained A* is the strongest clear-F1 result in this comparison:
96/3/12, F1 92.75%, versus frozen A84.35%.** It removes22 clear FP, rescues3 FN
and loses4 existing TP; one rod onset is delayed0.5s and strict event detection
falls25/30 to24/30. Retain this meaningful precision/F1 challenger with its costs,
not an unconditional replacement or a head-specific algorithm gain.

The prior [v2 Development result](PUBLIC_POSITIVE_V2_RESULTS_20260917.md) remains
unchanged. Every current method uses one frozen model and threshold for every
frame, with no outer-fold selection or scene routing. The one prescribed capture
and evaluation are complete; no new-outcome threshold tuning or successor ran.

## Frozen model and data comparison

The [protocol](PUBLIC_SINGLE_PROTOCOL_20260917.md) fits the ordinary balanced
return BCE model on all 1,344 existing frames: anchor192 plus four complete
288-frame cohorts. The public ToF head remains 30->32->16->1, 1,537 parameters,
with fit-data normalization, 120 epochs, seed185017 and the last checkpoint.
No new RGB/depth encoder or loss is added. Fitting takes 12.64 s on the measured
faster CPU backend. The original MZ136 dev48/test48 remain excluded.

The single positive threshold is **5.8390960693359375**, selected using the
1,152 previous BCE outer-held logits and the predeclared final-OR clear-F1 rule.
These are consumed Development selection scores, not predictions from the new
all-data head and not fresh validation. At this one threshold, pooled selection
clear counts are A415/97/17 versus OR424/97/8; strict counts are A529/141/47 versus
OR538/144/38. The three added boundary FP in this pooled selection must not be
confused with v2's six separately selected thresholds, which added no FP.
Score transfer from held-fold models to the all-data model is tested below.

The same-data **A*** comparator uses all 1,344 authenticated 2,485-feature rows,
the inherited HGB structure and strict frame labels. Six whole-scene OOF fits
select its standalone clear-F1 threshold **0.5568065433174727**, then one final
model fits all rows. Structure: 150 iterations, 7 leaves, learning rate .05,
L2 1, minimum leaf8, no early stopping or weighting. The seed is fixed185017
instead of original177017. Fit/selection takes 27.84 s. OOF clear398/12/34 is
selection performance, not the final model's confirmation result. A* has frame
supervision; the public head has richer per-return supervision. This is a
data-amount comparator, not an identical-supervision architecture ablation.
Original A, threshold .3917890013717321, remains the OR base throughout.

| Artifact | SHA256 |
| --- | --- |
| Frozen original A | `d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef` |
| Single public head | `e159434001ab018a107940218df12398dc0dfc3b0bb0c364d73c7552876a08a1` |
| Same-data A* | `dffd5c5a2d6d59c5a0546513148b21736d5aa8601e5a034bcdab9337295cd59c` |
| Recipe freeze | `786d19de6a599c4d562275503d7edbf6ba6fda8bab49951b6ab9ca594cb5d53b` |
| New source specification | `a1316dfae4c2ed67176512e3b391df4da70703ecf0f7d0f79f0a85d2b8630234` |

## Complete source and evidence boundary

Seed186017 fixes 24 new paired configurations, 48 six-frame 4 Hz episodes and
288 frames before acquisition. Changed HEAD heights/sizes, rod widths/positions,
wall distances 3.25/3.55/3.85/4.15/4.45/4.75 m and background geometries provide
targeted stress conditions. Styles: 8 portal, 5 staggered, 7 recessed and 4 full
wall configurations. All scene objects count in truth; near backgrounds retain
an opening. The two members of each pair differ only in lateral target position.
Whole-trajectory signatures do not duplicate any of the five checked sources.

The source contains 144 strict positives and144 negatives. Main 5 cm clear
evaluation keeps216 frames, 108 positives and108 negatives, **75% coverage**;
72 boundary-pressure frames remain separately reported. No configuration is
selected from A misses, new-head success, sensor hits or measured logits. Same
renderer, synthetic materials and sensor model remain explicit limitations;
this is new-configuration confirmation, not independent-generator, hardware,
natural-distribution or user evidence.

The capture uses the existing no-timeout worker adapter. Copied helper metadata
contains an inert historical `recovery.json` and S1/gate wording; active source,
recipe, dispatch and bundle hashes identify this new run. These inherited words
do not describe a model retry or DA-V2 execution. Native geometry is inspected
for controller source admission, but never enters the prediction process.
All public outputs are sealed before the scoring process joins native truth.

## Final alert results

| Method | Clear TP/FP/FN | Precision | Recall | F1 | Core events |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen A | 97/25/11 | 79.51% | 89.81% | 84.35% | 17/18 |
| A + single public head | 97/25/11 | 79.51% | 89.81% | 84.35% | 17/18 |
| Same-data A* | **96/3/12** | **96.97%** | 88.89% | **92.75%** | **18/18** |

A* improves clear F1 by8.41 percentage points and reduces clear FP88%, while
recall falls0.93 points. Negative-frame alarm burden falls25/108=23.15% to
3/108=2.78%; false clear episodes fall12 to3 and false clear segments13 to3.
These are sampled controlled burdens, not naturally occurring daily alert rates.

| Stratum | A TP/FP/FN; F1 | A+head TP/FP/FN; F1 | A* TP/FP/FN; F1 |
| --- | --- | --- | --- |
| Strict, all288 | 111/37/33; 76.03% | 112/38/32; 76.19% | 107/6/37; 83.27% |
| Boundary72 | 14/12/22; 45.16% | 15/13/21; 46.88% | 11/3/25; 44.00% |
| BODY clear72 | 35/4/1; 93.33% | 35/4/1; 93.33% | 35/1/1; 97.22% |
| HEAD clear72 | 36/10/0; 87.80% | 36/10/0; 87.80% | 35/0/1; 98.59% |
| Rod clear72 | 26/11/10; 71.23% | 26/11/10; 71.23% | 26/2/10; 81.25% |

Strict precision/recall are A75.00/77.08%, A+head74.67/77.78%, A*94.69/74.31%.
Across18 clear-eligible configurations, A* increases correct-frame counts in9,
ties9 and decreases0. Across all24 strict configurations it improves12, ties11
and worsens1. Tied correctness can hide a TP/FP exchange: rod scene5 loses3 TP
while removing3 FP, and HEAD scene2 loses1 TP while removing1 FP.

The public branch is positive on78/288 frames,76 clear and2 boundary. All76
clear activations are already A true positives; no clear negative activates it.
Post-seal frame-level comparison matches sampled-witness presence on all216
clear frames. This is an observed frame-level agreement, not perfect return
localization or demonstrated rescue. A has11 clear misses across3 episodes:
10rod frames have no native ray hit on the target despite48--50 other usable
returns, and1 BODY frame has a missing ToF packet despite native ray hits.
Thus **0/11 clear misses supply the intended returned-support repair opportunity**.
Sixty-nine zero-return frames preserve A, as required. No source is regenerated
to manufacture repair opportunities.

The two added public alerts share boundary configuration scene2 at time0:
`exit_00` is a true alert whose maximum-scoring return itself has native corridor
support; `enter_00` is a false alert whose maximum return has no such support.
Both exact public scores and the selected return slots replay bitwise. This is
one physical configuration, two episodes and two frames, not independent clear
successes. It exposes a small boundary cost while preserving the earlier v2
Development rescue result and leaving clear rescue transfer unresolved.

## Event retention and timing

A+head retains every A alert and all onsets. Core detection remains17/18 and
strict detection25/30. Boundary scene2 exit first alert improves0.25->0s;
all other detected event onsets remain unchanged. No clear event improves.

A*'s3 clear rescued frames occur in two rod episodes; its4 lost clear TP occur
in two episodes,3rod scene5 and1HEAD scene2. Its22 removed clear FP span9 negative
episodes/configurations. It detects previously missed rod scene0 at its first
observation, but rod scene5 is delayed0->0.5s. The new scene0 detection gives
18/18 core events; it does not negate the delay or4 lost true frames.

Across strict truth, A* rescues8 FN, loses12 TP, removes32 FP and introduces1
boundary FP. It gains3 events and loses4 boundary events, leaving24/30 versus
A25/30. The lost events are shallow scene0 enter, scene1 enter, scene2 exit and
scene3 enter; gains are rod scene0, shallow scene2 enter and scene5 enter.
The latter starts0.5s after strict entry. These costs must accompany the high F1.

All18 core episodes are left-censored; they begin with obstacles already present.
There is no core release opportunity. The six strict shallow-boundary exits give
A/A+head first-off delays[.25,0,0,unobserved,0,.5]s; A* is off at all six first
negative samples. These short intervals do not establish stable real-world
clearance or advance warning, especially when some preceding A* events are missed.

## Host algorithm latency

All288 new frames are timed after whole-path warmup on three consumed frames;
one resident instance serves every episode. No slow frame is removed.

| Path | Mean ms | p50 ms | p95 ms | Observed max ms |
| --- | ---: | ---: | ---: | ---: |
| A complete algorithm | 63.95 | 63.09 | 72.53 | 79.40 |
| A + public head | 66.15 | 65.48 | 75.43 | 81.60 |
| Added token frontend + head | 2.20 | 2.07 | 3.17 | 3.58 |
| Same-data A* comparison | 66.59 | 65.61 | 75.22 | 81.83 |

The head adds3.45% mean cost. Timings include RGB read/decode, original A's public
feature extraction and HGB plus the indicated head. A* reuses the same feature
vector; its conservative reported total also includes the original A head call,
but not the positive branch. These are measured component sums, not a standalone
A* optimized replay. Loading all three models takes1.72s, reported separately.
Sensor acquisition, transport, phone execution and user response are excluded.

## Runnable delivery and verification

[Runtime instructions](SINGLE_VERSION_USAGE_20260917.md) expose one resident
public-input path using `raw.jsonl` and RGB only. Zero usable ToF returns preserve
A; no alert does not certify clear space. The optional comparison output reports
A* independently. Bundle weights, thresholds and inference source were frozen
before capture dispatch. This is a host research runtime, not Android integration.

Fifteen focused checks pass for token construction, group isolation, missingness,
OR behavior, model identity, threshold precision and source geometry. Independent
model auditing reconstructs full-data normalization, all120 training orders,
pooled threshold candidates and model seals. A*'s six saved checkpoints replay
all1,152 OOF scores exactly. On288 consumed public inputs, original-A scores,
independently computed linear head scores/argmax slots and cached-feature A*
scores match bitwise. This consumed replay checks implementation, not transfer.

The new capture verifies319 file hashes, decodes all288 RGB images and matches
all source/native geometry (maximum coordinate error8.88e-16m). The sealed
confirmation passes independent count/threshold/onset accounting, separately
computed native geometry/5cm strata and exact288-frame public-forward replay.
The latter reuses the production inference class and lineage label builder;
it is not a wholly independent inference/label implementation. An independent
linear-head calculation is supplied by the consumed parity audit above.
The comparison figure PNG/SVG was rendered and visually inspected. A separate
read-only review confirmed group, event and frame costs and their interpretation.
`describe_single_cases.py` reproduces the post-seal activation/missing-ray
accounting without changing predictions. Documentation links pass. Staged
whitespace checking reports one trailing blank line in the frozen source
generator; its frozen bytes are preserved, with no other whitespace failure.

Retain `COMPONENT_OR_CHALLENGER / COMPONENT` intent for the runnable public head
and prior v2 Development gain: this confirmation tests false activation but has
no supported clear miss, so it cannot confirm or reject clear rescue transfer.
Retain A* as a `COMPONENT_OR_CHALLENGER / CHALLENGER` precision-oriented candidate
with the observed strong clear-F1 gain and strict event costs. Frozen A remains
the OR reference. No default-App replacement or A*+head combination is evaluated.
This comparison does not show that the public head outperforms a same-data HGB,
nor isolate data quantity, supervision, refitting or threshold selection as the
sole cause. The useful A* result belongs to its complete frozen recipe.

Artifacts are retained under
`artifacts.local/work/corridor-public-single-20260917/`: model/checkpoint and
optimizer states, fitting orders, calibration curves, model/data/source seals,
public predictions, native evaluator records, RGB, timing samples and receipts.
Registration remains pending at the existing `experiments/index.jsonl:303`
input-fingerprint mismatch; the supported-command failure is retained. No
shared ledger entry is edited or bypassed.

The supported inheritance command also reports `unknown terminal id`; intended
roles above remain pending structured registration (`inheritance.log` retained).
The capture ran once22:32:24--22:55:29 HKT with no timeout/restart. Controller
and worker retain raw data, logs and receipts. Worker raw resides at
`G:\DevWorkspace\BlindAssist\artifacts\work\corridor-public-single-20260917\capture-v1`;
local raw is `source/returned-v1/capture-v1` under the artifact root. Task-owned
processes, Zen port27032 and scheduled task are verified released. Only the
owned reproducible DDC was removed:455,074,796 logical bytes. Training/inference
processes exited; no service or paid allocation remains. Durable evidence and
weights stay available for reproduction; no extra capture or fitting follows.
