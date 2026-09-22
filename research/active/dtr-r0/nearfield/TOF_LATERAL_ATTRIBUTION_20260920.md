# One local RGB lateral-attribution mechanism pilot

2026-09-20. COMPLETE. EXPLORE, existing96 consumed controlled frames only.

**The5,915-parameter RGB head reaches30TP/10FP/6FN, retaining every baseline
TP, event onset and native corridor contributor.** The matched noRGB head
reaches30/13/6; the simple Otsu region method leaves30/15/6 unchanged. This is
positive **in-sample mechanism/fitting evidence**, not source-disjoint transfer.

## Measured four-arm comparison

| Same96 frames | A calibrated ToF | B Otsu region | C local RGB head | C noRGB control |
| --- | ---: | ---: | ---: | ---: |
| TP / FP / FN | 30 / 15 / 6 | 30 / 15 / 6 | **30 / 10 / 6** | 30 / 13 / 6 |
| Recall | 83.33% | 83.33% | 83.33% | 83.33% |
| Precision | 66.67% | 66.67% | **75.00%** | 69.77% |
| FPR over all60 known negatives | 25.00% | 25.00% | **16.67%** | 21.67% |
| Interior events | 4/5 | 4/5 | 4/5 | 4/5 |
| All events, including contact | 5/6 | 5/6 | 5/6 | 5/6 |
| False-alert segments | 4 | 4 | **4** | 6 |
| False-alert sampled duration | 3.0s | 3.0s | **2.0s** | 2.6s |
| UNKNOWN / TN | 86 / 0 | 86 / 0 | 86 / 0 | 86 / 0 |
| Pure lateral FP | 6 | 6 | **1** | 4 |
| Depth-only FP | 9 | 9 | 9 | 9 |

C removes f0019..f0023 (s01 frames7..11,1.4..2.2s): all5 eligible lateral FP.
The remaining f0018 (1.2s) has a3.007462m return and stays outside the strict
near-return intervention. No eligibility threshold was relaxed to remove it.
The missing same-zone s06 event and its6 FN remain untouched.

The RGB head removes5/15 calibrated FP (33.33%). Of that total, a geometry-only
learned readout can already remove2 under the same recipe. The **increment over
the matched noRGB fit is3 FP**, not5. NoRGB removes only f0020 and f0022; the
interleaved remaining s01 warnings fragment its false segment, raising overall
segments4->6 and failing the unchanged-segment guard despite fewer FP frames.
C satisfies the full alert/support guard; its4 segments persist because s01's
first false frame remains. No new FP, lost TP or rescued FN occurs in any arm.

All five originally detected events still first alert at entry1.2s. All
first-within-clip alert times also remain A-identical: s00/s01/s03 at1.2s,
s02/s04 at0.4s, s05 at0.8s, none for s06/s07. The early false warnings in
s02/s04/s05 remain; zero event delay is not proof of accurate onset discrimination.
Interior88-frame TP/FP/FN: A/B23/15/5, C23/10/5, noRGB23/13/5. Boundary8 remains
7/0/1 for every arm. Negative abstentions rise45->50 for C, never converted to TN.

## What the local outputs establish

All60 B crops produce usable appearance hypotheses, but only4/13 true-OUTSIDE
zone instances are classified OUTSIDE; the other9 remain CROSSING. Other zones
continue voting in every affected frame, so those4 zone changes yield0 final
false-alert removals. The whole-target classes INSIDE35/CROSSING12 are never
classified OUTSIDE by B. Close this exact minority-intensity recipe as a
negative alert control; this is not evidence against all lightweight RGB geometry.

C fits all60 supervised zone labels correctly and suppresses exactly13 OUTSIDE
votes; noRGB fits55/60, suppresses10 OUTSIDE votes and leaves3 OUTSIDE as CROSSING.
Its remaining2 classification errors are INSIDE->CROSSING and do not cancel alerts.
Every suppressed zone in all three arms has **zero native corridor contributors**;
there is no concealed contributor loss in frames that retain another alert vote.
All4,912 original valid anchors and full intervals remain immutable, and all
definite-support votes remain active. Retaining an observation is distinct from
retaining its alert vote; both are reported explicitly.

The matched comparison supports **an additional usable RGB cue for this fixed
small-head fit on these inputs**. It does not prove correct echo ownership in
new scenes, establish a geometry-only information ceiling, or show which image
feature caused the effect. The single material/background and single OUTSIDE
physical arrangement can support memorization. C learned from evaluator-derived
full-target extent labels on the same samples being reported; it is not a held-out
method result or a transferable3-class attribution system.

## Execution, validation and disposition

The immutable protocol SHA256 is
`ab6e17b16a03fc5285cf6737b2ed430a554ca6455d1a64a801712c634152c9b1`.
There was one observable preparation, two prescribed fits of400 updates each,
one final prediction pass and one evaluation. Public RGB/geometry and B outputs
were sealed before supervision materialization; final predictions were sealed
before native auditing. These boundaries establish dataflow, not independence.

Independent saved-output recount passes: input/model/decision seals, the fixed
0.95 cutoff, matched initialization/budget,30TP identities, event times, UNKNOWN,
all suppressed native contributors and noRGB's4->6 segment cost are verified.
The independent audit performed no model inference, fitting or alternative run.

Six core synthetic tests and a CUDA forward/backward check passed. Before freeze,
the CUDA determinism check exposed unsupported adaptive-pool backward; equivalent
fixed6x6 pooling produces the same4x4 geometry and passed. Before any optimizer
update, training launch exposed a missing required CPU-probe argument in the
backend-selection API. The exact pre-fit repair, original runner/evaluator and
failed log are retained in `backend-repair.json`; protocol, public/B outputs,
labels, architecture, seed, optimizer, update counts and cutoffs were unchanged.

The equivalent60-crop forward/backward probe measured CPU median25.396ms versus
CUDA1.638ms and selected the RTX5060 Laptop GPU. Actual400-update fits took1.155s
(RGB) and0.751s (noRGB). Single final60-crop batch inference took10.020ms and
0.650ms respectively; these single desktop samples are not stable timing
comparisons or end-to-end/edge-device latency evidence. Crop/Otsu uses CPU OpenCV.

Retain C as `COMPONENT_OR_CHALLENGER / COMPONENT`, explicitly in-sample only.
B is a `NEGATIVE_CONTROL` for this exact alert role. Retain noRGB as the necessary
matched control with disclosed fragmentation, not as an improved alert system.
A remains the frozen baseline for future comparisons. No further fit, cutoff,
frame split, capture, missing-event rescue or successor ran after this result.

Global registration remains blocked by the existing ledger303 fingerprint error;
inheritance still reports unknown terminal. Preserve actual command receipts and
`local-disposition.json` without rewriting global metadata. Durable inputs,
weights, labels, predictions, native audits and receipts remain in the payload.
No persistent process, sensor session or paid allocation remains active.

## Frozen question and contrasts

Can local RGB distinguish which side of a corridor boundary an already returned
near surface occupies, and remove lateral false alerts without losing the
calibrated baseline's30 TP, five detected events or native corridor evidence?
Keep the [calibrated A baseline](TOF_CORRIDOR_CALIBRATION_20260920.md) at30/15/6
and threshold0.007085703945147101, with no range/score/threshold changes.

Compare A, one B simple RGB geometry recipe, and one C tiny local classifier.
C has one matched noRGB input control using the same initial weights, architecture,
labels and training budget. This is needed to distinguish RGB contribution from
memorizing the already available zone/range/corridor geometry; it is not an
architecture search. No fifth-event rescue, dense depth, NFO, temporal fusion,
sensor simulation, new capture or hardware acquisition is in scope.

Prior MZ137/139 edge/shape negatives tested different body-frame inputs,
association and readouts. They caution against unsupported narrowing; they do not
settle this nominal45 camera-frame question. Preserve every raw support and audit
suppressed zone contributions even when another zone keeps the frame alerting.

## Eligibility and sample limitations established before fitting

Require a valid existing scalar0.1<=z<3m, an original possible and nondefinite
zone, and lateral ambiguity of its **entire original distance interval** and
zone angular footprint at X=+-0.3m. No RGB edge creates a missing near range.
Near eligibility produces60 zone instances over28 frames. All96 frames remain
in final alert denominators; noneligible evidence retains A exactly.

The6 pure lateral FP are s01 frames6..11. Frame6/f0018 has only a3.007462m
possible return and is not eligible. The remaining5 frames have13 eligible
near zone votes,3/2/3/1/4 per frame, without residual votes that block rejection.
Thus this fixed intervention can address at most5 of the6 lateral FP.

For supervision only, use authenticated **whole target X bounds**, conditional
on pure target winning-bin ownership. Exact contact is CROSSING, never OUTSIDE.
Mixed/non-target return labels are UNKNOWN and excluded from supervised fitting.
Native sampled point OUTSIDE is insufficient: a horizontal bar can have outside
sampled returns while its continuous surface contacts the corridor.

The fixed eligible cohort has INSIDE35, OUTSIDE13, CROSSING12, all pure. By pair:
g1 has6/13/0, g2 has0/0/12, g3 has29/0/0, g4 has0/0/0. Leaving g1 out removes
every OUTSIDE training example; leaving g2 out removes every CROSSING example.
Therefore C is explicitly **in-sample consumed fitting**, not a train/holdout
evaluation. Do not create a misleading random-frame split. The common material,
background and single negative physical arrangement further limit interpretation.

## One implementation per mechanism

B uses a3x3-zone native RGB crop. Grayscale Otsu partitions two intensity groups;
the smaller group is an appearance foreground hypothesis, either polarity.
Require mean contrast>=12/255 and at least2 selected pixels inside the queried
zone. Use all selected pixels' horizontal extent with one native pixel of padding,
clipped to the zone. Combine that extent with the unchanged full distance interval.
Only a completely lateral-OUTSIDE hypothesis removes that zone's alert vote.
No usable appearance region returns UNKNOWN/full-zone fallback. Minority intensity
is an explicit prior, not proven return ownership or complete surface coverage.

C takes a48x48 crop with RGB plus5 geometry maps: zone mask, X at measured Z,
X at both original interval bounds, and Y at measured Z. Coordinate maps describe
the sensor hypothesis, not inferred dense depth. Conv8->8->16,6x6 pooling to4x4,
and linear256->16->3 yield INSIDE/OUTSIDE/CROSSING scores. Use AdamW lr0.003,
weight_decay0.0001, class-balanced cross entropy,400 full-batch updates, seed20260920.
Use the final checkpoint only. C_no_rgb zeroes RGB channels and repeats exactly
the same fixed fit from identical initial weights. No other seed, schedule, head,
loss, checkpoint or cutoff is selected after outcomes.

Only OUTSIDE softmax score>=0.95 suppresses a C zone vote; the value is an
uncalibrated classifier score, not physical confidence. Retain every definite
support vote and every noneligible/residual vote. Recompute the original max-score
decision with the original calibrated threshold and the surviving votes.
Raw anchors/intervals remain unchanged. Suppression is a decision hypothesis,
not a certified removal of possible physical occupancy; silence remains UNKNOWN.

## Sealing, acceptance and stop

Freeze code/input/protocol hashes first. Seal public RGB/geometry inputs and B
outputs before materializing supervision. Train exactly the two matched C fits,
seal their final weights/scores and final96 predictions before native auditing.
Training labels are disclosed evaluator-derived synthetic supervision; the
inference functions never receive labels, actor geometry or contributor points.
This dataflow separation does not make in-sample results independent evidence.

Report each arm's TP/FP/FN, precision, recall,FPR over all60 negatives, interior
and boundary strata,4/5 interior and5/6 total baseline events, event and clip-first
timing, false segments/duration, UNKNOWN, eligibility/fallback counts and lateral
FP identities. Preserve all30 baseline TP identities, definite supports and
first in-event alerts. Audit every suppressed zone for removed native corridor
contributors, including changes hidden by another still-alerting zone.

Reducing4 or5 eligible lateral FP with these guards is the aspirational target;
report smaller changes honestly. C must beat matched C_no_rgb to support an
incremental RGB finding. Successful in-sample fits alone do not prove transferable
attribution; a failed B/C does not prove single-frame RGB information is absent.
Keep any useful component only in its measured role and end this single round.

Payload: `artifacts.local/work/ba-tof-lateral-attribution-20260920/`.
Implementation: [local mechanisms](tof_lateral_core.py), [runner](run_tof_lateral.py),
[evaluator](evaluate_tof_lateral.py), [synthetic checks](test_tof_lateral_core.py).
Training uses the available CUDA GPU with actual backend receipts; B crop processing
uses installed CPU OpenCV (`GPU_BACKEND_UNAVAILABLE` for this implementation).
Record training and batch inference time without claiming target-device latency.
Preserve weights, protocol, source hashes, inputs, per-zone/per-frame decisions,
native audits and receipts; release task processes after this bounded run.
