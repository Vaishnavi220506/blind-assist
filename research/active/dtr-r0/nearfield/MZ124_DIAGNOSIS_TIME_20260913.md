# MZ124 consumed spatial diagnosis and causal temporal contrasts

Date: 2026-09-13. Phase: `EXPLORE / CONSUMED_DEVELOPMENT`.

This bounded branch asks whether MZ123's errors primarily reflect unavailable
learned spatial evidence, a threshold/readout mismatch, or isolated temporal
dropouts. It does not reopen MZ123 as independent confirmation, train a model,
select a replacement threshold, or change MZ116. The user authorized trying the
brainstormed directions; this branch closes the frozen-output diagnosis and
simple temporal comparisons. Measurement-anchored correction and paired training
are separate MZ124 branches.

## Main findings

The three highlighted failure families differ. BODY shows a strong directional
spatial response that the `.58` alert readout fails to exploit. Rod/farwall shows
little consistent response to the paired change. The fully missed HEAD event
has no valid ToF returns and persistently low, wrongly localized model output,
despite intermittent Radar evidence used by MZ116. These observations narrow
the diagnosis; they do not identify a unique internal network cause or establish
a sensor information ceiling.

On all 288 consumed frames, no attainable global threshold on the unchanged
early core score simultaneously reaches MZ116's `TP >= 139` and `FP <= 116`.
This statement is stronger than the original `.58` failure but remains scoped
to this checkpoint, this scalar readout and these consumed samples. It does not
rule out other readouts, representations or training.

Neither fixed temporal rule preserves the desired recall/precision tradeoff.
Holding a previous positive helps isolated misses while increasing false
duration. Majority voting removes some false frames while losing more true
frames and increasing delay. No temporal arm recovers the full missed HEAD
event from the early alerts.

## Spatial diagnosis

Evaluator-native AABBs are intersected with the exact MZ120 45-cell contract.
The scorer records maxima separately over true core, wrong core, true off-core
and wrong off-core cells. These masks are evaluator-only; none feeds temporal
predictions. The independent NumPy copy of the grid is checked against the
original source's grid assignments by a focused test.

`mz123_suspended_head_pair0_in` is positive for all 12 frames:

| Diagnostic | Observed result |
| --- | --- |
| Valid ToF targets | 0 on all 12 frames |
| Radar-return frames / MZ116 alert frames | 10 / 10, the same frames |
| Early true core cell 21 probability | .13455–.21181 |
| Early maximum wrong core probability | .16441–.25476 |
| Maximum over the whole grid | Always wrong cell 24, .17098–.26190 |
| Early alerts at .58 | 0/12 |

Cell 21 is the true HEAD corridor interval at forward distance 1.4–2.6 m;
cell 24 is the HEAD corridor interval at 3.6–4.2 m, outside the near-alert core.
The true cell does not outrank the wrong core maximum in any frame. Thus this is
not a sequence of scores narrowly below `.58`, nor a correct high-probability
HEAD cell merely discarded by the core mask. RGB/attention interventions would
still be needed to separate feature, association and learned-score causes.
Zero ToF returns is an observed input condition, not proof that every sensor
lacks useful information.

For each matched frame pair, changed truth cells are evaluated by the sign of
their probability change. An occupied-to-empty cell should decrease, and an
empty-to-occupied cell should increase. Independent sensor noise remains in the
pairs, so this is a response diagnostic rather than a pure causal intervention.

| Family, three pairs each | Changed cells moving in correct direction | Median positive-minus-negative frame score, by pair | Both frame answers correct at .58 |
| --- | --- | --- | --- |
| Rod/farwall | 107/216 (49.54%) | +.00138 / −.00433 / +.00027 | 0/36 |
| Substantial BODY | 221/222 (99.55%) | +.19350 / +.23650 / +.18781 | 0/36 |

All 36 BODY positive scores lie above all 36 BODY negative scores in this sample
(minimum positive `.77656`, maximum negative `.69990`), yet both sides exceed
`.58`. This is a within-family diagnostic fact, not permission to use evaluator
family labels to set runtime thresholds. Rod positive/negative scores remain
heavily overlapping at roughly `.8–.86`.

The exhaustive diagnostic enumerates 290 attained score boundaries, including
all-positive and all-negative outputs. Minimum FP while retaining at least 139
TP is **131**; maximum TP while allowing at most 116 FP is **129**. No threshold
is selected or transferred. The original checkpoint and `.58` result are intact.

## Causal temporal replay

Six arms use the same ordered 24 complete episodes at 0.25-second cadence:
MZ116 or frozen early `.58`, each with instantaneous output, `hold1` (current OR
previous input alert), or `two_of_three` (at least two current/past alerts).
Missing startup votes are false. Every episode resets state. Hold uses input
alerts, never recursively held outputs. The prediction function accepts only
episode/time metadata and sealed alerts, with no truth, family, image or actor
information. Temporal predictions are written and hashed before evaluator
parsing in this run. A false output means no alert, not certified clear space.

| Arm | TP / FP / FN | Precision | Recall | Events | False seconds / segments | Maximum detected-event delay |
| --- | --- | --- | --- | --- | --- | --- |
| MZ116 instant | 139 / 116 / 5 | 54.51% | 96.53% | 15/15 | 29.00 / 27 | .25 s |
| MZ116 hold1 | 143 / 131 / 1 | 52.19% | 99.31% | 15/15 | 32.75 / 20 | .25 s |
| MZ116 two-of-three | 130 / 111 / 14 | 53.94% | 90.28% | 15/15 | 27.75 / 18 | .75 s |
| Early instant | 122 / 104 / 22 | 53.98% | 84.72% | 14/15 | 26.00 / 15 | .25 s, one missed event |
| Early hold1 | 126 / 109 / 18 | 53.62% | 87.50% | 14/15 | 27.25 / 14 | .25 s, one missed event |
| Early two-of-three | 112 / 94 / 32 | 54.37% | 77.78% | 13/15 | 23.50 / 11 | .50 s, two missed events |

Suspended HEAD TP/FP/FN are MZ116 `33/24/3`, hold `36/30/0`, majority
`33/25/3`; early `22/6/14`, hold `23/9/13`, majority `21/6/15`. Early retains
only 2/3 HEAD events in every temporal arm. Majority additionally loses a
boundary-stress event. All early rod and BODY pair answers remain 0/36 correct
on both sides under either temporal rule: majority drops each startup frame
on both positive and negative episodes, yielding no paired discrimination.

Delay is measured from a contiguous positive event's onset to its first true
alert. Missed events have `delay_s=null` plus observed-duration lower bounds;
the 12-frame HEAD miss has a 3.0-second sampled-duration lower bound. A finite
maximum among detected events must not hide misses. False duration is `.25 s`
per sampled false frame, including the final sample's nominal interval. Segment
counts reset across episodes and are not actual notification counts. Hold has
no same-object verification, and fewer segments can coexist with longer false
duration; the MZ116 hold result demonstrates that distinction directly.

An IMU-warped grid arm was **not tested**. The inherited grid projections already
use accumulated yaw in fixed episode axes; the grid is metric, and unknown
within-cell position plus unavailable metric translation makes an additional
yaw-only transport interpretation unjustified here. These fixed alert-history
rules test smoothing only, not motion-based spatial disambiguation or a learned
temporal model.

## Reproduction and evidence

Code: [mz124_diagnosis_time.py](mz124_diagnosis_time.py).
Tests: [test_mz124_diagnosis_time.py](test_mz124_diagnosis_time.py).

```powershell
python -m unittest discover -s research/active/dtr-r0/nearfield -p test_mz124_diagnosis_time.py
python research/active/dtr-r0/nearfield/mz124_diagnosis_time.py --source artifacts.local/work/mz123-frozen-early-20260913/returned-v1 --output artifacts.local/work/mz124-all-directions-20260913/diagnosis-time-v1
```

Use a compatible Python environment with NumPy. The output path must be new;
use a different revision for a reproduction.
Five focused tests pass: exact grid contract, nonrecursive hold/reset, majority
startup/reset, future-prefix invariance, and event/duration/missed-delay handling.
The original frame truth, early scores and baseline alerts are reproduced exactly.

Payload root:
`artifacts.local/work/mz124-all-directions-20260913/diagnosis-time-v1/`.
It contains `summary.json`, `frame-diagnosis.json`, `episode-diagnosis.json`,
`pair-diagnosis.json`, `threshold-curve.json`, `temporal-results.json`, sealed
`temporal-predictions.json`, `backend.json` and `completion.json`.
Each consumed capture input and frozen prediction file is hash-verified and its
identity saved in the summary. Local returned RGB is partial; this branch reads
no RGB and does not claim a complete image hash audit. An initial full-receipt
check stopped on an absent unconsumed RGB; the corrected check verifies exactly
the consumed raw/evaluator/spec inputs and original prediction seal.

Actual execution: NumPy 2.4.2 on CPU, `TASK_NOT_GPU_SUITABLE` for small scalar
metadata/scoring; 0.840 seconds total measured run. The shared backend selector
records the actual CPU device and scalar-scoring probe. No inference, persistent
process, capture, paid allocation or accelerator resource was started.

Decision for this branch: retain MZ116, preserve these consumed diagnostic and
negative temporal comparisons, and use the differentiated BODY/rod/HEAD evidence
to interpret the separately authorized MZ124 correction/training arms. No fresh
validation, runtime promotion, hardware or safety claim follows. Structured
registration and inheritance are owned by the root task and are not asserted
complete by this branch.
