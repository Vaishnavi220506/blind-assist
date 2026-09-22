# One calibrated ToF corridor readout on consumed Development96

2026-09-20. COMPLETE. EXPLORE, one fixed score and one operating-point calibration.

**The single calibrated readout retains all30 baseline TP and reduces FP19->15
on the same consumed96 frames.** Interior events stay4/5, all events5/6, and all
five detected events retain their1.2s first in-event alert. This is an in-sample
Development gain for one fixed score, not held-out transfer or hardware evidence.

## Measured result

| Metric, same96 frames | Possible-overlap baseline | Calibrated score |
| --- | ---: | ---: |
| TP / FP / FN | 30 / 19 / 6 | 30 / 15 / 6 |
| Recall | 83.33% | 83.33% |
| Precision | 61.22% | 66.67% |
| FPR, all60 known-negative frames | 31.67% | 25.00% |
| Interior events | 4/5 | 4/5 |
| All events, including boundary-only | 5/6 | 5/6 |
| False-alert segments | 6 | 4 |
| False-alert sampled duration | 3.8s | 3.0s |
| Prediction-UNKNOWN frames | 86 | 86 |
| Ambiguous alert frames | 39 | 35 |
| Negative abstentions / TN | 41 / 0 | 45 / 0 |

All30 original TP identities are preserved; no FN is rescued and no new FP is
introduced. Interior88 frames change23/19/5 to23/15/5, recall82.14%, precision
54.76%->60.53%. Boundary8 frames remain7/0/1, recall87.50%, precision100%.
The6 missed frames remain the small same-zone object s06; silence is still FN.
There are10 definite-supported alerts in both arms. UNKNOWN positive/negative
counts stay26/60. Pixel IoU is not defined for this scalar task.

The selected inclusive threshold is **0.007085703945147101**, bound by f0042
(s03 boundary-contact bar, frame6). It preserves20 ambiguous TP plus10 definite
TP. This is the maximal cutoff for the frozen score with complete TP retention;
raising it loses that required boundary TP. Its low numerical value reflects
the chosen geometric measure, not a small or large physical collision probability.

| Strict FP geometry subtype | Baseline | Candidate |
| --- | ---: | ---: |
| Distance only | 12 | 9 |
| Lateral only | 6 | 6 |
| Both distance and lateral | 1 | 0 |

Removed frames are f0016 (s01,0.8s) and f0039..f0041 (s03,0.6..1.0s).
The first is a combined outside/depth case; the latter three are pre-entry
depth-only warnings. These remain FP in the baseline; no label was changed.
Per-clip FP changes are s01:7->6 and s03:3->0; s02:4,s04:4,s05:1 and the other
zero counts are unchanged. **The score has not solved the six pure lateral
false alerts, nor recovered the fifth interior event.**

All five detected events s00/s02/s03/s04/s05 still alert at entry1.2s; s03 is
boundary-only and is included in the retention check. First-within-clip alerts
change s01:0.8->1.2s (both false) and s03:0.6->1.2s (false pre-entry to true
entry). Thus event onset is retained, while clip-first timestamps do move later
where false alerts were removed. s02/s04/s05 still have earlier false warnings;
zero event-relative delay does not establish correct onset discrimination.
There are no left-censored first alerts in this cohort.

## Evidence, limits and disposition

The saved input/decision/anchor baseline replays exactly over all96 frames.
All4,912 valid frame-zone returns, including102 measured-proxy returns<3m,
retain their original supports and intervals. The23 frames with native sampled
corridor contributors remain alerting; the possible zones contain5,282 such
contributors across80 frame-zone instances. The four withheld frames contain
zero native corridor contributors. This is exact sampled winning-bin evidence,
not full unsampled surfaces, ownership certainty or physical ToF measurements.

Six synthetic tests pass, including an independent dense-integral comparison,
threshold tie/retention behavior, definite bypass and missing-observation UNKNOWN.
The one96-frame CPU scoring pass averages2.331ms, P95 2.805ms, including the
baseline and geometric score but excluding disk IO, capture and transmission.
These are desktop scalar-stage timings, not end-to-end or target-device results.

Protocol SHA256 is
`0d780c803bb63a64fb9a07fd2a93f89dbec65dc2ebbedf75b9acc3d787426134`.
Observable scores were sealed before Development-label calibration; selected
predictions were sealed before native auditing. Both calibration and reporting
use the same96 consumed frames. This establishes an in-sample working point,
not transfer, independent validation or a calibrated sensor likelihood.

One pre-freeze syntax typo was fixed before any scoring. One evaluation-only
repair handled the parent wrapper's extra nested `boundary_detected` event key.
The original runner, failed evaluator log and exact hash-bound repair receipt
are retained; score formula, sealed scores, cutoff, predictions, labels and
metric definitions were unchanged. Scoring and cutoff selection each ran once.

Retain as `COMPONENT_OR_CHALLENGER / COMPONENT` for consumed in-sample readout
only. The original baseline and inputs stay immutable. This authorized single
comparison is complete; no RGB, new capture, additional score, threshold repair,
training, runtime promotion or successor is started.

Global registration is still blocked by the pre-existing ledger303 fingerprint
mismatch; terminal inheritance reports unknown terminal. Keep the actual command
receipts and structured `local-disposition.json`; do not claim global metadata
completion. No persistent task process, device session or paid resource remains.
The evidence payload is retained for reproduction, with no disposable datasets
or model intermediates created.

## Question and frozen scope

Can a continuous measure of existing support overlap reduce the nominal45x45
baseline's19 FP while retaining all30 original TP,4/5 interior events,5/6 total
events and their first in-event alerts? Prior work changed the sensor footprint
and exposed new target returns but also increased false alerts. This comparison
changes only the decision readout of the existing96 sealed observation vectors.

Use all8 clips as **consumed, in-sample Development**, with no held-out split or
generalization claim. Keep the original corridor, timestamps, strict contact
labels, boundary stratum, all60 negative frames and all36 positive frames.
Inputs are `artifacts.local/work/ba-tof-fov45-20260920/`; original labels and
native audit evidence are from `ba-camera-corridor-20260919/`.

## One score, one cutoff

Keep every valid return and the exact original interval
`[max(.1,z-(.1+3*(.01+.02*z))), z+(.1+3*(.01+.02*z))]`.
At each axial depth Z, compute the fraction of the **entire zone ray-slope
rectangle** that lies inside the fixed X/Y corridor. Integrate that fraction
over the part of the original interval inside0.3..3m, and divide by the full
original interval length. Use an exact piecewise integral; each piece has form
`c0+c1/Z+c2/Z^2`. The frame score is the largest score among possible-hit zones.
Record depth fraction and conditional angular fraction separately for diagnosis.

Uniform measure over ray slopes and axial depth is a fixed geometric ranking
convention, **not a physical uncertainty distribution, confidence or collision
probability**. It uses the unchanged proxy's error-envelope scale; calibration
here means selecting an alert operating point on Development labels, not fitting
a hardware error model. No centre ray, narrowed interval, new range or RGB is used.

Always alert on original definite support. For ambiguous baseline alerts, use
`score >= threshold`. Choose the maximum threshold that preserves every original
baseline TP: the minimum score among required ambiguous TP. This gives the fewest
FP possible for this one monotone score under exact TP retention; no grid or
iterative threshold search is needed. Equal scores cannot be split. If no such
point reduces FP, stop this recipe, without claiming all ToF features exhausted.

Seal observable-only scores before opening labels for cutoff selection. Seal
selected predictions before native-contributor auditing. The complete96 frames
are reused for calibration and reporting; sealing establishes dataflow integrity,
not independent testing. All original anchors, intervals, validity and source
identities remain intact. Withheld alerts are UNKNOWN; they are not free-space
certificates. Any emitted alert is counted, and silent positive frames are FN.

## Acceptance, accounting and stop

Require exact retention of the original30 TP identities, fewer than19 FP,
unchanged detected events and no later first in-event alert. Preserve every
definite support alert and audit possible-zone native corridor contributors
under every withheld alert. Report all/interior/boundary TP/FP/FN, recall,
precision,FPR over all60 negatives, abstentions, UNKNOWN, false segments/duration,
per-clip and paired-group counts. Pixel IoU does not apply to scalar alerts.

Report clip-first alerts separately: removing a pre-entry FP can move the first
clip alert later without delaying a true event alert. Record left-censoring and
preexisting alerts at entry. Split strict FP by evaluator geometry into depth,
lateral, vertical or combined separation; no subtype becomes a correct alert.

One score, one cutoff,96 reused frames; zero captures, new scalar simulation,
models, fitting updates, RGB calls or automatic successor. A successful result
retains an in-sample component only. A failed joint target closes this exact
score/cutoff recipe for this role. Implementation faults may be repaired with
receipts without changing the hypothesis or scoring formula.

Payload: `artifacts.local/work/ba-tof-corridor-calibration-20260920/`.
Implementation: [score and calibration](tof_corridor_calibration.py),
[staged runner](run_tof_corridor_calibration.py),
[synthetic checks](test_tof_corridor_calibration.py).

CPU scalar geometry only (`TASK_NOT_GPU_SUITABLE`); measured desktop scoring
time excludes acquisition and IO and does not establish target-device latency.
Preserve protocol/code/input hashes, immutable stage seals, observations by
reference, per-frame predictions, native audit, result and failure receipts.
No persistent process or paid allocation is required.
