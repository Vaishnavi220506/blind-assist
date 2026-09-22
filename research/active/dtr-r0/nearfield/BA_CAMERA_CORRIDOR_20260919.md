# One frozen camera-forward corridor comparison

Status: `EXACT_CORRIDOR_READOUT_GAIN_NOT_MET`.

**The frozen Depth Pro + global ToF correction produces no alert in any of the
96 frames and misses all five interior intrusion events.** Retained NFO detects
one of five; raw ToF detects two of five. Zero false alerts from the candidate
comes with zero detections, so the predeclared retain gate fails. Close this
exact corridor replacement on this cohort as `NEGATIVE_CONTROL`. This result
does not reject all spatial representations or all camera-corridor approaches.

## Results

Truth covers all 96 frames: 36 positive, 60 negative; eight positive frames are
within the boundary band. There are six positive events, five containing interior
frames. Three events contain boundary frames, overlapping the interior-event set;
these strata must not be added as if disjoint event counts.

| Method | Interior events | All events | Frame TP / FP / FN | Precision / recall | False-alert segments / sampled duration | UNKNOWN: all / positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw ToF | 2/5 | 3/6 | 12 / 5 / 24 | 70.59% / 33.33% | 2 / 1.0 s | 96 / 36 |
| Retained NFO | 1/5 | 2/6 | 7 / 5 / 29 | 58.33% / 19.44% | 3 / 1.0 s | 1 / 1 |
| Depth Pro + global ToF | 0/5 | 0/6 | 0 / 0 / 36 | undefined / 0% | 0 / 0 s | 0 / 0 |

Known-negative frame false-alert rates are 5/60 (8.33%), 5/60 (8.33%) and 0/60.
Boundary-frame TP/FN are 1/7, 1/7 and 0/8; boundary-event detection is 1/3, 1/3,
0/3. All 17 raw ToF alerts are spatially ambiguous, hence UNKNOWN. Its other
79 frames abstain (24 positive, 55 negative); none becomes a TN. NFO's single
UNKNOWN is an ambiguous positive alert. Model-derived non-UNKNOWN remains an
uncalibrated point/bin readout, not a certainty or free-space guarantee.

| Clip | Raw ToF: first positive hit relative to entry | NFO | Depth Pro + global |
| --- | --- | --- | --- |
| G1 center cylinder | Miss | Miss | Miss |
| G1 right cylinder (negative) | 1 false frame | 1 false frame | No alert |
| G2 intrusion bar | +0.2 s | Miss | Miss |
| G2 exact grazing contact | +0.8 s, boundary only | +0.8 s, boundary only | Miss |
| G3 large plate | 0 s, already alerting before entry | 0 s, already alerting before entry | Miss |
| G3 thin plate | Miss | Miss | Miss |
| G4 inside same-zone object | Miss | Miss | Miss |
| G4 outside same-zone object (negative) | No alert / UNKNOWN | No alert | No alert |

All positive events start at nominal t=1.2 s. Raw ToF's large-plate alert starts
at t=0.4 s; NFO has an isolated false alert at t=0.2 s and a continuing alert from
t=0.6 s. Their zero entry-relative delay therefore does not establish timely
onset discrimination. The small same-zone pair yields no inside detection, so
there is no evidence of successful inside/outside disambiguation.

Relative to raw ToF, NFO loses five positive alert frames and rescues none;
it removes one false frame and adds one. Depth Pro loses all 12 raw positive
alerts, rescues none and removes all five false frames. No fallback union was
added after seeing this result. The retain gate fails its required increase in
interior-event detections, while its four other checks pass.

## Why the candidate never alerts

A posthoc diagnostic of the already sealed arrays finds **zero corridor pixels
in all 96 calibrated predictions**, not merely fewer than the four-pixel component
threshold. Across the 36 positive frames, all 7,693 sampled visible target pixels
are beyond 3 m in both bare and calibrated Depth Pro. Bare Depth Pro is inspected
here only to locate the failure; it is not an added scored competitor or tuned arm.

| Positive target, pooled sampled pixels | Native GT median | Bare Depth Pro median | Global-corrected median |
| --- | ---: | ---: | ---: |
| Center thin cylinder | 2.687 m | 4.514 m | 4.904 m |
| Intruding horizontal bar | 2.660 m | 5.582 m | 6.563 m |
| Large plate | 2.670 m | 5.117 m | 5.529 m |
| Thin plate | 2.670 m | 4.414 m | 4.846 m |
| Inside same-zone small object | 2.680 m | 5.887 m | 6.528 m |
| Exact grazing bar, boundary only | 2.660 m | 5.547 m | 6.455 m |

Positive-frame global scales range from 1.035 to 1.253: they move the already
overestimated target distances farther away in this cohort. This does not undo
the calibration gain previously measured on the separate Hypersim cohort, but
that gain did not transfer to these targets. All 30 positive frames in the five
interior-containing clips have at least four connected native-GT corridor pixels
on the same sampling grid; the smallest G4 support is seven. Exact-contact G2 is
different: primitive contact is positive even when pixel-center rays miss it,
and only one of its six boundary frames has four sampled native corridor pixels.

The diagnostic checks all 96 RGB/native pairs, exact low-RGB resizing, point
sampling, saved scaling and corridor masks. The 268.5119 px focal length matches
640 px width / 100° horizontal FOV and the official model's W/f convention.
All 96 target traces project onto their native target masks; native optical-depth
versus trace error is at most 1.233 mm. No obvious unit, focal-length or sampling
wiring error was found. These checks locate a distance-prediction failure; they
do not distinguish domain shift from target representation/background mixing or
prove the model's internal cause. No saved depth or decision was modified.

## Question and frozen scope

Does the retained Depth Pro model with the existing global ToF scale correction
detect more camera-corridor intrusion events than retained NFO without increasing
false alerts or positive UNKNOWN? This is a new controlled Development task under
the [camera-forward contract](CAMERA_FORWARD_CONTRACT_20260919.md). The original
near2m failures and the [rejected layer recipe](BA_NFO_DEPTHPRO_ECHO_20260919.md)
remain unchanged. There is no new training, model selection or threshold search.

All eight clips / four pairs / 96 samples form one cohort. The camera is level,
1.7 m above the road. In camera coordinates (X right, Y down, Z optical forward),
the closed volume is X [-0.3, 0.3], Y [-0.2, 0.9], Z [0.3, 3.0] metres. The
vertical band corresponds to world height 0.8–1.9 m. A 2 cm band is reported
separately; exact surface contact remains positive truth. Each clip has 12 posed
views, 0.1 m apart, with nominal 0.2 s sample spacing. These are posed trajectories,
not a real-time video, dynamic-obstacle trial or measured reaction latency.

| Pair | Frozen contrast |
| --- | --- |
| G1 | 4 cm vertical cylinder centered in the corridor versus offset 40 cm right |
| G2 | 60 cm horizontal bar penetrating the corridor versus exactly grazing its right edge |
| G3 | 40 cm wide plate versus 4 cm wide plate, both centered |
| G4 | Small object inside versus outside, both wholly within the same ToF zone |

The same Willow map, camera path, background wall and target material are used.
Native geometry, depth and actor traces belong to source construction/evaluation.
Inference sees opaque sample identities, RGB, calibration and simulated ToF only.
The existing single-return sensor proxy uses identical random seed identities
within each pair; actual returned ranges are not forced equal. It does not model
complete VL53L8CX physics or provide measured hardware evidence.

## Readouts and decision rule

- Raw ToF retains the full zone angular extent and range band
  ±[0.1 + 3 × (0.01 + 0.02z)] m. Any possible intersection alerts; angularly
  ambiguous alerts remain UNKNOWN. Missing/outside-only returns do not certify
  free space.
- NFO retains its original 0.081 cumulative cutoff and ordinal depth intervals;
  intervals are never replaced with invented metric depths. Positive-length
  interval/volume overlap creates possible support. The open-ended far bin's
  singleton at exactly 3 m does not trigger an all-image alert.
- Depth Pro uses the retained official checkpoint, native 640×360 RGB and
  calibrated focal length, followed by the exact prior global echo-scale recipe.
  Its metric point predictions are intersected with the corridor. A 2 cm predicted
  boundary band represents geometry ambiguity, not calibrated model confidence.
- Dense readouts require four four-connected possible pixels on the shared
  256×192 grid. With fewer than four definite pixels, an alert remains ambiguous.
  No debounce, late ToF union or posthoc threshold selection is added.

All ambiguous alerts count as TP or FP. Unalerted positive UNKNOWN counts as a
miss; negative UNKNOWN is not TN. NO_SUPPORTED_HIT is a prediction readout, not a
free-space certificate. Metrics retain all frames, event intervals, false-alert
segments, inclusive sampled durations, boundary strata and first detection relative
to entry. Preexisting alerts at entry are disclosed separately.

The predeclared retain gate requires **strictly more detected interior events
than NFO**, no more full-cohort false-alert segments or sampled duration, no more
positive UNKNOWN frames, and identical truth/event denominators. Raw ToF remains
a separately reported comparator. Neither pixel IoU nor raw-only superiority
passes this gate.

## Source repairs and evidence boundary

The protocol and scene specification were sealed before inference. Capture attempt
1 failed before frame 1 on an obsolete Unreal Python mesh accessor. Attempt 2
captured 96 frames but was rejected before any model scoring: background textures
were still loading, visibly changing the same camera view. All rejected frames and
logs are retained. A mechanical repair added the existing native capture-readiness
helper and explicitly loaded its verified plugin; geometry, assets, warmup counts,
readouts and the 96-frame budget were unchanged.

The final acquisition has 96 per-view READY receipts with zero pending asset,
shader and render-asset counts, completed streaming updates, and unchanged map
hash. Target traces, actual bounds/poses, visibility, competing intrusions and
the G4 common-zone footprint are audited. No frame is discarded for method quality.
The failed source attempts are not algorithm negatives or additional evaluation
cohorts. The observations are sealed before prediction, and all predictions are
sealed before evaluator truth access.

Depth Pro receives native detail while NFO receives 256×192 resized RGB with its
inherited aspect-ratio distortion. NFO is trained on Hypersim and evaluated here
on Unreal. This comparison therefore cannot isolate architecture from resolution
or domain effects. The small controlled source supplies exploratory evidence only,
not natural-distribution, device, deployment or safety validation.

## Reproduction and retained artifacts

Twelve focused tests pass (five geometry/readout and seven event/UNKNOWN tests).
The initial unittest invocation used the wrong working directory and could not
import the local modules; its failed log is retained, followed by the successful
correct-directory invocation. The evaluator checks all saved payload hashes,
replays each readout from saved arrays, and reconstructs source admission for all
96 frames. Both fixed figures were rendered and visually inspected: every timeline
and predeclared frames 0, 6 and 11 from all eight clips are shown.

An independent saved-output audit passes for all three arms, eight clips and four
pairs without importing the experiment's metric/evaluator/readout functions.
Frame counts, event memberships, false-alert segments, UNKNOWN and preexisting
alerts all reproduce. All 96 calibrated depth maps exactly equal the specified
native point sampling multiplied by the saved global scale (maximum error zero);
scales match the exponential median of the saved anchor log-scales. The audit
reproduces the failed retain gate. It reuses sealed truth labels; source geometry
is audited by the separate evaluator, not independently relabelled by this recount.

Actual inference used CUDA on an RTX 5060 Laptop GPU, Torch 2.11.0+cu130: NFO
fp32 and official Depth Pro fp16, with CPU scalar/graph readout. The sealed loop
took 100.609 s including postprocessing and compressed output, excluding model
startup. Synchronized model-call totals are NFO 0.836 s (8.710 ms/frame) and
Depth Pro 85.646 s (892.146 ms/frame). Peak allocated CUDA memory is
3,977,107,968 bytes. There are exactly 192 model calls and zero training updates.
This is host batch-one execution, not end-to-end or phone latency. The prior
equivalent model/device CPU/GPU placement receipt was reused.

Final capture map/actors and all task-owned Unreal descendants were released;
the inference and evaluation processes exited. No paid allocation or background
model worker is retained. Source failures, sealed evidence and diagnostic scripts
are retained for audit; shared model weights and runtime remain shared assets.

Local structured disposition closes the exact candidate as `NEGATIVE_CONTROL`;
retained NFO remains the inherited comparator, without claiming that its 1/5
event result is adequate. Raw ToF remains an ambiguous sensor comparator. The
global knowledge registration is pending: the actual command failed on the
pre-existing `experiments/index.jsonl:303` fingerprint mismatch; terminal
inheritance then returned `unknown terminal id: ba-camera-corridor-20260919`.
Both failure receipts are preserved. The local record is not represented as a
successful global registry update, and the unrelated ledger was not edited.

Payload root: `artifacts.local/work/ba-camera-corridor-20260919/`. Retain frozen
`protocol.json`, `spec.json`, all three capture attempts, native source and RGB,
mechanical-repair/source-readiness receipts, observations, model outputs and seals,
per-frame results, metrics, figures and process-release receipts. Models are reused
from their existing canonical artifact roots, not duplicated.
The independent audit script/receipt are `verify_camera_corridor.py` and
`verification.json`; the bounded failure diagnostic is `failure-diagnostic.py`
and `failure-diagnostic.json` under that payload root. No reproducible temporary tree was
removed: all newly retained payloads support source-failure or result audit;
reclaimed space is zero bytes.

Protocol SHA-256: `3007ac49adb64eeac8a413f9b87a78babef8f9960f8e5e33171aaf17433e8a3e`.
Specification SHA-256: `206ee5dbf697790d0996c2459ae2c88d6a789eea4abc8a7667e754896943ac16`.

Implementation: [readouts](ba_camera_corridor.py), [scene spec](ba_camera_corridor_spec.py),
[capture](ba_camera_corridor_capture.py), [launcher](launch_ba_camera_corridor.py),
[materialization/inference](run_ba_camera_corridor.py),
[evaluator](evaluate_ba_camera_corridor.py), [metrics](ba_camera_corridor_metrics.py),
[geometry tests](test_ba_camera_corridor.py), [metric tests](test_ba_camera_corridor_metrics.py),
[renderer](render_ba_camera_corridor.py).
