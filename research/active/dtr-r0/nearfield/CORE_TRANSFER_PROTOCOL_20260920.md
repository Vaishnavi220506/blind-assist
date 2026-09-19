# Frozen Core Transfer Validation

2026-09-20. Representative BODY/HEAD Controlled Obstacles.
Question: do frozen calibration and local RGB lateral attribution transfer to
new controlled objects, backgrounds and physical arrangements?

The previous96 reused frames establish fitting/mechanism evidence only:
raw30/19/6, calibrated30/15/6, noRGB30/13/6, RGB30/10/6. The incremental RGB
effect is3 FP. They do not establish new-scene generalization.

## Frozen source and geometry

One36-scene x12-frame cohort: six geometry types x three lateral configurations
x two new backdrop configurations,432 frames total. Types are HEAD horizontal
obstacle, hanging plane and protruding edge; BODY protruding plane, suspended
solid and larger solid. Opaque cube meshes are geometric representatives, not
real branches, people or semantic object recognition. All36 target instances,
sizes, materials, relative paths and arrangements are specified before capture
in `core_transfer_spec.py`; every OUTSIDE arrangement is independent of the old
negative arrangement. No frame split or result-selected replacement is allowed.

Backdrop materials/configurations change from Limestone to Brick/Wood. Target
materials exclude old Bronze. The Willow map, primitive mesh library, renderer,
lighting system and sensor proxy are shared with Development. Thus this tests
new arrangements/appearance under a shared synthetic generator, not a new world,
natural distribution, physical ToF or device performance.

Use the original camera-frame volume X[-.3,.3],Y[-.2,.9],Z[.3,3] metres,
level camera1.7m above the road. The volume rotates with the camera; this remains
**Camera-forward upper-body obstacle awareness**, not body trajectory prediction.
Nominal12 samples at0.2s spacing,0.12m forward advance/sample. Timing is sampled
posed-trajectory timing, not measured real-time latency.

For complete authenticated obstacle bounds, define signed penetration into each
volume slab as min(upper-volume_min,volume_max-lower), then take the minimum over
X/Y/Z. INSIDE means penetration>2cm; BOUNDARY means absolute margin<=2cm;
OUTSIDE means margin<-2cm. Strict closed-volume intersection remains the binary
truth, including exact contact. This preserves full extent: a wide obstacle may
cross a boundary and still be INSIDE. Centre offset never defines the label.
The2cm tolerance is inherited and frozen. `layout_relation` is the planned lateral
configuration; `relation` is per-frame full3D geometry, including pre-entry depth.
Report both and never turn planned arrangement names into outcome labels.

Eighteen INSIDE/BOUNDARY/OUTSIDE configurations per background use distinct target
dimensions, material assignment and relative poses. OUTSIDE lateral gap is8cm
beyond the full target edge; BOUNDARY is exact lateral contact. INSIDE includes
centred and partially intruding full extents. Every clip includes pre-entry
frames. BODY/HEAD is a frozen scene-type height stratum, not a learned height claim.

## Four immutable arms

- A raw nominal45x45-degree8x8 single-return proxy and original possible/definite
  corridor readout, with unmodified ranges, validity, uncertainty intervals.
- B frozen calibrated readout, threshold0.007085703945147101; no recalibration.
- C B plus original5915-parameter RGB head from `d844c347`, `C.pt`.
- D B plus original matched noRGB head, `C_no_rgb.pt`.

Weights SHA256: C `9fa41303a413b68ec884e8fe1b2f627e4cdc4aa5566210631afcad94c140aaa9`;
D `7a48ef96c559dc6851714244969e6ba4fda4b3ca36f3e3d732a4f6303e168e39`.
Original eligibility,48x48 crops, geometry channels,0.95 OUTSIDE cutoff and
definite-vote bypass stay unchanged. C/D retain their original classifier labels
INSIDE/OUTSIDE/CROSSING based on whole-target lateral extent, with exact contact
CROSSING. These are diagnostic labels distinct from Core's tolerance labels.
Pure target-owned winning-bin contributors receive diagnostic labels; mixed and
non-target bins are UNKNOWN and remain in final alert denominators.

Zero training updates, checkpoint selection, new models, crop selection or
threshold changes. Simulation and scalar/OpenCV processing use CPU; the small
frozen heads use the available CUDA GPU, with actual runtime recorded. No edge
latency or hardware feasibility follows from host execution.

## Integrity and metrics

Freeze protocol/spec/source-code/model hashes before capture. Authenticate live
mesh, material, full render bounds, camera pose and native target centre trace;
retain renderer readiness and actor/process release receipts. Native depth is
used by the sealed sensor constructor only. Require visible target pixels and
<4 competing native corridor pixels on every frame; a failed source is
NOT_EVALUABLE, never a method-negative or silent exclusion.

Inference consumes only RGB,64 public ranges/boxes and frozen parameters. Seal
all432 four-arm predictions before scoring truth or native contributors. Audit
every suppressed eligible zone, including contributor loss hidden by other votes.
All raw anchors/intervals remain immutable. UNKNOWN abstentions never become TN.

Report all-frame TP/FP/FN, recall, precision, FPR over all known negatives,
UNKNOWN/abstentions; BODY/HEAD, type, background, planned lateral configuration
and full3D INSIDE/BOUNDARY/OUTSIDE strata. Core event recall uses12 planned INSIDE
clips with strict intersection events; report12 BOUNDARY-contact clips separately.
Also report all strict events, each event's first in-event alert and clip-first
alert, preexisting/left-censored alerts, false segments and inclusive sampled
duration (alert frames x0.2s). Segment masks retain full timelines.

Primary success: RGB strictly reduces OUTSIDE-layout false-alert frames against
both B and D, and total FP against both; retains every raw/B/D TP, detected event
and first in-event alert; and suppresses zero native corridor contributors.
Nonempty positive/core-event/OUTSIDE-frame/eligible-OUTSIDE opportunities are
required. Report segment fragmentation and duration explicitly even if frames
improve. Calibration transfer is evaluated separately against raw with fewer FP
and full TP/event/onset retention. Diagnostics are RGB-minus-noRGB FP, eligible
pure-subset classification accuracy/confusion and native contributor retention.
Repeated frames are not independent samples;36 scene arrangements are the unit
of transfer scope. A pass is bounded controlled evidence, not general reliability.

## Stop and evidence ownership

End after one complete evaluation and delivery. Failure does not trigger training,
0.95 tuning, crop/architecture changes, extra backgrounds or another seed.
Mechanical source/API repairs may retain unchanged geometry/weights/criteria with
original failure logs and exact repair receipts; never select scenes by outcomes.
No automatic successor. Keep the original96 and old thin-cylinder/plate/same-zone
results unchanged as **Thin-object Challenge**; no retrospective Core relabelling
or merged score. Future4/6/8/10cm collection is not part of this run.

Payload: `artifacts.local/work/ba-core-transfer-20260920/`. Preserve sealed
protocol, scenes, weights by reference, RGB/native source, observations, lineage,
predictions, per-frame/per-zone metrics, runtime/failure/metadata receipts and
release records. Release all task-owned UE/GPU processes after completion.
