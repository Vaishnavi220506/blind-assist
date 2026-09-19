# Camera-forward corridor: adopted research scope

User decision, 2026-09-19. The camera optical axis defines the current
forward/attention direction. The immediate minimal system is RGB + 8x8 ToF,
near-field spatial evidence, camera-frame corridor intersection, then alerts.
This supersedes the four-sensor/body-heading requirement for this scoped work.
Head/body separation, IMU head-turn compensation, future body trajectories and
SLAM are outside it. Radar is not required by this minimal comparison.

This is an adopted task definition, not a measured alert improvement or an
implemented runtime change. Historical A/A*, four-sensor experiments and their
labels remain reproducible under their original definitions. Existing closed
recipes remain closed in their tested scopes; this decision starts no new fit
or capture and does not change Android defaults.

## Geometry

Use metres and camera coordinates: X right, Y down, Z forward along the optical
axis. Adopt the user's initial corridor dimensions for the next comparison:

`0.30 <= Z <= 3.00`, `abs(X) <= 0.30`, `Y_top <= Y <= Y_bottom`.

The vertical bounds are required fixed profile parameters, chosen from the
mounting height and intended upper-body span before evaluation. For a level
camera at height H, a nominal height band [h_low, h_high] gives
`Y_top = H - h_high`, `Y_bottom = H - h_low`. Record the actual values and
mounting orientation with the profile; no numerical height was supplied in
this decision. The volume rotates with the camera, including pitch and roll.
It is a camera-relative attention corridor, not a gravity-aligned body envelope
when the wearer tilts the camera. No online body-pose correction is implied.

For an undistorted pixel (u,v), calibrated intrinsics give
`a=(u-cx)/fx`, `b=(v-cy)/fy`. With optical-axis depth Z,
`(X,Y,Z)=(a*Z,b*Z,Z)`. With radial range r measured from the camera centre,
`(X,Y,Z)=r*(a,b,1)/sqrt(1+a*a+b*b)`.
For a separate ToF, back-project in its own calibrated ray coordinates first,
then apply the fixed extrinsic transform `p_camera=R*p_tof+t`.
Preserve whether each input is axial depth or radial range.

The user's horizontal formula is equivalent to `Z=r*cos(theta)` and
`X=r*sin(theta)` for a ray in the horizontal plane. If d denotes axial depth
instead, `X=d*tan(theta)`. At 10 degrees, radial ranges 1m and 3m give lateral
offsets about 0.174m and 0.521m: the first is inside and the second outside.
A fixed central image rectangle cannot express this constant-width volume.

Test obstacle spatial extent against the volume, not merely the object centre.
A wide object centred 40cm sideways can still intrude. Closed-volume contact
counts as intersection in the strict geometry label; exact contact and uncertain
boundary cases must also be reported separately with their declared tolerance.
Distances below 30cm are outside this evaluation volume, not evidence of safety.

## Observable evidence and readout

An 8x8 zone return constrains coarse angular/range support; it does not identify
the exact pixel or cover every surface in the zone. Preserve missing/ambiguous
support as UNKNOWN and retain valid native observations when adding RGB support.
Outside-only returns and absence of an alert do not establish free space.

The retained [NFO component](BA_NFO_MATCHED_20260919.md) predicts nested
near-depth scores at 1.0/1.5/2.0/3.0m. They are not a single metric depth and
do not resolve the 0.30m lower boundary. Do not silently convert a binary
near-mask or zone-centre range into exact per-pixel geometry. A future readout
must expose its depth-bin/range-support uncertainty and state how ambiguous
intersection becomes an alert or abstention. Choose that policy on development
inputs before reporting held-out results; pixel thresholds are not automatically
valid alert thresholds.

Reference depth/meshes can generate evaluator-only camera-corridor labels when
intrinsics, axis convention and the fixed vertical profile are known. No body
heading or SLAM pose is needed for this camera-frame task. Such labels describe
visible/reconstructed reference support, not unseen physical occupancy; missing
truth remains UNKNOWN. Never feed reference depth or actor identity to inference.

## Smallest decision-changing comparison

Compare a fixed RGB+ToF/NFO corridor readout with a fixed raw-ToF support readout
under the same camera profile and alert policy. Reuse the retained NFO weights
first; record new-task baseline metrics separately from historical A* numbers.
Keep the original-task reports available; relabelling cannot establish an
improvement over a baseline evaluated under another task definition.

Use authenticated paired scenes with one intended variable changed:

| Pair | Controlled change and required distinction |
| --- | --- |
| Same thin pole | Centred versus shifted 40cm right; verify the entire shifted pole is outside. |
| Same horizontal bar | Intrusion versus grazing; retain a separate exact-boundary stratum. |
| Same distance/material | Large versus thin obstacle, both with explicitly known corridor intersection. |
| Same ToF zone | Near foreground inside versus outside the corridor; keep background/range setup matched and record actual public returns. |

Keep complete physical pairs/sequences in one split. Same zone does not mean
identical ToF readings; verify rather than assume observation equivalence.
For each method report TP/FP/FN, event recall, false-alert segments and duration,
first alert relative to first labelled entry, and UNKNOWN/evaluable coverage.
An event is a contiguous positive interval of the same obstacle in a sequence;
a false-alert segment is consecutive alert frames on known negative truth.
UNKNOWN gaps split evaluable intervals and are counted separately. Report
already-active alerts at sequence start as left-censored; missed events have
no first-alert time. Fix timestamps and any debounce before comparison.

Pixel IoU remains a component metric. Retain a new readout only for demonstrated
corridor-alert benefit with disclosed miss/false-alert/timing tradeoffs. Improved
IoU or pair ordering alone does not establish that benefit. This delivery fixes
scope and evaluation semantics; runtime implementation and alert validation
remain to be done.
