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

## Useful core capability and challenge coverage

User clarification, 2026-09-20: optimize for useful, measurable obstacle alerts
within a declared operating scope and limited compute. Extreme-case perfection
and lossless improvement on every frame are not universal project requirements.

- **Core:** representative controlled BODY/HEAD obstacles clearly intersecting
  the corridor, paired with clearly outside obstacles and negative periods.
  Prioritize event detection, timely first alerts and fewer false interruptions.
  The six geometry types in the completed Core transfer are controlled examples,
  not measured frequencies of everyday obstacles or proof of general coverage.
- **Boundary:** exact contact and uncertain/grazing intersections are reported
  separately, with their original strict labels and declared tolerance retained.
- **Thin-object Challenge:** retain the old 4 cm rods/plates and other declared
  extreme cases as capability-boundary evidence. Their failure limits coverage;
  it does not by itself invalidate a useful Core result.

Define future cohort membership from dimensions, distance, placement and source
conditions before outcome scoring. Do not invent a universal minimum width from
the old 4 cm failures, discard difficult Core errors after observing them, or
merge Core and challenge scores. Missing sensor support remains an informative
miss/abstention where truth is known, not a reason to remove the sample.

Judge future candidates against the adopted Calibration baseline by event
misses, first-alert timing, false-alert segments/duration and UNKNOWN coverage;
retain frame TP/FP/FN and precision/FPR as supporting diagnostics. A candidate
may offer a useful tradeoff without preserving every baseline-positive frame.
State the intended benefit and acceptable miss/delay cost before a new comparison;
there is no default permission to accept a lost event for a small FP reduction.
Set only criteria relevant to that question; one bounded comparison is enough
unless an observed failure or evidence gap changes the decision.

The old zero-TP-loss, zero-onset-delay and zero-native-contributor-suppression
gates remain binding for the experiments that claimed **lossless** improvement.
They are not automatically inherited by every future alert policy. Preserve raw
valid observations and UNKNOWN; an alert-policy choice must not rewrite sensor
evidence or manufacture foreground distance/free space. Contributor audits are
required when the proposed mechanism or claim depends on attribution/retention,
not as a universal prerequisite for every unrelated reversible change.

This clarification changes future task selection and acceptance, not frozen
labels, results, thresholds or negative-control dispositions. It starts no new
training, capture or runtime change. Current Core evidence already separates
these strata; the remaining demonstrated baseline weakness is false alerts.

## Limited edge-compute constraint

User clarification, 2026-09-19: the intended system must not depend on a large
model or assume powerful edge hardware. Compute is a design constraint from
the start, not a later compression task after demonstrating a large-model path.
The online candidate must not require desktop-class CUDA, a large dense-depth
backbone or off-device inference to satisfy this task.

The [completed Depth Pro comparison](BA_CAMERA_CORRIDOR_20260919.md) remains
historical diagnostic evidence. Its use ends with that frozen comparison; do
not continue large-model trials or make their success a prerequisite for the
research route. Retained NFO is a comparator, not proof of deployment suitability.

Start candidate design with the 64 zone observations, calibrated geometric
support and bounded image processing. Use limited-resolution or selected-region
RGB evidence to investigate surface extent and echo association; a small learned
component is eligible only within the actual execution budget. Cheap contours
are not guaranteed object boundaries, and RGB alone must not invent an unsupported
foreground range. Preserve independently valid ToF support and UNKNOWN when
association or distance remains unresolved.

Before selecting the next runtime candidate, record the target processor,
available acceleration, working-memory limit, required update rate and end-to-end
latency budget. Power/thermal limits need device evidence where relevant. These
quantities have not been specified by this clarification; do not invent numerical
budgets or assume an NPU is available. Verify the proposed path on the target or
a clearly identified constrained proxy before claiming that it fits. Model size
or desktop timing alone is insufficient; include image processing, transfers,
fusion and decision cost. Offline source construction and saved-result analysis
do not establish online feasibility.

This clarification changes future candidate selection, not frozen experiment
inputs, outcomes, denominators or historical timing claims. It starts no new
model execution, training, capture or hardware acquisition.

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

## Baseline and bounded comparisons

The adopted [Calibration baseline](CALIBRATION_BASELINE_20260920.md) supplies
the current camera-forward comparator. NFO reuse and the original thin-object
pairs below are historical diagnostic options, not mandatory next steps. Select
one concrete Core benefit before proposing a candidate; extra model complexity
or an RGB contribution is not itself an acceptance requirement. Keep the original
reports available; relabelling cannot establish improvement over a baseline
evaluated under another task definition.

When the question needs a geometry diagnostic, use an authenticated pair with
one intended variable changed. These pairs do not define the Core success gate:

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
