# Four-sensor simulation mainline

User-confirmed architecture, 2026-09-13: **one RGB camera + ToF + Radar + IMU**.
The immediate priority is algorithmic benefit in class-agnostic forward obstacle
awareness. Work remains simulation-only. A second camera, stereo depth, removal
of Radar, or substitution of simulator truth for an observable input changes the
architecture and requires an explicit new user decision.

## What the next implementation must answer

1. Does a measured surface occupy the current walking/body/head corridor?
2. Which image region, ToF zone and Radar return can credibly refer to one target?

RGB supplies image extent and angular localization. ToF supplies zonal range and
validity, not exact per-pixel depth. Radar supplies uncertain range, bearing and
radial velocity, not object identity. IMU supplies rotation increments, not
accurate metric translation or the user's future intended path. Time and
extrinsic calibration are explicit inputs. Association ambiguity stays visible.

Use identical ToF/Radar/IMU packets, geometry, history and task labels in the
three-sensor baseline and the four-sensor candidate. Report individual small-pole
and HEAD retention, false alarms, missing support/UNKNOWN, association coverage
and cost. Preserve independently valid ToF evidence. Missing RGB detections or
missing range returns cannot establish clearance. Never suppress an independent
sensor merely because another sensor missed the object.

## Evidence routing and correction

- MZ90--100 are range/rotation sensor experiments; their predictors do not consume
  RGB. They do not establish a tested four-sensor fusion system.
- MZ101--106 are a **separate stereo RGB + ToF branch**, with no Radar in the
  evaluated pipeline. Preserve their negative results and baselines within that
  branch. They do not diagnose the four-sensor mainline's algorithmic ceiling.
- MZ107 starts a fresh, bounded, synchronized four-sensor simulation canary.
  Its simple rendered appearance is controlled Development, not a general visual
  detector benchmark. Native geometry and target identities are evaluator-only.

## Advancement and stop points

First verify input provenance and that actual RGB pixels affect association.
Then measure a paired task effect with an RGB-disabled control. A gain must
retain critical baseline positives; fewer reported points alone is not a gain.
MZ109 exposes an additional requirement: also report absolute critical-stratum
recall and positives unique to the active RGB challenger. Retaining100% of a
baseline that detects1/16 true boundary frames does not establish adequate recall.
Keep comparator-specific gains and tradeoffs visible; do not promote a method
solely because a baseline-only gate passes.
If image proposals or associations fail, report that specific mechanism and stop
the bounded attempt. Do not jump to larger depth models or more sensors.

Only after a useful observable association mechanism is fixed should a complete
new scene split test it unchanged, including ambiguity, low contrast, occlusion,
time/calibration errors and sensor gaps. Current-corridor occupancy and future
collision prediction are different tasks; report them separately. Each experiment
ends at its stated budget, followed by evidence and code delivery.

## Revised design direction: spatial evidence before object categories

User proposal, 2026-09-13; not a validated algorithm. Represent coarse forward
regions with local contour/extent candidates, range and extent uncertainty,
known/unknown height, sensor provenance, evidence age and unobserved space.
Do not fill a whole grid cell or clip a thin structure at a cell center. A
ToF-zone range is not automatically the depth of every image pixel in that zone.

ToF supplies zonal distance and validity; Radar supplies independent distance,
coarse bearing and motion with HEIGHT_UNKNOWN where unobserved; RGB supplies
boundaries and angular localization without requiring a named object category;
IMU compensates short-term rotation without inventing metric motion or intent.
Allow separate RGB+ToF and RGB+Radar candidate branches. ToF absence must not
prevent consideration of an RGB+Radar pair. Conflicting ranges can belong to
different objects; preserve alternative explanations and independent support.
Repeated Radar detections alone do not establish independent corroboration.

Use the initial body/walking reference for the first simulation comparison,
explicitly retaining reference uncertainty. Head orientation and future motion
remain distinct. A multizone footprint/multireturn simulator is a separate
sensor-model study: existing center-ray results cannot decide physical thin-pole
detection, and a replacement must not assume that every thin pole is detected.

[MZ110 diagnosis](MZ110_RESULTS_20260913.md) finds six ToF-blocked opportunities
among 30 nominal misses; simple gate removal recovers four but adds a net FP.
All nine pole misses lack target Radar and five new FP have correct diagnostic
identity but wrong geometry. Preserve the diagnostic as a component, keep both
existing comparators, and do not promote the gate-removal counterfactual.

[MZ111](MZ111_RESULTS_20260913.md) and its unchanged new-scene
[MZ112 evaluation](MZ112_RESULTS_20260913.md) establish controlled recall gains
from independent RGB/Radar association plus current-visual-correspondence-gated
short range persistence. Fresh nonstress FN29 to7 at the same29FP and maximum
first alert delay1.0s to0.25s; full-panel FP38 to41 exposes the boundary tradeoff.
Keep these as components/challengers, with explicit age/ambiguity and retained
nominal/interval/hold comparators. Plane/filter complexity shows no fresh gain;
zero-velocity controls show most persistence gains do not require Doppler.
The actual reference remains initial body axes; pose drift, full miss/nuisance
curves and ghost discrimination remain unresolved.

[MZ113](MZ113_RESULTS_20260913.md) tests actual pixel correspondence on a new
240-frame dynamic source with15obstacle exits. The optical-flow combination
improves incumbent84TP/60FP/4FN to87/61/1, recovering three crossing-pole frames;
the same-policy angular control remains84/60/4. All16events start alerting at
the first sampled positive bin. Post-exit false support stays26frames/6.5s.
The one addedFP correctly tracks a pole still7mm outside the corridor. Retain
the controlled tracking component and its ablation, not a nuisance-improvement
claim. Dynamic sampled boxes do not establish natural tracking or full occlusion
robustness; initial body reference and hypothetical RF limits remain.
Post-seal diagnostics identify40independent persistent-ghost FP among61total FP;
the sole FN has range/image evidence but reciprocal association ambiguity. These
are separate responsibilities, not reasons to increase persistence lifetime.

[MZ114](MZ114_RESULTS_20260913.md) rejects a fixed global-assignment/range-prior
alert replacement:768consumed frames375TP/138FP/21FN ->377/142/19. Preserving the
old matched-pair range smoothing restores one lostTP (posthoc378/142/18), but
no net FP or timing benefit appears; ghost-to-object mistakes can masquerade as FP
removal or reinforce new alerts. Keep it as a negative control within that role.
Suppressing unmatched Radar would remove71FP but lose31TP on480frames, including
25correct hazardous-actor supports. Neither absent center rays nor repeated
compatible ranges are enough to identify persistent ghosts. Study finite-footprint zonal
return information next; do not assume a reported range-noise sigma is a true
within-zone spatial depth distribution, or strongest-first is nearest-first.

After fixing an observable method, evaluate complete new scenes with identical
alert distance/corridor/margins and the full miss-versus-reminder relationship.
Report obstacle-event misses, first correct alert timing and irrelevant alert
count/duration per route. Keep centimeter-boundary pressure tests separate from
event-level headline results. If RGB cannot reduce nuisance at matched recall
and timing, retain it for direction display/explanation within that tested scope.
