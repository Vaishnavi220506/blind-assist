# Active next-observation selection: one parametric mechanism pilot

Frozen before the sole main run, 2026-09-21. EXPLORE, new analytic synthetic
Development. This is an observation-selection question, not a new alert model,
hardware test, replay of MZ155/MZ182, or comparison against Calibration.

Question: with two measurements, can choosing a second viewpoint from current
ambiguity resolve more fixed-corridor questions than a fixed scan or repetition?
The mechanism is separation of competing geometric hypotheses, not retrospective
selection of a view using the true scene or its future sensor returns.

## Frozen scene, query and observations

- World coordinates equal the initial camera coordinates: X right, Z forward.
  The query stays fixed at X[-0.30,0.30], Z[0.30,3.00] throughout. All targets
  span Y[-0.20,0.40] and the stipulated query profile is Y[-0.30,0.60]; only the
  horizontal slice varies. This is a disclosed profile for a toy source, not a
  user-specified mounting height or general three-dimensional result.
- Rectangles have depth thickness 0.04m. Their complete extent, not their
  center, determines strict intersection. Exact boundary contact is positive.
  Empty scenes contain only the wall at Z4.20m, outside the query.
- Eight center rays cover a 45-degree horizontal field (centers at half-zone
  spacing). Nearest first-hit radial range is quantized to 0.10m by rounding
  floor(r/0.10+0.5). Radial range is **not** treated as optical-axis Z: rendering
  uses (sin(theta),cos(theta)), while task truth intersects Cartesian boxes.
  No actor identity, depth image, true angle or metric surface enters an
  observation. This ideal eight-ray slice is not a validated 8x8 ToF simulator.
- No stochastic noise, packet loss, motion blur, reflectance or occlusion
  uncertainty is simulated. Camera translation and the forward model are known
  exactly. A ray can miss a narrow object; the wall remains the measured return.

## Frozen finite prior and cohorts

The declared prior contains 621 possibilities: empty, plus the Cartesian product
of X={-0.60,-0.56,...,0.60} (31), Z={1.0,1.5,2.0,2.5,3.3} (5), and
width={0.04,0.12,0.28,0.48}m (4). All use wall Z4.20m. The in-prior cohort is
the entire bank, one equally weighted case each. It is explicitly a closed-world
mechanism test, not an independently sampled/generalization benchmark.

Two separate out-of-prior stress strata are fixed: (a) off-grid X={-0.54,-0.38,
-0.22,0.22,0.38,0.54}, Z={1.25,2.25}, width={0.08,0.20}, yielding24 cases;
(b) X={-0.40,-0.24,0.24,0.40}, Z={1.5,2.5}, width0.12, but wall Z4.80m,
yielding8 cases. These are model-mismatch checks, not a comprehensive open world.
Keep every case, including invisible, already-decidable, no-match and boundary
cases. Report the initially ambiguous opportunity subset separately.

## Three arms and prediction boundary

Every arm gets the same first observation at (0,0) and exactly one subsequent
measurement. Passive repeats (0,0), with zero motion. Fixed scan moves to
(+0.12,0)m. Adaptive selects one of (+0.12,0),(-0.12,0),(0,+0.12),(0,-0.12)m;
each costs 0.12m one-way translation. Fixed and adaptive match measurement and
translation magnitude budgets. Passive matches measurement count only. No arm
returns to origin or receives intermediate-motion measurements in this pilot.

Predictor matches all eight quantized ranges exactly against the explicit bank.
It may forecast observations from bank hypotheses. Adaptive minimizes the sum,
over predicted next-observation groups, of positive-hypothesis count times
negative-hypothesis count. Ties minimize sum of squared group sizes, then use the
listed action order. With no hypothesis match, use the first action and remain
UNKNOWN; no hidden fallback or truth-directed choice. This is finite-prior
disagreement reduction, not a calibrated posterior or physical entropy claim.

Persist and hash source, protocol, predictor/model and initial observations;
persist and seal **all action choices before any actual second observation**.
Actual next observations are then supplied only at the selected view. Seal
public final predictions before task truth accounting. Log stage order and
verify upstream hashes. No true scene, case identity or actual future observation
is an argument to the selection function. Independent evaluator-only first-hit
attribution counts newly observed target rays after prediction sealing.

After the second observation, retain consistent hypotheses. If all agree on
query intersection, emit INTERSECTS or NONINTERSECTING_HYPOTHESES; otherwise emit
UNKNOWN. The latter negative decision is conditional on this finite source
model, **never a free-space certification**. A no-match bank is UNKNOWN, not a
negative. Prior labels are predicted geometry of public hypothetical states,
not labels of the true test scene. Do not shift/recenter the query with the camera.

## Outcomes, decision and stop

For full in-prior and each mismatch stratum report initial/final hypothesis
counts, initially mixed-label opportunities, decisive/correct/wrong/UNKNOWN
query decisions, TP/FP/FN among decisions plus abstention-positive/negative
counts, target-hit rays/new target-hit rays and target-ever-seen cases. A correct
negative is conditional-model identification, not independently certified clear
space. Report information reduction and costs alongside task outcomes; no event,
latency, safety or device claim is supported by two static views.

Retain only a scoped in-prior mechanism component if adaptive yields more correct
decisive answers than **both** fixed and passive on initially mixed-label
in-prior cases with no increase in wrong answers. Otherwise record no demonstrated
adaptive task benefit. Stress errors limit all positive conclusions; do not hide
them behind the primary gate. This gate does not authorize deployment/integration.

One fixed CPU analytic run, no search, fitting, phase/translation sweep, main
cohort rerun or successor. Small scalar geometry is TASK_NOT_GPU_SUITABLE.
Focused tests cover future/truth exclusion, seal enforcement, radial geometry,
extent intersection and fixed query frame. Mechanical defects can be repaired
with retained failed receipts and unchanged scientific settings. All evidence
stays under artifacts.local/work/ba-active-view-20260921; no persistent workers.
