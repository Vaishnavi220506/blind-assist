# Shared sensor-bias variables on consumed bent-path observations

2026-09-22 EXPLORE. User authorizes the proposed measurement-error mechanism after
the exact model rejected both classes under two small injected perturbations.
That run remains frozen. This is a new declared observation model, not a rescue
by fitting a tolerance to correct labels. No actual sensor calibration is available.

## Fixed explanation and contrasts

Each complete history has one unknown radial offset b and one unknown camera
translation(px,pz), shared by every ray and view. Measurement before quantization
is original_radial(scene, nominal_camera+(px,pz)) + b. Keep the original object
domain, query frame, thickness.04m, wall4.20m,8rays and.10m quantization.

Two presets, fixed before scoring:

- range_only comparator: b in[-.002,.002]m; px=pz=0.
- range_pose PRIMARY: b in[-.002,.002]m; px,pz each in[-.001,.001]m.

Bounds come directly from the prior frozen SYNTHETIC injector design, not hardware
specifications, a newly estimated error distribution, per-case truth or favorable
readout. Both signs and zero remain possible for every input. Full model also
admits combined offsets; previous five conditions do not exhaust that domain.
No condition name or true bias selects a preset, fixes its variables or chooses
the returned hypothesis. Range_only deliberately omits pose error and is only
fully specified for nominal and range-only conditions.

## Continuous constraints and witness authority

Use L,R,F,interior slack,b,px,pz as continuous variables plus disjunction binaries.
Biases are shared across the history, not independent enlarged bins per ray.
Wall observations also constrain b,pz. A wall return is no longer forced to equal
the nominal wall bin. Model interval envelopes separate possible object returns
from wall returns by much more than declared uncertainty; document and test the
branch rule and inclusive outer boundaries before scoring. Global pose variables
enter hit/miss and range equations as well as wall depth. Physical query labels
remain in original coordinates, not translated along with the camera.
The existing injector rounds translated cameras to12decimal places. Before scoring,
derive a fixed1e-12m per-coordinate outer envelope for that numerical operation;
propagate it outward through both wall-bin bounds, target-bin bounds and hit/miss
rows. It is a numerical enclosure, not a new measured-noise bound or fitted epsilon.

The outer relaxation must contain every declared exact model realization at
slack0, including inherited EPS contacts and half-open-bin closure. Opposite-class
status2 is only numerical model-conditional exclusion, never a formal certificate.
Exactly two original-budgetMILPs per public history/preset (1s/10000nodes each),
one IN and one OUT. No retry, budget sweep, fitted epsilon or learned calibration.

Candidate geometry AND nuisance values must pass original-domain and preset-bound
validation and reproduce ALL observed bins using the unchanged scalar ray law,
global pose translation and prequantization bias. Use the prior fixed nominal-query
IN projection only for a rejected candidate; leave nuisance values unchanged and
revalidate every bin. Store raw vectors, decoded candidates and failure reasons.
A same-class witness plus ownstatus0 and opposite status2/no candidate permits a
conditional label. Incomplete search, invalid witness, both classes infeasible or
conflicting constructive evidence remain UNKNOWN. A nuisance estimate explaining
bins is not evidence that the true bias or object geometry has been identified.

## Input, evaluation and stop

Use all749public13view histories sealed in ba-bent-path-20260922/run-v1. Replay
only; no new actual/simulated observations or fresh source. Keep the five existing
conditions, three paths and180cases per cell (90positive/90negative). Copy and
hash-check old evidence. Seal model/protocol/input before solves and all new
predictions before joining source truth, condition labels or true injected biases.
Inference receives only nominal camera+bins and the global preset name.

Compare each new preset with frozen exact-model predictions for each same path/
condition. Report conditionalTP/FP/falseOUT/correctnegative/UNKNOWN, gained/lost IDs,
retained nominal decisions, and genuine opposing witnesses versus incomplete
search versus both-class infeasibility. Primary x_then_z remains primary; report
straight_x and z_then_x controls without choosing a winner after readout.

Main capability test: on x_then_z, both range_minus2mm and pose_plus1mm should
recover correct commitments from prior0/180 without wrong commitments. Also
report nominal coverage sacrifice; passing a collapse recovery does not establish
lossless replacement or robustness. A stronger local stability claim additionally
requires retaining every correct nominal baseline case in every modeled condition
with no newly wrong case. Report any failure rather than relax the condition.

Independent audit replays witnesses and analytically checks true geometry+known
injector assignments against the full-model outer rows AFTER prediction sealing,
not as proposal inputs. Record any erroneous exclusion of the actual valid model
as a defect. Check range_only truth consistency only where its assumptions apply.

One749query x2preset run,2996MILP calls maximum, original per-call budgets. No
envelope enlargement, nuisance variant, new error amplitude, path choice or second
scored repair after results. Preserve mechanical failures. Complete focused tests,
saved-output audits, scoped report/current/code delivery and supported metadata
attempts. Evidence remains consumed model-only Development; no hardware/safety/
natural-distribution claim. CPU TASK_NOT_GPU_SUITABLE; no paid/persistent resources.
Artifacts: artifacts.local/work/ba-shared-bias-20260922/run-v1; refuse overwrite.
