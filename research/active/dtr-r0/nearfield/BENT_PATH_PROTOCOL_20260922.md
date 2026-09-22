# Equal-cost path geometry and a fixed perturbation smoke check

2026-09-22 EXPLORE. User authorized the proposed first priority: change observation
geometry rather than classifier tuning. Prior fixed+X13view evidence gives79correct
model-conditional commitments and101UNKNOWN,98with explicit opposing witnesses.
This experiment changes the physical path; it does not reopen the single-pair
selector or numerical projection recipes. No yaw or RGB arm is included.

## Hypothesis, source and paths fixed before observation generation

Combining lateral and axial motion may constrain different rectangle faces at the
same motion/observation budget. Reuse all180scenes/90pairs from sealed
ba-observation-mechanisms-20260922/run-v1. This is consumed same-generator
Development, not fresh source confirmation. Newly simulated off-axis views must
be counted as new observations; do not describe this as saved-trace-only replay.

Three globally fixed trajectories,13observations including the origin,1cm per
step and12cm Euclidean travel. No scene-conditioned routing or choosing a winner:

- straight_x: +X12cm, incumbent baseline.
- x_then_z: +X6cm then +Z6cm, the PRIMARY challenger.
- z_then_x: +Z6cm then +X6cm, prespecified order control, same final pose as primary.

No orientation change. Equal sample count and travel do not establish equal human
effort, turning cost or sensor acquisition time. Simulated static scenes have no
motion blur or elapsed-time dynamics. No path/adaptation fit follows readout.

## Inference and conservative authority

Extend only the accepted nominal pose whitelist to the union of the three paths.
All nominal cameras have cz<=.06m, behind the minimum rectangle front .58m;
the unchanged ray/box outer-constraint derivation still applies. Keep the same
continuous single-box domain,8rays,.10m bins,wall4.2m,thickness.04m and query.
Use exactly2MILPs per unique camera+bin history, original1s/10000node budget.
Retain original EPS outer contact envelope and frozen closed-query IN projection
plus exact forward validation. No old code edits, tolerance sweep, threshold fit,
cache-based selection or extra fallback solve. Capture raw candidate diagnostics.

Conditional IN/OUT requires original ownstatus0, a valid same-class witness and
opposite outer-relaxation status2/no candidate. Candidate rejection, search limit,
both-side infeasibility, or conflicting constructive evidence remain UNKNOWN.
Splitting one witness pair is NOT evidence of excluding the whole opposite class.
Infeasibility reports remain numerical single-rectangle-model claims, not formal
certificates or measured free space. Full180denominators and UNKNOWN preserved.

## Fixed small perturbations, no robust-model claim

Five conditions, applied identically across paths and cases:

1. nominal exact known pose and range.
2. range_plus2mm: add+.002m to every radial return BEFORE .10m rounding.
3. range_minus2mm: add-.002m before rounding.
4. pose_plus1mm: actual camera=nominal+(.001,.001)m; inference receives nominal.
5. pose_minus1mm: actual camera=nominal-(.001,.001)m; inference receives nominal.

One bias type per condition. Global constant pose translation preserves12cm travel;
sensor/head yaw is unchanged. These fixed values are an uncalibrated smoke check,
not hardware error estimates, random noise statistics or a worst-case bound.
The inference model remains exact; perturbed arms deliberately violate its sensor
assumptions. Label them MIS_SPECIFIED_SENSOR_STRESS and count any incorrect
conditional commitments. Valid model witnesses then explain reported bins, not
necessarily the actual biased sensor geometry. Never claim stress-free-space.

## Evaluation and bounded decisions

Seal paths, condition values, code and source references before generating views.
Generator sees source geometry only to create observations. Inference accepts only
nominal camera+bins; source truth joins only after all predictions are sealed.
Reuse identical public histories computationally; report unique calls separately
from3paths*5conditions*180cases. Count unique generated observations, inherited
nominal +X replays, and selected histories separately. All measurement metadata
and actual perturbed poses remain evaluator-only, outside the inference input.

Primary contrast is x_then_z versus straight_x for each matched condition. Report
z_then_x separately without replacing the primary. FullTP/FP/falseOUT/correctnegative/
UNKNOWN, recall/FPR, exact per-case gains/losses and left/right-width/depth slices.
Nominal baseline reproduction should preserve prior corrected34TP/45correctnegative;
record any mismatch rather than force agreement. Check prior98opposing fixed-path
UNKNOWN recovery and baseline decision loss, not just a favorable subgroup.

Strong nominal improvement requires newTP, retaining every baselineTP, no newly
wrong IDs and nondecreasing correct commitments. Report partial tradeoffs if this
fails. A gain is NOT robust if it disappears or introduces newly wrong IDs in any
of the four fixed stress conditions. This smoke check cannot establish robustness
even when passed. Compare same stress against same stress as well as each path's
stress versus its own nominal result. Do not add a second perturbation scale.

One scored3x5comparison. If failed, preserve the result; do not tune path length,
turn point, sample placement, bias, model domain or solver. Narrow mechanical
repairs preserve failed outputs and must be disclosed. Focused tests, independent
saved-output audit, local dispositions, supported metadata attempts, scoped
commit/push complete delivery. No automatic yaw/RGB successor in this experiment.
CPU TASK_NOT_GPU_SUITABLE; no paid or persistent resources.
Artifacts: artifacts.local/work/ba-bent-path-20260922/run-v1, refuse overwrite.
