# Separate boundary sampling, range quantization and coordinate ambiguity

2026-09-22 EXPLORE. User asks to continue after the
[fixed drift test](BIAS_DRIFT_RESULTS_20260922.md) found all60boundary cases UNKNOWN
under the full model. Those180scenes are now consumed Development. This is a
fixed observation-information diagnostic, not classifier tuning or confirmation.

## Hypotheses and predeclared comparisons

Sparse ray directions or coarse quantization may hide opposite-label pairs;
known angular variation may separate some. Separately, unknown global camera-X
translation can make global-query labels non-identifiable even with ideal ranges.
These are different mechanisms; breaking one selected pair is not proving class
uniqueness over the continuous model.

Reuse exactly180 saved geometries and their30designated boundary pairs. Same
nominal x_then_z path,13views and8rays/view. No source reselection, new geometry
search or path optimization. Three known-yaw schedules in radians:

- fixed: yaw0 throughout, frozen baseline.
- sweep_plus PRIMARY: yaw_i=i/12 times one ray spacing(5.625degrees).
- sweep_minus CONTROL: yaw_i=-i/12 times5.625degrees.

Both sweeps start at yaw0 and add5.625degrees total orientation travel; baseline
adds0. Translation remains12cm and sample count104rays/history. Rotation has real
additional actuation/pose-estimation cost and exact known yaw is assumed. No
hardware feasibility or equal-total-motion claim. No yaw sweep chosen by truth.

Keep original slab-ray law and fixed wall. Capture unquantized radial ranges and
target-hit flags once per scene/schedule/view. Apply two fixed quantizers to those
same returns: existing .10m and a hypothetical .001m precision control. Finer
quantization is a change of measurement capability, not free software recovery;
raw ranges are an ideal information diagnostic, never inference inputs.

For each boundary pair/schedule report identical versus different histories at
both quantizations, raw maximum absolute separation(tolerance1e-10m only for
floating comparison), and target-hit mismatch. Report face-specific and overall
counts, gained/lost separated pair IDs relative to fixed at the SAME quantizer;
also compare precision controls within schedule. A selected pair becoming
different is a necessary information opportunity, not TP, TN or global uniqueness.
No recognition/recall/FP improvement may be claimed from pair separation alone.
Report changed bins for the full180scenes versus original fixed observation.

## Stronger counterexample: global lateral translation

After observation sealing, use evaluator truth only to construct an explicit
opposite-label scene for all40lateral boundary cases. Shift object X and every
actual camera X together by delta=±.001m, choosing the sign to cross the relevant
left/right query face. Original scene uses zero biases; alternative uses shared
pose_x=delta, pose_z=range_bias=0. Hold width, depth, wall, nominal poses and yaw
fixed. Check both scenes inside the original geometry domain, |delta|<=1mm,
opposite original-query labels and full-bin replay for ALL3yaw schedules and
BOTHquantizers. Save all alternative geometries, actual poses and raw residuals.

The algebraic reason is relative object-camera X invariance, while the infinite
horizontal background wall does not constrain X. Exact relative geometry makes
the argument independent of ray count, angle and range resolution in the ideal
declared model. Finite float replay is a check, not a formal solver certificate.
This is an evaluator-constructed non-identifiability witness, never a production
proposal or calibration estimate. It applies to the full pose-uncertain model;
range_only excludes this counterexample by assumption, not by measuring absolute
pose. Far-face cases are outside this lateral proof. No claim about RGB, landmarks,
independently measured pose or redefining the query relative to the true camera.

## Execution, outcome and stop

Seal protocol/code/dependency and old-input hashes before captures. Read source
geometries for simulation only; do not adapt schedules to their labels. Seal all
observations before evaluator pair scoring and counterexample construction.
Old fixed/.10m captures must replay bit-identical to prior nominal observations.
No inference changes or MILP calls. One180x3x13x8 capture comparison,56160base
ray returns; counterexample validation is saved separately and counted separately.
No angle/precision/margin/bias sweep after results, solver adaptation, new source
or successor. Complete focused tests, independent saved-output audit, report,
current, supported metadata attempts and scoped commit/push.

If angle sweeps separate pairs, retain as a measured sampling mechanism with
costs; if precision separates more, retain that capability-dependent diagnostic.
If all40lateral counterexamples validate, prefer independently grounding lateral
pose or explicitly changing the query contract over claiming angle/resolution
alone resolves those labels. A failed counterexample must be reported by case;
do not generalize the argument beyond verified domain and assumptions.
CPU scalar TASK_NOT_GPU_SUITABLE, no paid/persistent resources. Refuse overwrite.
Artifacts: artifacts.local/work/ba-boundary-observability-20260922/run-v1.
