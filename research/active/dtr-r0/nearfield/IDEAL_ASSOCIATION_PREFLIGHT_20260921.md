# Ideal contour/return association: bounded feasibility preflight

2026-09-21, EXPLORE. User authorized one existing-data diagnosis, no fitting,
capture, thresholds, protected-test access or automatic successor. Question:
can perfect RGB localization and return association, with unchanged public ToF
intervals, support whole-obstacle lateral discrimination on the 1152-frame cohort?

## Admissible information and feasibility condition

An evaluator adapter may reconstruct a target's visible 2D mask and tag existing
winning-return contributors. Only these oracle 2D identities may cross to the
diagnostic. True depths, object dimensions/3D bounds, winner-bin index, native
inverse-depth weights and truth labels must not become geometric decision inputs.
There is no permission to assign a returned interval to noncontributing pixels,
interpolate unmeasured depth, assume a constant-depth/planar object, or treat
outside sampled returns as evidence that the whole object is outside.

Inspection finds the original visible mask is reconstructed by native depth
backprojection into authenticated target bounds with the existing 2 cm tolerance;
it is not an independently rendered per-pixel instance-ID image. Preserve that
provenance and reproduce the source admission counts, rather than silently
claiming perfect segmentation. Return lineage identifies only pixels in an
observed winning bin; rejected bins and dropout are not observations.

The fixed public interval (.1 + 3*(.01+.02*r) radius) includes a 10 cm scatter
allowance and may support a conditional contributor-surface diagnostic. It is not
a hard physical guarantee and cannot bound noncontributing object surfaces.
Full-object negative decisions require additional coverage/depth constraints;
pixel centres alone do not certify continuous surfaces or occluded extent.

## One bounded check

Verify the sealed source records and all 1152 native payload hashes. Reconstruct
the same visible target masks only in the evaluator adapter. Count native and
sampled-lattice target pixels, those contributing to observed returns, and the
remaining sampled pixels: outside the ToF footprint, in a zone with no observed
return, or in an observed zone but outside its winning contributor set. Check
return IDs/ranges against public observations; never read winner bins/weights
for this computation. Seal these coverage results before joining saved alerts
and target truth for subset counts.

Report full-cohort and original B current incremental subsets (13 lateral FP,
15 distance FP,118 Boundary rescues,2 Core rescues), including per-layout counts.
Coverage is not an alert, a metric depth map, an IoU result, or an information
ceiling. Do not fit/select a classifier from these data.

If source/association identity is invalid, report NOT_EVALUABLE with its cause.
If legal inputs cannot constrain the full target, report
NOT_EVALUABLE_WHOLE_TARGET_ORACLE and stop before a geometric classifier. Do not
replace the missing constraint with evaluator 3D truth or silently downgrade the
requested question to returned-point classification. A usable complete contract
would permit the one planned geometry check; no automatic alternative follows
failure. This is a feasibility disposition, not a negative method result.

CPU mask/metadata reductions are TASK_NOT_GPU_SUITABLE; no model or paid worker.
Preserve the original A/B/R/U/G, UNKNOWN, old negative controls and consumed-data
status. Save source/protocol/code hashes, coverage, evaluation joins and supported
registration receipts under artifacts.local/work/ba-ideal-association-preflight-20260921/.
