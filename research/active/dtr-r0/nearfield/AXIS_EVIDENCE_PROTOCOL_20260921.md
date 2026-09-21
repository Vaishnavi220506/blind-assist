# Public ToF distance and lateral evidence audit

2026-09-21, EXPLORE on consumed same-simulator Development. User authorized
an existing-observation audit after the current-only result. No fitting, model
inference, threshold selection, conditional hold, new data or protected-test
access. Complete this diagnostic and delivery, then stop.

Question: do the existing public ToF supports distinguish high-score negative
frames from positive frames suppressed by the frozen current-only cutoffs?
Audit all 16 transfer layouts / 1152 frames, retaining the prior SELECTION
g00/g01 and EVALUATION g02/g03 roles. Include complete INSIDE/BOUNDARY/OUTSIDE
clips and negative periods. All evidence is consumed Development.

Public extraction accepts only saved `observations.npz` ranges/boxes and frame
IDs. These ranges are axial simulated single-return proxies, not radial hardware
measurements. Validity is finite and 0.1 <= r < 8. Preserve the frozen full
interval [max(.1,r-radius),r+radius], radius=.1+3*(.01+.02*r), and full zone
footprint. Use existing tof_readout, support_score and lateral_relation without
changing constants. Preserve d=depth-overlap fraction, o=conditional joint
horizontal/vertical overlap, and joint=d*o; o is undefined when d=0 and is not
a purely horizontal score. None of these fractions is a probability.

Per zone record missing, observed point, full interval, distance interval state
(contained / partial overlap / disjoint), full horizontal support relation,
possible/definite corridor support, d/o/joint. Exact contact remains possible
even if overlap measure is zero. Per frame record all 64 validity positions,
valid count, minimum range, counts of these states, maximum d/joint, and fixed
evidence-presence flags: any possible-depth interval; any contained-depth
interval; any contained-depth interval whose full support can intersect the
corridor. These are descriptive witnesses, not new alerts or target ownership.
Outside-only or missing returns do not establish clear space. No target, layout,
phase, score, truth or native contributor selects the zones or features.

Seal all 1152 public descriptors before reading evaluator bounds/labels/native
values. Then reproduce frozen A score, validity, definite count and UNKNOWN;
join saved original/uniform/balanced scores/flags without changing them. Verify
actual rendered full target bounds against saved truth. Stratify exact target
extent by distance outside [.3,3], distance-valid horizontal outside [-.3,.3],
vertical-only outside [-.2,.9], or positive. Preserve independent axis flags and
do not relabel near-boundary samples. Native counts are evaluator-only lineage
attribution, never observable range or proof of ownership of a particular zone.

Report full-role and per-layout cross-tabs, fixed witness presence/absence,
missingness versus baseline UNKNOWN, and score/descriptor distributions. Primary
subsets: original fixed-high (7.6612162590026855) added negatives and Boundary
positives rescued by that original supplement but excluded by each frozen
current-only readout. Include all positives and all negative classes as controls.
Report native-backed positives separately. Describe known f0676/f0465 examples
only alongside the complete cohort; do not choose a new cutoff from them.

A distance qualification is supported only as a prospective research direction
if a prespecified witness excludes all high-score distance negatives on BOTH
roles while retaining all native-backed suppressed Boundary positives, with
retained positives spanning >=4/8 layouts per role. This diagnostic criterion
does not certify target ownership, a deployable gate, or lateral specificity.
Otherwise record the measured rejection/retention tradeoff and stop. Overlap of
these summaries does not prove that the full 64-zone input has no information.

Use CPU for small saved-array/geometry reductions (TASK_NOT_GPU_SUITABLE).
Output `artifacts.local/work/ba-axis-evidence-20260921/`; preserve sources and
the demo. Independently recount descriptors/subsets/seals, record local
disposition and supported registration receipts, then scoped normal Git delivery.
