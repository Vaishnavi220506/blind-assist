# Frozen lateral-pair response diagnostic

2026-09-21. EXPLORE, consumed Development only. Reuse the sealed16-group,
1152-frame spatial supplement transfer. No model execution, training, threshold
selection, capture, test access, or alert policy change.

Question: does the fixed classifier lower its score when the same obstacle moves
outside the corridor, and are failures associated with motion inside a ToF zone?
The prior transfer establishes complementary recall with17 added OUTSIDE current
FP in three groups; it does not establish erroneous depth propagation or a cause.

Pair all frames by base group and frame index:384 INSIDE/BOUNDARY/OUTSIDE triplets.
Before scoring, verify identical actual camera, time, target dimensions/material,
height/forward position, backdrop and nominal noise key, with only target lateral
position changed. Equal optical-axis depth is not equal radial range. The proxy's
conditional random draws can differ despite a shared seed. Occlusion, visible
background and shadows can change as consequences of moving the target.

Primary comparisons use all triplets where INSIDE and BOUNDARY are positive and
OUTSIDE negative; other frames remain a separately reported negative-depth control.
Compare INSIDE versus OUTSIDE and BOUNDARY versus OUTSIDE separately. Report score
drops, ties/inversions, fixed-high-threshold discrimination, all-group counts,
three known failed groups, native support, and the unchanged event/FP/UNKNOWN
context. Include INSIDE versus BOUNDARY as a descriptive positive-positive contrast.
No significance claim treats adjacent frames as independent layouts.

Evaluator geometry may only stratify these already sealed scores. Full target
footprint is ray-AABB intersection on the exact native-point-sampled192x256
lattice, aggregated into the existing64 ToF boxes; it is geometric coverage,
not visible/native-return ownership. Primary descriptor: identical nonempty
zone sets versus changed zone sets (empty coverage separate). Supplemental:
the front-face corridor-facing vertical edge projects to the same ToF column
versus another column/outside FoV. This edge is not a full-extent decision rule.
Also report changed validity cells and common-valid range differences: identical
geometric footprint never implies identical measured input. No arbitrary masks,
synthetic returns, or feature interventions.

Freeze rules and code before producing new paired aggregates. Verify original
source/prediction/evaluation seals, preserve them, and record input/output hashes.
Use CPU for this small saved-array/geometry reduction (TASK_NOT_GPU_SUITABLE).
Export one paired record table and per-group summaries plus score trajectories
for all16 groups. Choose image triplets deterministically: earliest extra OUTSIDE
current FP per failed group and one first-alphabetical error-free group at its
first true BOUNDARY detection; no manual best-example selection.

Decision: consistent inversions in matched pairs support a spatial discrimination
failure. Association with same-footprint/edge strata can prioritize a position
representation experiment, but is not proof of feature loss, return ambiguity,
or physical nonidentifiability. If scores generally fall but remain above the
fixed threshold, distinguish insufficient margin from absence of response; do
not retune. Small/confounded strata leave the mechanism NOT_EVALUABLE. Stop after
this diagnosis and delivery, even if a successor looks promising.
