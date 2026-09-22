# Frozen-input spatial feature mediation

2026-09-21 EXPLORE, newly authorized continuation. One diagnostic of the saved
1152-frame appearance feature cache, with the same frozen B/N weights and
cutoffs. No capture, image editing, feature re-encoding, fitting or policy search.
The previous pure-material source terminal remains NOT_EVALUABLE.

Question: does intervention-associated score drift enter the frozen heads mostly
through the central four feature-grid columns or the complementary outer four?
The model has 216 RGB-derived and four public sensor channels on an 8x8 grid.
Fix center columns2:6 and outer columns0:2,6:8 before this diagnostic. These
are network feature locations, not object masks or physically isolated pixels:
encoder receptive fields may overlap either side. The partition is not fitted
to truth, score changes or model weights.

For all288 paired poses and repeat/target/background, evaluate four combinations:
reference, changed, only central RGB entries changed, only outer RGB entries
changed. Keep all sensor channels exactly reference; public ranges are already
pairwise identical. Reference/changed logits must replay the original frozen
predictions within2e-5 and reproduce every threshold/current decision, otherwise
stop as replay failure. No tolerance increase or head replacement.

Compute the exact two-part allocation:
center=.5*((center_only-reference)+(changed-outer_only)); outer analogously.
Their sum must reproduce total drift within floating arithmetic tolerance.
Also report nonlinear interaction changed-center_only-outer_only+reference.
These quantify network-input perturbations, not pure-material physical causality.
Report signed and absolute allocated changes, current flips, family counts and
all source-admission failures; do not omit failed native pairs. No independence
or calibrated probability is inferred from correlated frame counts.

Retain localization only if an allocation is consistent across families/models;
mixed or interacting effects argue against one blanket spatial suppression.
After predictions are sealed, evaluator may count gains/losses as consumed
diagnostics, never select a deployable hybrid policy: an online device does not
have the same-scene reference appearance. Preserve all original seals and results.
End after this one input-location contrast, validation and reporting.

Reuse the previous worker head backend measurements only after checking unchanged
model/runtime/device and batch64. Save runtime, cache/checkpoint/code hashes and
prediction/evaluation seals in a new task-owned artifact tree. Scientific run
budget120 seconds; no UE allocation or paid service. Release process on exit.
