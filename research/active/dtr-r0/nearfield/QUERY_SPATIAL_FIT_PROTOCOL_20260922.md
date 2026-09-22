# Query-conditioned spatial fitting check

Status: fixed before fitting; EXPLORE, consumed training data only.
Authorization: user `推进` following the frozen-checkpoint diagnosis.

The selected earlier models had invariant six-query ordering and zero training
mask IoU. That does not identify the cause or rule out other models. This check
asks whether an explicit image-coordinate carrier for metric queries can learn
useful query-dependent localization on a tiny training subset. It is a fitting
test, not generalization, alert improvement, or a retry of the old held gate.

## Inputs and pairing

Reuse only the existing query-occupancy public RGB and regional 8x8 ToF inputs,
the local ImageNet initialization, and `labels/train.npz`. Select the first
lexicographic training base group in each of the four `type_id` families using
identity metadata, all three layout relations, and frame indices
`[0,2,3,4,5,6,7]`: 84 images, 504 queries. Selection never uses predictions or
labels. Do not infer or train on other image rows; do not open dev/held labels.
Whole-container hashes are integrity checks, not held-row model evaluation.
Run through `tools/ba.ps1 run research-ue` with admitted consumed input roles.

Both arms share the original trainable encoder, regional fusion, global FiLM,
decoder, first-hit and mask heads, and exactly the same initial state and batches.
`global` bypasses the added branch and is the original forward function.
`spatial` adds a zero-initialized 1x1 projection of 24 query/ray channels before
the decoder. Six fixed axial depth hypotheses `[.525,1,1.5,2,2.5,2.875]` multiply
camera rays to form hypothetical X/Y; signed distances to each query's four
faces are divided by its width/height and clipped to [-1,1]. These are geometry
hypotheses, not observed pixel depth. No native depth, actor geometry, identity,
relation, frame index, labels, baseline decision or previous frame enters forward.
Both state dictionaries contain the extra 768 weights; only spatial uses them.
Thus this tests the spatial-branch package, not a strict equal-effective-capacity
proof. No hard oracle mask is applied to predictions.

## Fixed fitting opportunity

Fresh shared initialization, seed 202609224. One fit per arm: 100 epochs,
batch 12, exactly 700 updates, AdamW weight decay 1e-4, fixed head LR 1e-3 and
encoder LR 1e-4. All encoder blocks train. Final state only, no best-epoch choice.
The same deliberately more fitting-friendly objective is used in both arms:
unweighted valid-query distance cross entropy plus per-query foreground/background
balanced mask BCE plus Dice over visible positive queries. For a positive map,
BCE is half mean positive-area log loss and half mean negative-area log loss;
an empty map gets negative-area mean alone. Invalid area is excluded. Dice uses
the native fractional foreground area, coverage, and smoothing 1. This shared
loss/budget differs from the old recipe and cannot retrospectively attribute its
failure to architecture alone. No search, threshold selection or retry on results.

Compare actual CPU/CUDA training-step timings on cloned models and this loss;
record observed device. Probes cannot alter the paired initial state. Seal input,
source, initialization, batch schedule, checkpoints and final predictions.

Before fitting, enumerate all 720 constant query permutations on this fixed
training cohort, and compute the optimal binary coarse-grid IoU from fractional
area labels. If a constant order already reaches 0.95 pair accuracy, or macro
oracle IoU cannot reach 0.50, stop as NOT_EVALUABLE without changing the cohort.
These label-only feasibility diagnostics do not enter the predictor.

## Decision and stop

At fixed probability/mask threshold 0.5, useful tiny-train fitting requires all:
query precision >=0.95, recall >=0.95; macro IoU of visible-positive query masks
>=0.50; same-image occupied-over-empty query pair win rate >=0.95 (ties half).
Report raw TP/FP/FN, precision/recall/FPR, lateral and height pair counts, all and
positive distance-bin accuracy, empty-query mask false area, per-family counts,
query-order variation and unchanged sensor UNKNOWN. These correlated pixels,
queries and frames are descriptive; four training groups are not independent
generalization trials. Distance-bin accuracy is secondary, not measured range.

Spatial passes and global fails: retain spatial package as a fitting COMPONENT,
with no claim that extra geometry rather than extra capacity caused the effect.
Both pass: explicit geometry is not shown necessary; old optimization remains
a plausible contributor. Neither passes: this bounded fitting package is negative,
not proof that spatial conditioning is impossible. Global alone passes: reject
this added spatial branch. A gate pass is never an alert-model promotion.

Stop after these two final-state fits and the associated audit, report and
registration. Preserve old negatives, A, UNKNOWN and all consumed evidence.
No dev/held evaluation, video fusion, new capture, threshold tuning or automatic
successor. Mechanical corrections retain failed receipts and cannot depend on
method outcome. Task-owned compute ends when the fits finish.
