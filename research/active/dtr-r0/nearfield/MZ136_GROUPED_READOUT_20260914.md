# MZ136 grouped readout stability check

2026-09-14, user-authorized continuation of the
[training-fit and frozen-transfer work](MZ136_TRAIN_FIT_REPAIR_20260914.md).
Fixed RGB +8x8 ToF +Radar +IMU; existing data only. No new capture or original-test
inference. Phase: disclosed consumed Development.

## Outcome

Stronger regularization and a paired midpoint penalty reduce readout instability,
but the selected candidate still fails the alert objective. Keep MZ129. Do not
promote this correction or present relative improvement over the unstable head
as superiority over the unmodified network.

| Readout diagnostic on192 original TRAIN frames | Correct | Both correct /96 | BCE | Mean absolute pair midpoint |
| --- | ---: | ---: | ---: | ---: |
| Original linear readout, held scenes excluded from head fit | 122/192 (63.54%) | 26 | 6.2042 | 19.7931 |
| Stronger L2, same exclusions | 140/192 (72.92%) | 44 | 0.9785 | 3.2660 |
| Stronger L2 +midpoint penalty, same exclusions | 152/192 (79.17%) | 56 | 0.5945 | 2.8327 |
| Unmodified frozen BCE network, reference only | 154/192 (80.21%) | 60 | 0.3630 | 0.9449 |

**These are readout-only cross-fitting diagnostics. The fixed backbone already
saw all192 TRAIN frames. They are not independent cross-validation of the whole
network.** The unmodified network reference was added after selection and has
seen all192 labels during its own training; its numbers are context, not a fair
independent generalization comparison. It is nevertheless a necessary reminder
that the regularized correction has not demonstrated superiority over the
uncorrected model. No threshold was searched in this grouped diagnostic.

Metadata clarification: the original run freeze used `dev_access=false` and
`test_access=false`. The shared loader parses the complete manifest/evaluator
before selecting TRAIN records, so these names were too broad. No dev/test
outcomes entered fitting or selection, and no new test inference ran. A separate
metadata correction records that distinction; future runner fields use the
precise outcome-use wording. Original frozen bytes and model are preserved.

## Change and selection

Four folds each exclude one complete scene per family (48frames) from the new
head fit; the other144 frames fit that head. Both intervention members and all
frames from a scene stay together. Normalization mean/std are computed only on
the fold's144 fitting frames, with the prior0.001 std floor.

The three recipes were fixed before fold outcomes: BCE +L2 weight0.0001;
BCE +L2 weight0.01; and BCE +L2 weight0.01 +midpoint penalty weight0.1. All use
the same385-parameter linear residual and LBFGS(max200,strong-Wolfe) from zero
head weights. For an opposite-label pair(a,b), the midpoint term is
`mean(((score[a]+score[b])/2)^2)`: it penalizes a common shift in the two **final**
logits. A ranking hinge can leave that common shift unconstrained. This loss
uses pairs only during fitting; inference does not require a counterpart,
scene/family label, native geometry, or future frame.

Predeclared selection used minimum pooled out-of-fold BCE, with simpler recipes
preferred on exact ties. A final refit was allowed only if selected BCE improved
at least10% over the plain recipe without lower accuracy or both-correct rate.
The midpoint recipe passed this screen and was refit once on all192 TRAIN
frames, before any new dev scoring. That final refit gives178/192 correct
(92.71%),82/96 both correct,85TP/3FP/11FN. It **does not** satisfy the earlier
95%/90% training-fit criterion; regularization trades that fit for stability.
The original191/192 training-fit checkpoint remains unchanged.

## Actual alert transfer

The selected checkpoint was then frozen and checked on the previously consumed
48 dev frames, using the existing exact-threshold recall/timing rule.

| Dev48 (24positive/24negative) | TP | FP | FN | Detected events /5 | False alert duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| MZ129 | 24 | 13 | 0 | 5 | 3.25s |
| Previous unregularized readout, zero logit | 12 | 6 | 12 | 2 | 1.50s |
| Selected midpoint readout, zero logit | 18 | 0 | 6 | 4 | 0s |
| Selected midpoint readout, retained recall/timing | 24 | 22 | 0 | 5 | 5.50s |

Exact dev-selected logit threshold: -4.466347694396973. Baseline and selected
candidate detect all five events at onset. The candidate therefore adds9FP and
2.25s false-alert time at matched recall/timing. The required20% nuisance
reduction fails; original-test and shifted-test inference were not run.

At zero logit, five of the six misses are shallow-boundary frames and one is a
suspended-head frame; body and rod families have neither misses nor FP. The
original frozen BCE also had18TP/0FP/6FN on dev, so these aggregate counts are
not an improvement over that simpler model. All24 paired orderings are correct,
but only18/24 pairs have both decisions correct at zero logit, falling to2/24
at the selected common threshold. Better ordering alone still does not solve
the absolute boundary decision across scenes.

The intervention demonstrates a tractable reduction in new-head instability.
It does not establish that background alone caused the failure, that existing
sensor information is insufficient, or that the revised loss improves alerts.
The next decision should concern better observable spatial features or training
support for shallow-boundary decisions; another automatic loss tweak on this
same pooled linear readout is not justified by these results.

## Evidence, checks, and delivery limits

Canonical root: `artifacts.local/work/mz136-corridor-pair-20260914/`.
`grouped-readout-v1` contains the pre-run freeze/source snapshot, feature matrix,
fold IDs, all three OOF score arrays, selection, optimizer placement, and the
selected model. `unchanged-base-reference.json` is an explicitly posthoc
reference, added without modifying the original run receipt. The selected-fit
receipt and `grouped-readout-transfer-v1` preserve the frozen dev evaluation.

Selected model SHA256:
`7f7bf7d82227ebc7739429d23192309ea94e74417f97f314168adee2d80104cc`.
Cached-feature fitting and normal observation-only inference agree within
2e-4 absolute /1e-4 relative tolerance. Original base scores and source receipts
were checked. Focused tests verify complete scene/pair folds, exclusion of held
features from normalization and optimization, and midpoint gradients. All24
MZ136 tests pass.

Torch2.11.0+cu130, RTX5060 Laptop GPU. Feature extraction took0.893s. One actual
equivalent fold optimizer probe took CPU2.251s versus CUDA0.676s, selecting CUDA.
The three four-fold recipes took6.323s,0.918s,1.316s respectively, excluding
feature extraction and the backend probe. No task-owned background worker was
created. Training/model weights are not changed during transfer evaluation.

Intended terminal: `MZ136_GROUPED_READOUT_NO_JOINT_ALERT_GAIN`, inheritance
`NEGATIVE_CONTROL` scoped to this exact fixed-feature correction/selection.
The prior fitting diagnostic remains available. Supported experiment
registration is still blocked by the pre-existing ledger303 input-fingerprint
mismatch. Pending metadata and the actual attempt are saved without rewriting
or bypassing the ledger.
