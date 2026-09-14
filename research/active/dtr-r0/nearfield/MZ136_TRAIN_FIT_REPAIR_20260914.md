# MZ136 training fit repair

Date: 2026-09-14. User-authorized continuation after the failed
[MZ136 comparison](MZ136_RESULTS_20260914.md). Scope: diagnose and repair fitting
on the existing 192 TRAIN frames only. No new capture, dev/test inference,
threshold search, or replacement of the original negative result.

## Result

The original fixed-zero-logit fit criterion is now met: accuracy >=95% and
both members correct in >=90% of pairs. The retained development component gets
191/192 frames and 95/96 pairs correct; all 96 pair orderings are correct.
It uses the original BCE checkpoint with its encoders frozen, plus a direct
385-parameter RGB-feature residual. MZ129 remains the alert baseline.

| Same 192 TRAIN frames, zero logit threshold | TP | FP | FN | TN | Accuracy | Both correct /96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original MZ136 BCE | 64 | 6 | 32 | 90 | 80.21% | 60 |
| Optimizer-only continuation | 75 | 14 | 21 | 82 | 81.77% | 65 |
| Ordered horizontal-column residual | 75 | 11 | 21 | 85 | 83.33% | 66 |
| Frozen features + standardized direct readout | 95 | 0 | 1 | 96 | **99.48%** | **95** |

| Family, 48 TRAIN frames each | Original errors | Direct-readout errors |
| --- | ---: | ---: |
| Near rod / far wall | 15 | 0 |
| Shallow boundary crossing | 16 | 1 |
| Substantial body obstacle | 2 | 0 |
| Suspended head obstacle | 5 | 0 |

These are training-fit numbers on reused controlled simulation data. They do not
establish a useful unseen-scene alert threshold, reduced real-world nuisance,
robustness, or a benefit from the pair-ranking loss. No such evaluation was run
in this continuation. The remaining boundary miss is retained in `cases.json`.

## What was diagnosed and changed

The training-only sampling audit found that only 2/38 original errors have target
native AABBs entirely outside the core sampling receptive fields. All 16 boundary
errors have sampled target support. Boundary paired feature RMS has median
0.007054 before pooling and 0.001514 after pooling. This is attenuation, not proof
of lost information; the original shallow RGB encoder was trained from random
initialization, not a pretrained backbone.

Two bounded checks kept the old decision rule and zero threshold. First, 3,600
additional balanced-pair steps without brightness augmentation used cosine LR
0.0003 -> 0.00003, AdamW decay 0.0001. Second, the same schedule from the same
original BCE checkpoint added a zero-initialized linear residual over seven
ordered horizontal RGB columns. Neither met the original fit criterion. Minimum
full-TRAIN BCE selected their saved checkpoint; their complete trajectories and
original failed outputs are preserved. These are diagnostic interventions, not
a controlled contribution claim for the final convex readout.

The successful change takes the **existing** mean/max RGB feature vector from
each of the eight corridor cells and concatenates them in their fixed cell order
(8 x48 =384). Each coordinate is centered and scaled using TRAIN-only mean/std,
with a 0.001 standard-deviation floor. A single linear layer adds a scalar
residual to the frozen original corridor score. The original ToF, Radar and IMU
paths remain inside that base score; no evaluator geometry or pair/scene/family
ID becomes an inference input.

All original 11,305 parameters are frozen. Only 384 weights and one bias are
optimized, for 11,690 total parameters. Full-batch LBFGS, max 200 iterations,
strong-Wolfe line search minimizes BCE +0.0001 times the sum of squared readout
weights. The single final result used 117 objective evaluations; there is no
checkpoint or threshold search. Scaling constants are stored buffers and never
recomputed from an inference batch. Pair IDs only define evaluation grouping;
this repair uses ordinary binary supervision and no pair-ranking objective.

The result refutes “these saved features cannot fit these examples.” It identifies
the old feature scaling/readout/optimization combination as a tractable fitting
bottleneck. It does **not** isolate their individual causal contributions, and
does not show that image pooling alone caused the failure: the successful head
uses the already-pooled features. Memorization or simulator-specific reliance
remains possible and must be separated from future generalization claims.

## Evidence and validation

Canonical artifact root:
`artifacts.local/work/mz136-corridor-pair-20260914/`.

- `train-fit-repair-v1`: optimizer-only freeze, scores, checkpoint and trajectory.
- `train-position-repair-v1`: horizontal-column intervention and measured backend.
- `train-representation-audit-v1`: all TRAIN frame/pair support diagnostics,
  source/checkpoint hashes, and the same-time boundary RGB/sampling figure.
- `train-direct-readout-v1`: source snapshot, freeze, trained `model.pt`, all
  192 before/after scores and cases, recovered summary, backend and hash receipt.
- `train-fit-comparison.html`: offline same-case before/after gallery.

The original BCE scores were reproduced before each fit. After direct-readout
training, cached-feature scores agree with the normal observation-only inference
path. A fresh process then reloaded the saved checkpoint and reproduced all 192
scores within 1e-4; it verified exact TRAIN membership and source receipt hashes.
Model SHA256:
`f6abcad0d2aa37e25ad73159ec729740e5d0fd3b593c0ec43b8d6439228d0cc0`.

Focused tests check zero-residual equivalence, horizontal-order sensitivity,
inference-batch independence, and JSON serialization of successful/failed fit
reports. Original MZ136 checks remain applicable; no Android behavior changes.

Actual execution: Torch 2.11.0+cu130 on RTX 5060 Laptop GPU. The original-network
and horizontal-column diagnostic fits took 96.36s and 96.24s respectively.
Reloaded direct-readout inference over 192 frames took 0.735s. Equivalent batch-8
median inference was CPU 0.07952s vs CUDA 0.00488s. The direct fit completed, but
its training duration was not persisted before a report serialization error;
that duration is unknown and is not reconstructed from unrelated wall times.

Mechanical recovery: a NumPy boolean in the successful fit report caused JSON
serialization to fail **after** the model, scores and case records were saved.
The helper now converts the report's rate/boolean to native Python types. The
report was recovered from the saved model/scores with a fresh inference replay;
optimization was not rerun. Original source snapshots remain intact. The first
optimizer diagnostic's freeze also contained a nonexistent backend filename;
the actual reused record is `full-fits-v1/backend.json`, and subsequent runner
versions validate that path and record its hash.

## Decision and next boundary

Retain the direct readout as a **TRAIN-fit development component**. The fitting
check is complete; do not repeat it just to eliminate the last training error.
The next decision, under a separately scoped validation, is whether this frozen
component improves unseen-scene nuisance at matched recall and reminder timing.
Consumed MZ136 dev/test outcomes cannot become fresh independent confirmation.
No new data collection or alert promotion is implied by this fit repair.

The original MZ136 BCE/hinge joint-alert negative control is unchanged. Intended
inheritance for this separate fit component is `COMPONENT_OR_CHALLENGER`, confined
to disclosed Development. Supported registration remains pending the existing
`experiments/index.jsonl:303 input_fingerprint` mismatch; no ledger was bypassed
or rewritten. Pending metadata and the actual registration attempt are saved
alongside these artifacts.
