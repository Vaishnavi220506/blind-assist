# Public returned-evidence positive readout

EXPLORE, consumed grouped Development, user-authorized implementation following
the tri-state diagnostic. Train one small public-input return classifier and
OR its positive output with unchanged A. No veto, no extra oracle combinations,
no DA-V2, RGB encoder, capture, sensor changes or App-default changes.

## Reuse and changed mechanism

Reuse MZ161/MZ174's public 132 x 21 token contract and MZ171's per-return
sampled-corridor-point target builder. A token-only adapter removes unused
dense RGB-map computation and must reproduce TRAIN192 tokens bitwise. Preserve
MERGED envelopes, outside-RGB slots and public missingness. No native points,
lineage, actor identity, family, scene, frame ID or ground-truth extent enters
the model. Radar stays in A; its four token labels remain UNKNOWN and the new
pointwise head emits evidence only for the 128 ToF slots.

For each ToF token concatenate nine deterministic public spatial features:
three per-axis signed nominal-center margins, three possible-overlap margins
of the public support envelope, and three full-containment margins. Use the
inherited nominal corridor [0.2,3.6] x [-0.3,0.3] x [0.4,2.05] metres.
These are hypotheses derived from public range/zone/IMU, not actual hit points
or proof of free space. They add explicit geometry, not new sensor information.
Standardize all 30 values using fitting valid ToF tokens only, std floor 0.01.

One shared 30 -> 32 -> 16 -> 1 MLP, ReLU, 1,537 parameters, no graph/messages.
Invalid slots cannot produce evidence; zero usable ToF returns force an inactive
branch, retaining A. Frame evidence is the maximum valid ToF logit. A's threshold
remains 0.3917890013717321; final alert = A OR (valid_return AND score >= tau).
The final policy cannot delete or delay an A alert, but may add false alerts.

Return labels: any resolved sampled contributor in the nominal corridor means
positive; fully resolved nonempty outside lineage means negative for that
return only. Unresolved/absent returns receive no loss. No frame-positive
propagation to background returns. Equal total known-positive and known-negative
loss mass per fitting partition, inherited from MZ174. AdamW lr=0.001,
weight_decay=0.0001, batch16, 120 epochs, fixed seed 183017 + fold; use the last
checkpoint. No architecture, seed or epoch search. Run the fixed schedule to
completion without an arbitrary wall-clock cutoff; log progress and failures.

## Existing data and honest comparison

Use original MZ136 TRAIN192 as fitting anchors, with its original test48
excluded. MZ170288 and changed-background S1-confirmation288 are already consumed
and become explicitly grouped Development. Neither is fresh confirmation.
Original dev48 is not reused for selecting this branch: A has no misses there,
so it cannot expose the desired recall contribution. No additional capture.

Use six fixed outer folds over scene index 0..5 across both 288-frame cohorts.
For fold k, report all scene-k groups (96 frames: 48 per cohort), calibrate on
scene-(k+1 mod6) groups (96 frames), and fit on the other four indices plus the
original TRAIN192 (576 fitting frames). Group identity includes cohort/family
and scene; keep both paired trajectories and all their frames together. Group
and index metadata are only split controls. Every reported frame is excluded
from its own model fitting, normalization and threshold selection. Calibration
groups are also excluded from that fold's fitting. A was historically fit on
the anchor192 only. Shared generator templates remain a limitation; this is
not a leave-generator-out or fresh-domain generalization claim.

Choose exactly one tau per fold on that fold's calibration data by final OR
F1 on 5 cm clear frames, then fewer FP, then higher tau. Include an inactive
branch candidate. Candidate thresholds come from calibration scores only.
Allow extra FP when the final metric improves; no no-new-FP acceptance gate.
Seal weights/normalization/tau before scoring held-group outcomes. Offline
targets may already have been decoded for other folds, but held-group labels
do not enter their own fold's fitting/calibration path.
Do not pick the best outer fold or pool outer outcomes to retune tau. Report
pooled out-of-group predictions and fold variation; no single final all-data
deployment model or unbiased fresh confirmation is implied.

## Required evidence and stopping point

Main table per cohort: A, A + public positive evidence, existing sampled-support
OR oracle reference. Report clear P/R/F1, TP/FP/FN and 75% coverage; strict and
boundary controls, rescued FN/new FP by family and fold, core events/first alert,
negative alert burden, zero-return fallback and A-alert retention. Return metrics
and fit loss diagnose the branch but do not replace final-alert evaluation.
For added warnings report whether any sampled corridor witness actually exists.

Authenticate input/cache/model hashes. Separate public token cache from native
target files. Test token parity and ignored-metadata invariance, invalid/Radar
masking, OR monotonicity and grouped exclusions. Independently audit held-group
predictions and calibration choices from saved outputs. Benchmark equivalent
CPU/CUDA forward/backward, record selected backend and actual training times;
any head/frontend timings exclude A and are not phone/end-to-end latency.

Keep a useful public OR tradeoff as a Development component, with all extra FP
shown. A null result closes this fixed fit/readout, not all spatial fusion.
Observed implementation or optimization defects may be repaired transparently;
do not rescue outer results by tuning. Finish code, report, validation and scoped
delivery after this one recipe. RGB/context, suppression and further models stay
separate future choices.
