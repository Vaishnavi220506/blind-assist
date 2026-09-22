# Fixed half-full / half-target loss: five gates pass, local coverage remains short

One authorized 512-update joint fit is complete. On the same 32 TRAIN frames,
target recall 98.75% / IoU 66.72% and full-image retention meet the sealed gates.
Only 11/16 positive zones meet their individual criteria, below 14/16. Five of
six gates pass; the all-pass condition fails. **Validation was not run**.
Retain the joint-supervision diagnostic contribution, preserve original NFO,
and close this fixed supervision-allocation sequence without ratio tuning,
additional steps or an automatic successor.

| Same 32 TRAIN inputs, 2m at 0.081 | Original NFO | Full-only fit | Target-only fit | Fixed joint fit |
|---|---:|---:|---:|---:|
| Target-positive recall | 71.25% | 97.50% | 96.00% | 98.75% |
| Target-positive IoU | 13.62% | 25.83% | 60.00% | 66.72% |
| Target-positive TP / FP / FN | 285 /1693 /115 | 390 /1110 /10 | 384 /240 /16 | 395 /192 /5 |
| Positive zones meeting recall 90% and IoU 50% | 2/16 | 1/16 | 11/16 | 11/16 |
| Target-negative FP /7689 known pixels | 0 | 13 | 16 | 1 |
| Full-image recall | 97.07% | 99.40% | 58.80% | 98.89% |
| Full-image IoU | 61.81% | 80.93% | 29.34% | 74.94% |
| All pure-far FP | 21,214 | 4,937 | 70,192 | 16,645 |

Target-positive denominator remains 400 near / 7,531 far pixels. Joint target
precision is 67.29%; negative-control FPR is 0.0130%. The six gates are target
recall>=95%, IoU>=65%, negative FPR<=1%, at least 14/16 positive zones with
recall>=90% and IoU>=50%, full-image recall>=original NFO on these 32 frames,
and all-pure-far FP<=original NFO on these 32 frames. Only the 14/16 gate fails.
The global guards compare original NFO, not the stronger full-only fit.

## Remaining five zones

| Scene / camera / frame / zone | Near pixels | TP / FP / FN | Recall | IoU | Failed condition |
|---|---:|---:|---:|---:|---|
| ai_003_004 /01 /0056 /27 | 7 | 7 /15 /0 | 100% | 31.82% | Excess FP |
| ai_050_004 /04 /0065 /46 | 9 | 9 /23 /0 | 100% | 28.12% | Excess FP |
| ai_048_010 /02 /0044 /55 | 4 | 0 /0 /4 | 0% | 0% | Entire target missed |
| ai_048_009 /00 /0037 /27 | 5 | 4 /0 /1 | 80% | 80% | Recall below 90% |
| ai_015_009 /00 /0057 /49 | 35 | 35 /50 /0 | 100% | 41.18% | Excess FP |

The failures include both overprediction and missed near support. Four targets
have at most 9 near pixels, but one 35-pixel target also fails. Do not describe
the residual as exclusively subpixel loss, exclusively boundary spread, or
proof that the input/architecture cannot represent these targets.

## Frozen contrast and conditional validation

References: [full-image fit](BA_NFO_HARDFIT_20260919.md) and
[target-only fit](BA_NFO_TARGETFIT_20260919.md).
Reuse the exact 32 TRAIN identities, selected zone boxes, full 256x192 RGB,
actual public 8x8 ToF, architecture and all four heads. Restart from original
NFO, not either specialist checkpoint. Initialization outputs match the
original 32-frame baseline bit-for-bit. Copy the same 512 batch plan and verify
its hash. Seed 190921, batch 8, AdamW .002/.0001, constant LR, gradient clip 5,
512 updates and final-step checkpoint all remain fixed. Cutoff stays 0.081.

Loss is **0.5 L_full + 0.5 L_target**, where each L uses the unchanged
four-head BCE+.2ordinal formula and its own known-pixel mean. Full supervises
all 1,558,122 known image pixels; target supervises 15,620 known pixels in the
selected complete zones, including foreground/background and all 16 negative
controls. Unknown labels remain excluded. Target boxes/depth truth are used
only by loss/evaluation; network forward sees full RGB and public ToF only.
No ratio, threshold or checkpoint is selected from the outcome.

The protocol permits one fixed original 500 validation-frame check only after
all six training gates pass. That prerequisite failed. The run wrote
`validation-status.json: SKIPPED_TRAIN_GATE_NOT_MET, frames_read=0`; no
validation predictions/results were created and no test frames were evaluated.
The unexecuted validation branch cannot supply transfer or implementation-test
evidence. Training gains are not generalization or fresh confirmation.

## Interpretation and disposition

The fixed joint loss retains much of target-only localization while avoiding
its whole-image collapse. Relative to original NFO, target and whole-image
metrics improve on these seen frames. This supports a useful contribution
from supervision allocation under the tested budget and initialization.
It does not establish uniform fitting: five individual targets still fail.

The joint model does not dominate the full-only fit on every metric: full
recall 98.89% is below 99.40%, full IoU 74.94% below 80.93%, and pure-far FP 16,645
above 4,937. Preserve these costs alongside its much better target IoU.
Retain as COMPONENT diagnostic evidence, not an App/default replacement or
successful final method. End this sequence without relaxing 14/16, evaluating
validation anyway, trying 0.3/0.7, or extending the 512-step budget.

## Verification and retained evidence

[Implementation](ba_nfo_jointfit.py); [two tests](test_ba_nfo_jointfit.py).
Tests verify the half-weighted loss/gradient decomposition, retained outside
supervision, UNKNOWN exclusion and identity when both supports span the full
known domain. Both pass. An independent process reloads final weights and
feeds only RGB/ToF: all 32 score arrays reproduce bit-for-bit, all frame/domain
metrics match, target counts are independently recomputed, selection/batch
hashes match, 512 loss logs satisfy the fixed mixture, original weights remain
unchanged, and five failed zones/validation skip are independently checked.

Actual backend is CUDA NVIDIA GeForce RTX 5060 Laptop GPU. The fit took 17.30s,
not phone inference latency. Fit and verification exited; no worker or further
training remains running. Evidence root:
`artifacts.local/work/ba-nfo-jointfit32-20260919/` retains protocol, selection,
batch plan, baseline, actual checkpoint, 512 step losses, scores, results,
failed-zone table, previews, verification and skip/disposition receipts.

Global registration still reports the existing ledger 303 input-fingerprint
mismatch; global inheritance remains pending. Local structured disposition
and actual command receipts preserve this result without a ledger bypass.
