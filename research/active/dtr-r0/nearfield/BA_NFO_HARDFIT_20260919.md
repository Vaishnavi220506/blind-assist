# NFO hard-case fit: useful learning, incomplete localization

One authorized TRAIN-only32-frame fit is complete. Retain original NFO as the
baseline and this check as diagnostic evidence. The unchanged model learns
substantial recovery while reducing foreground-zone FP, but fails the sealed
localization and per-zone coverage criteria. No second fit or cutoff change.

| 2m metric, same selected TRAIN inputs | Original NFO | After512 updates | Diagnostic target |
|---|---:|---:|---:|
| Target-positive recall | 71.25% | 97.50% | >=95% |
| Target-positive IoU | 13.62% | 25.83% | >=65% |
| Target-positive TP / FP / FN | 285 /1,693 /115 | 390 /1,110 /10 | Joint localization |
| Positive zones with recall>=90% and IoU>=50% | 2/16 | 1/16 | >=14/16 |
| Target-negative FP / known pixels | 0 /7,689 | 13 /7,689 | FPR<=1% |
| Target-negative FPR | 0% | 0.169% | <=1% |
| All pure-far FP in32full frames | 21,214 | 4,937 | Report-only |
| Full-image IoU | 61.81% | 80.93% | Report-only |
| All mixed-zone recall / IoU | 95.19% /52.75% | 98.57% /67.98% | Report-only |

Target-positive denominator:400near and7,531far pixels in16selected zones.
The fit rescues105previous near misses and loses0old target TP; it adds481FP
and removes1,064. Sixteen negative target zones add13FP from an original0.
The full32-frame pure-far decrease does not hide those newly introduced errors.

The fit meets recall and negative-FPR targets, but misses IoU and per-zone
coverage. One zone reaches recall94.12% /IoU80%; another reaches86.96% /64.52%
and fails the recall condition; one5-pixel target remains completely missed.
The saved16zone records preserve this heterogeneity. Preview colors are greenTP,
redFP, blueFN, purpleUNKNOWN. Visual review shows persistent broad near masks
around multiple small supports, even as large erroneous regions contract.

## Why this check was run

The preceding [frozen training-gap audit](../../../../artifacts.local/work/ba-nfo-training-gap-audit-20260919/REPORT.md)
found far-small train recall69.89% /IoU11.27%, versus test69.61% /12.09%.
The old full-image fit32 success did not establish fitting this difficult
subgroup. Many original near pixels were close to the2m threshold, so this
check selects depth-separated support instead of treating every small near
area as an isolated small object. No original benchmark denominator changes.

## Selection and sealed method

Selection uses only the original TRAIN manifest and realized prepared arrays.
First choose16positive frames, then16negative frames, each from a different
scene across all32. Use original manifest order and the first eligible zone;
baseline scores and fitting results never select or replace a case.

- Positive: >=90%known depth,4to20%-of-known near pixels, q90near<=1.8m,
  q10far>=2.2m, their gap>=.5m, and a finite actual public return>=2.2m.
- Negative: >=90%known, every known depth>=2.2m, public return>=2.2m,
  grayscale standard deviation>=12and mean absolute Laplacian>=8.

All32selected zones were visually inspected before fitting; no case was
changed. Positives include small fragments of larger objects and surface
boundaries, not only thin isolated obstacles. Negatives are reproducible
textured pure-far controls; they were not proven hard by baseline errors
(indeed, original target-negative FP is0). Quantile separation and visual
inspection are not semantic instance annotations or physical-sensor evidence.

Keep complete256x192RGB, contiguous NCHW layout, exact original8x8ToF values,
architecture258,656parameters, four distance heads and full-known-image
BCE+.2ordinal loss. No cropping, target masks, selection labels, region weights
or depth truth enter inference. Selected zones enter evaluation only.

Initialize from the retained trained NFO checkpoint. Train all parameters for
exactly512updates, batch8 sampled without replacement within each batch,
seed190921, AdamW lr.002/weight_decay.0001, gradient clip5, constant LR as in
the original fit32 procedure. Keep the final update only. Both baseline and
fit use original cutoff **0.081**, with no recalibration or threshold sweep.
Measure the frozen baseline after selection and before any update.

## Interpretation and boundary

The joint rise in target recall and IoU with fewer target FP demonstrates
learnable improvement on these seen examples. It is not merely increasing
recall by predicting more pixels everywhere. But IoU25.83% and1/16passing
zones do not establish complete small-support learning.

This exact full-image-loss/512-update procedure fails to fit the selected
targets to the stated criterion. It does not establish that the architecture
cannot express them, RGB lacks information, resolution must increase, or a
different loss would succeed. The400target-positive pixels are only0.0257%of
the1,558,122known image pixels in this set; full-image averaging remains a
plausible competing explanation, not a measured causal finding.

No validation/test evaluation was performed, so there is no generalization
or improvement claim on the500-frame benchmark. No trained output is promoted
to the App, alert pipeline or retained NFO default. Close this single fit and
retain the diagnostic; any targeted-supervision or spatial-resolution contrast
requires its own subsequent scope. No successor was launched.

## Verification and retained artifacts

[Implementation](ba_nfo_hardfit.py) and [two selector tests](test_ba_nfo_hardfit.py)
cover separated support versus a threshold slice, missing returns, UNKNOWN
coverage and textured versus uniform negative controls. Both tests pass.
An independent process reloads the actual final weights and feeds only RGB
and public ToF values. All32four-head score arrays reproduce bit-for-bit,
all domain/frame counts match, target counts are independently recomputed,
nesting passes and the original checkpoint hash remains unchanged.

Actual backend: CUDA on NVIDIA GeForce RTX5060 Laptop GPU;512updates take16.87s.
This is host fit time, not phone latency.14,742unknown-depth pixels are recorded
and excluded from evaluation and loss. No unknown depth is
converted into a negative label. The process exited normally and no worker remains.

Evidence root: `artifacts.local/work/ba-nfo-hardfit32-20260919/`.
Retain protocol/selection hashes,32case contact sheets, baseline and final
scores, real final checkpoint,512-step log, per-frame/per-zone results,
comparison sheets, independent verification script/receipt, and local disposition.
The source checkpoint is referenced without duplicating it.

Global registration again encountered the existing ledger303 input-fingerprint
mismatch. Structured global inheritance is pending; local diagnostic disposition
and command receipts preserve the result without repairing or bypassing the ledger.
