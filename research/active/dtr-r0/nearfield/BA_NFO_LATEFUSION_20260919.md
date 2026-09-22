# Late spatial conditioning improves localization, but coverage remains 12/16

One authorized architecture contrast is complete on the same 32 TRAIN frames.
A zero-initialized, full-resolution RGB/ToF residual branch improves target
localization and global retention relative to the fixed joint-loss fit. Five of
six original gates pass and all four paired retention guards pass; individual
positive-zone coverage is 12/16, below the required 14/16. Retain the measured
contribution as a component diagnostic, not a promoted model or generalization
result. The broader small-support domain loses recall and remains part of the
result. This one 512-update recipe is finished, with no successor or val/test run.

## Paired results

All values below concern the same seen TRAIN images, 2m threshold and 0.081 score
cutoff. Target zones are the original 16 positive and 16 negative selections;
the broader small-support domain includes all qualifying zones in these images.

| Metric | Previous joint fit | Late spatial branch |
|---|---:|---:|
| Selected-positive recall | 98.75% | 99.50% |
| Selected-positive IoU | 66.72% | 70.07% |
| Selected-positive TP / FP / FN | 395 / 192 / 5 | 398 / 168 / 2 |
| Positive zones meeting recall>=90%, IoU>=50% | 11/16 | 12/16 |
| Selected-negative FP / 7,689 known pixels | 1 | 0 |
| Full-image recall | 98.89% | 99.50% |
| Full-image IoU | 74.94% | 76.78% |
| Pure-far FP | 16,645 | 9,496 |
| Broader small-support recall | 92.43% | 90.23% |
| Broader small-support IoU | 21.88% | 25.15% |
| Broader small-support TP / FP / FN | 5,116 / 17,850 / 419 | 4,994 / 14,322 / 541 |

Target near/far denominators remain 400/7,531; all-pure-far denominator remains
585,548. The broader small-support recall cost is 122 additional FN, or 2.20
percentage points, despite its improved IoU. Do not describe this as dominance
over every domain. The prior full-only fit also retains better full-image IoU
(80.93%) and fewer pure-far FP (4,937), at much poorer target IoU (25.83%).

The six original gates remain target recall>=95%, target IoU>=65%, negative
FPR<=1%, at least 14/16 individual positive zones passing, full recall no worse
than original NFO, and pure-far FP no worse than original NFO. Only zone coverage
fails. Before fitting, this contrast additionally required retaining the previous
joint fit's target recall/IoU, full recall and pure-far FP for a positive paired
result. All four pass, but they do not override the failed coverage gate.

## What changed

The original NFO already has full-resolution RGB skip features; this experiment
does not claim those were absent. Its ToF features entered at the 1/8-resolution
bottleneck. The new path concatenates final decoder features (16 channels),
full-resolution RGB skip features (12), and the same encoded ToF features (16)
broadcast through the exact original public zone geometry. Two 3x3 convolutions
with the existing normalization/activation produce 16 channels, followed by a
zero-initialized 1x1 four-logit correction added to the original logits.

The original coarse path is preserved. The late path uses integer public-box
lookup, not resizing that could shift uneven zone boundaries. Neither target
boxes, native occupancy, depth truth nor failure identity enters forward inputs.
Shared ToF features derive only from the original public six-field observations.
The new branch does not add sensor information.

Parameter count rises from 258,656 to 267,460 (+8,804, about 3.4%). Thus added
capacity and late conditioning are coupled; this single contrast cannot assign
their separate causal contributions. All parameters train, just as all original
parameters trained in the previous joint fit. Only the final residual convolution
starts at zero; all 32 initial four-head score arrays match original NFO exactly.

Restart from original NFO, not the previous joint checkpoint. Reuse the exact
32 TRAIN identities, RGB256x192/public8x8ToF, 512 batches of eight, seed190921,
AdamW lr0.002/wd0.0001, constant LR, clip5, final checkpoint and cutoff0.081.
Loss remains 0.5 full-known + 0.5 target-zone loss, each using the existing
four-head BCE+0.2ordinal formula and separate known-pixel mean. UNKNOWN stays
excluded. No ratio, threshold, longer schedule or alternative architecture was
tried. The protocol allows no validation/test access during this bounded fit.

## Error mechanism and four remaining failures

The [native-support audit](BA_NFO_NATIVE_SUPPORT_AUDIT_20260919.md) supplied saved
diagnostic footprints. These are used only after inference for error analysis,
not for training, readout or relabeling. Empty-native-near FP falls from 125 to
96, and three of four formerly missed high-coverage near pixels are recovered.
The one formerly missed mixed pixel is still missed. This supports a localization
contribution on these seen cases; it is not just fewer alerts or a cutoff change.

| Remaining failed case | TP / FP / FN | Recall | IoU |
|---|---:|---:|---:|
| ai_003_004 /cam01 /0056 /zone27 | 7 / 11 / 0 | 100% | 38.89% |
| ai_050_004 /cam04 /0065 /zone46 | 9 / 19 / 0 | 100% | 32.14% |
| ai_048_010 /cam02 /0044 /zone55 | 3 / 0 / 1 | 75% | 75% |
| ai_048_009 /cam00 /0037 /zone27 | 4 / 0 / 1 | 80% | 80% |

Only ai_015_009 /cam00 /0057 changes from failed to passed; no previously passed
positive zone becomes failed. The original fully missed four-pixel target improves
to 3/4 but still fails the 90% recall criterion. Retain every example and the
original criteria; neither cross-zone connectivity nor partial recovery erases
the four remaining failures.

## Checks, compute and disposition

[Implementation](ba_nfo_latefusion.py) and [two focused tests](test_ba_nfo_latefusion.py).
Tests pass for exact initial logits, residual-gradient onset, post-update branch
effect, exact public broadcast boundaries, outside-coverage masking and broadcast
gradients. All 32 actual frame mappings match their original public boxes.

An independent process reloads actual final weights and passes only RGB/ToF:
all 32 four-head score arrays reproduce bit-for-bit; full per-frame/per-domain
metrics and native error counts match, masks are finite/nested, selection and
batch hashes match, all 512 loss logs satisfy the fixed half/half mixture, gate
counts match, and original/joint checkpoints remain unchanged. No held-out
prediction files exist. Independent read-only review found no material metric
or architecture implementation issue.

Backend selection benchmarks equivalent batch-eight forward/loss/backward work
without optimizer steps on disposable model copies: CPU median746.45ms, CUDA
median46.62ms, using the project backend selector. Actual training used NVIDIA
GeForce RTX5060 Laptop GPU and took25.84s. These are host research measurements,
not phone latency or a speedup over the original NFO. All processes exited;
`-B` avoided task bytecode caches. No worker or further fit remains running.

Retain protocol, selection/batches, backend receipt, 512-step log, actual weights,
scores, paired metrics, native error tables, previews, independent verification,
disposition and delivery receipts under:
`artifacts.local/work/ba-nfo-latefusion32-20260919/`.
Preview columns are fullRGB / zoneRGB / nearGT / originalNFO / lateFusion; the
table above compares previous joint fit, not the originalNFO preview column.

Global registration still fails at the existing ledger line303 fingerprint
mismatch; inheritance is pending. Actual failed-command receipts and local
structured disposition preserve the record without changing the ledger.
