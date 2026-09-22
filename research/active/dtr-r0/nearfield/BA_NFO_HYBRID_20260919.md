# Matched NFO with continuous-depth auxiliary supervision

## Completed result: joint target failed

Retain the original NFO baseline. The auxiliary changes the error tradeoff,
but does not recover sparse foreground while preserving localization.
This one frozen recipe is a **NEGATIVE_CONTROL**, not a rejection of every
multi-task approach or proof that the sensor lacks recoverable information.

| 2m metric | Depth | NFO | NFO + auxiliary |
|---|---:|---:|---:|
| Far-return <=20% recall | 89.64% | 69.61% | 79.98% |
| Far-return <=20% IoU | 8.71% | 12.09% | 10.02% |
| Far-return <=20% precision | 8.80% | 12.76% | 10.28% |
| Far-return <=20% FPR | 62.11% | 31.79% | 46.65% |
| Mixed IoU | 47.11% | 51.04% | 49.34% |
| Mixed recall | 93.09% | 94.70% | 93.28% |
| Full-image IoU | 38.04% | 56.66% | 48.14% |

Far-return small foreground changes from9651TP/65965FP/4213FN to
11088TP/96794FP/2776FN. Paired comparison rescues1711 previously missed
near pixels but loses274 existing true pixels; adds36741 false near pixels
and removes5912. Net1437 more TP comes with30829 more FP. This supports
partial recall recovery with excessive spill, not the hoped-for complementarity.

| <=20% near-area return group | NFO recall / IoU | Hybrid recall / IoU |
|---|---:|---:|
| near | 92.42% / 15.43% | 87.83% / 16.01% |
| far | 69.61% / 12.09% | 79.98% / 10.02% |
| missing | 72.97% / 12.71% | 71.58% / 11.15% |
| all | 83.01% / 14.09% | 84.13% / 13.03% |

| Near-area band, all returns | NFO recall / IoU | Hybrid recall / IoU |
|---|---:|---:|
| 0-5% | 66.32% / 4.01% | 67.26% / 3.53% |
| 5-10% | 75.50% / 11.70% | 77.16% / 10.50% |
| 10-20% | 87.44% / 20.34% | 88.42% / 19.62% |
| 20-50% | 91.48% / 38.21% | 89.95% / 37.60% |
| 50-100% | 96.97% / 76.30% | 95.34% / 75.28% |

Mixed IoU improves in 13/57 eligible scenes; scene-macro delta
is -2.54 percentage points. Test UNKNOWN is160365 pixels,
excluded equally from all arms. Every original full/mixed/small/known-return/
missing-return raw count at all four distances reproduces exactly.

| Distance, mixed | NFO recall / IoU | Hybrid recall / IoU |
|---|---:|---:|
| 1.0m | 96.34% / 50.01% | 94.67% / 50.44% |
| 1.5m | 94.47% / 46.76% | 91.86% / 47.06% |
| 2.0m | 94.70% / 51.04% | 93.28% / 49.34% |
| 3.0m | 97.15% / 52.27% | 95.81% / 51.73% |

The fixed initial loss ratio is0.8932267208 (NFO0.79933882 / depth0.89488906).
Training took153.00s on the RTX5060 Laptop CUDA backend; 12epochs/1500updates.
There are258673 training parameters and258656 deployed parameters.
Validation selected cutoff0.051 with95.005% recall and49.074% IoU.
No test threshold search or alternate lambda was run.

Four focused tests passed: identical main initialization, exact pruned output,
UNKNOWN loss-gradient exclusion and auxiliary gradient into shared layers.
An independent process loaded only deployed weights plus public RGB/ToF input;
finite nested four-threshold scores/masks passed. A read-only implementation
review found no scoped defect. The six preview images reuse the original
identity-fixed selection; visual inspection shows substantial red FP spill.

Preserve `hybrid-training.pt`, `trained-nfo.pt`, `results.json`, paired/frame
counts, calibration histogram, `comparison.jpg`, hashes and all receipts.
No per-image score cache was created. The owning process exited successfully;
no task-owned worker remains and no further training is launched.

Registration/inheritance remain globally pending: the actual registration
attempt hit the existing ledger303 fingerprint mismatch and the inheritance
tool rejected the unknown terminal. `disposition.json` preserves the local
NEGATIVE_CONTROL assignment. No ledger bypass or repair was performed.

Independent public-input inference:

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_matched.py infer --checkpoint artifacts.local/work/ba-nfo-hybrid-20260919/trained-nfo.pt --sample YOUR_PUBLIC_INPUT.npz --output YOUR_PREDICTION.npz
```

## Frozen experiment

Test one new supervision recipe against the retained matched depth and NFO
controls. A 17-parameter 1x1 log-depth head reads the original last decoder
features. Train with unchanged NFO BCE plus 0.2 ordinal loss and lambda times
the original valid-pixel log-depth SmoothL1 loss. The deployed network is exactly
the original four-output NFO architecture: the auxiliary head is removed.

Freeze lambda as the initial NFO/depth loss ratio on the first 32 TRAIN identities
in the original manifest, before any optimizer update. No validation/test labels
select this weight. Keep seed190921, all original main-head/backbone initial
values, 3000/500/500 Hypersim identities and simulated ToF values, 192x256,
12epochs, batch24, AdamW0.002/0.0001, cosine to10%, gradient clip5 and last epoch
selection. One fit only; no sweep, additional fit32 run, rescue or successor.
The extra head adds training work; equal updates do not imply identical FLOPs.

Calibrate once on validation 2m mixed-known pixels with the original rule:
maximize IoU subject to recall>=95% on the 0.001..0.999 grid. Freeze the cutoff
across all four distances and test images. Recompute both retained controls with
their original frozen cutoffs and require identical original raw domain counts.

The joint target, fixed before training, is far-return <=20% foreground
recall>=82%, subgroup IoU>=original NFO, mixed-domain IoU>=original NFO, and
mixed recall no more than one percentage point below original NFO. Validation
calibration must be feasible. Report all area bands and near/far/missing returns,
precision/FPR, TP/FP/FN/TN, paired rescued/lost TP and added/removed FP,
scene results, and UNKNOWN. The historical51.04% is mixed-domain IoU, not
full-image IoU. High depth-arm recall alone does not prove fine localization.

This is consumed synthetic Development with simulated single-return ToF. It
does not establish physical-sensor performance, thin-object semantics, final
alerts, or safety. A/A*, Radar and Android defaults remain unchanged.

Entry: `ba_nfo_hybrid.py`. Evidence:
`artifacts.local/work/ba-nfo-hybrid-20260919/`.

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_hybrid.py
```

The entry refuses an existing protocol to protect the single-run budget.
Partial training cannot resume; do not automatically restart a failed fit.
Completion or failure is recorded in `completion.json`.
