# BA-NFO small-area error decomposition

## Decision

The 2m recall loss is concentrated in small foreground zones **whose public
ToF return is far**, not zones with an observed near return. Do not start the
proposed near-return existence auxiliary as the presumed fix for this failure.
Keep the matched NFO Development gain and both trained models. This diagnostic
does not reject return-to-pixel assignment in general.

No training, threshold selection, model modification, data expansion, alert
change or oracle geometry experiment was performed. This is posthoc analysis
of the already consumed 500-frame synthetic Development test split.

## Fixed comparison and reproducibility

Use the retained 258,605-parameter depth and 258,656-parameter NFO models from
[the matched pilot](BA_NFO_MATCHED_20260919.md), frozen cutoffs 0.388/0.081,
identical RGB/realized public ToF inputs, all four distance thresholds. Each
zone is binned by known near pixels / all known pixels, excluding pure-near,
pure-far and all-unknown zones. Bins are (0,5], (5,10], (10,20], (20,50],
(50,100) percent. NaN truth remains excluded. Public return strata are near
(finite range < threshold), far (finite range >= threshold), and missing.
These strata and ground-truth area never enter inference.

Every TP/FP/FN/TN sum exactly reproduces the original mixed-zone results for
both arms at all four thresholds. One initial rerun used noncontiguous RGB
layout, which changed floating-point predictions at cutoff boundaries and
failed this exact-count guard; restoring the original contiguous tensor layout
resolved it. No weights, thresholds or labels changed. CUDA RTX5060 Laptop
GPU was used; existing equivalent-model placement evidence was reused.

Run `ba_nfo_area_diagnostic.py` with the configured research GPU Python.
Durable protocol, hashes, counts, public-return strata and timing are in
`artifacts.local/work/ba-nfo-area-20260919/`. `results.json` includes all bins
at all distances, not only the primary 2m table.

## 2m area bins

| Near area | Depth TP / FP / FN | NFO TP / FP / FN | Depth recall | NFO recall | Recall delta | Depth IoU | NFO IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–5% | 2667 / 81071 / 842 | 2327 / 54510 / 1182 | 76.00% | 66.32% | -9.69pp | 3.15% | 4.01% |
| 5–10% | 6419 / 58973 / 1214 | 5763 / 41644 / 1870 | 84.10% | 75.50% | -8.59pp | 9.64% | 11.70% |
| 10–20% | 23604 / 101236 / 2564 | 22881 / 86314 / 3287 | 90.20% | 87.44% | -2.76pp | 18.53% | 20.34% |
| 20–50% | 118886 / 191513 / 13804 | 121388 / 184971 / 11302 | 89.60% | 91.48% | +1.89pp | 36.67% | 38.21% |
| 50–100% | 360325 / 103874 / 19566 | 368385 / 102905 / 11506 | 94.85% | 96.97% | +2.12pp | 74.48% | 76.30% |

Below 10% accounts for 996/1719 (57.94%) of the net extra small-area misses;
the 10–20% band accounts for 723/1719 (42.06%). Thus the loss is strongest
at the smallest areas but is not confined below10%. Both medium/large bands
improve recall, FP and IoU at2m.

## Public-return decomposition of <=20% foreground

| Public return | Near pixels | Depth TP / FP / FN | NFO TP / FP / FN | Depth recall | NFO recall |
|---|---:|---:|---:|---:|---:|
| Near | 21648 | 19106 / 101670 / 2542 | 20008 / 107980 / 1640 | 88.26% | 92.42% |
| Far | 13864 | 12427 / 128855 / 1437 | 9651 / 65965 / 4213 | 89.64% | 69.61% |
| Missing | 1798 | 1157 / 10755 / 641 | 1312 / 8523 / 486 | 64.35% | 72.97% |

The far-return group adds2776 misses, offset by902 fewer in near-return and
155 fewer in missing-return groups: net1719. Near-return groups also add6310
FP pixels; the main FP reduction is62890 fewer in the far-return group.
This supports a specific competing explanation: the model relies more strongly
on the dominant far return in mixed zones, reducing overexpansion but losing
small foreground not represented by that return. It does not establish this
causal mechanism without a separate intervention.

An optimistic arithmetic bound illustrates the mismatch with the proposed fix:
current small-area NFO has30971TP/37310positive pixels. Recovering **all1640**
misses in near-return zones with no new FP gives32611/37310 = **87.41%**, below
the requested>=87.6% (requiring at least1713 recoveries). Matching the previous
depth recall87.617% requires1719 recoveries. This bound only applies to a change
restricted to near-return zones; a shared-network auxiliary could affect other
zones, but then success would not follow from the claimed near-return evidence.
It is a count decomposition, not a trained-model ceiling or a general
impossibility claim.

## Distance dependence

The ultra-near effect is not the same recall-loss pattern. At1m, NFO recall
increases in every area band: +12.52,+17.44,+19.45,+13.44,+6.94pp respectively,
while pixel false-positive rates rise in every band. At1.5m only5–10% loses
recall (-1.50pp); the other bands gain. At3m all<=20% bands lose recall, while
20–50% also loses0.99pp and>50% gains1.01pp. All raw values are retained.

The shared operating cutoff is a plausible source of distance-dependent
tradeoffs, but this diagnostic does not identify calibration versus learned
representation as the cause. No per-distance cutoff was tuned.

## Scope and next hypothesis

Retain the two-arm evidence as a Development result of these supervision
recipes, not proof that direct supervision universally dominates depth.
Retain this diagnostic as a COMPONENT: useful error/support decomposition.
The fixed proposed explanation "measured near evidence erased by pixel area"
is not supported as the dominant cause of the aggregate small-area regression.

A more targeted future question is whether RGB can preserve small foreground
**when only far background returned**, without restoring zone-wide FP. That is
an unreturned-surface/generalization problem; an assignment-only model whose
depth layers are restricted to observed returns cannot directly represent it.
Unknown must remain an explicit option rather than being treated as far/clear.
Do not relabel missing near observations as measured evidence. No successor
experiment is launched by this report.

Global experiment/terminal registration remains pending the already observed
ledger303 input-fingerprint mismatch and unknown-terminal condition. Preserve
local receipts and do not bypass or rewrite the ledger.
