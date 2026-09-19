# BA-NFO matched supervision pilot

This experiment asks whether direct near-field supervision improves localization
relative to continuous depth supervision with the same trainable RGB/ToF network.
It does not reopen the previous frozen-encoder four-arm recipe or the MZ183
geometric readout. A/A*, Radar, final alerts and the Android default are unchanged.
Return-to-pixel assignment remains a subsequent idea, not a third arm or oracle
prerequisite for this run.

## Completed result

**The matched supervision comparison meets the predeclared Development target.**
At 2m, mixed-zone IoU rises **47.11% to 51.04% (+3.93 points)**, while recall
rises **93.09% to 94.70% (+1.61 points)**. False near pixels fall from 536,667
to 470,344 (12.36% fewer), and missed near pixels fall from 37,990 to 29,147.
Precision rises from 48.82% to 52.54%; false-positive pixel rate falls from
68.33% to 59.89%. This is a useful relative localization signal, with substantial
absolute error remaining, not a usable alert-system result.

All numbers below are held-out 500-image test results with validation-frozen
cutoffs: depth 0.388, NFO 0.081. Validation recalls were 95.08% and 95.00%.
The primary domain contains 1,335,262 known pixels, 549,891 positive and 785,371
negative; both arms use exactly the same denominator.

| Mixed-zone threshold | Depth IoU | NFO IoU | Depth recall | NFO recall | Depth pixel FPR | NFO pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| 1.0m | 50.89% | 50.01% | 87.26% | 96.34% | 56.06% | 72.70% |
| 1.5m | 46.08% | 46.76% | 89.20% | 94.47% | 62.17% | 67.80% |
| 2.0m | 47.11% | 51.04% | 93.09% | 94.70% | 68.33% | 59.89% |
| 3.0m | 47.61% | 52.27% | 97.08% | 97.15% | 81.56% | 67.39% |

**Small foreground remains a weakness.** At 2m in zones with <=20% near area,
IoU improves 11.73% to 14.09% and FP falls 241,280 to 182,468, but recall drops
**87.62% to 83.01% (-4.61 points)**, with FN increasing 4,620 to 6,339.
Therefore this experiment does not establish preserved thin-object recovery.
Nor is the gain uniform across distance thresholds: 1m IoU declines and the
1m/1.5m false-positive rates rise at the shared operating cutoff.

| 2m domain | Known pixels | Depth IoU | NFO IoU | Depth recall | NFO recall |
|---|---:|---:|---:|---:|---:|
| Full image | 24,415,635 | 38.04% | 56.66% | 96.48% | 96.08% |
| Mixed, return known | 1,266,951 | 47.24% | 51.14% | 93.57% | 95.21% |
| Mixed, return missing | 68,311 | 44.60% | 49.04% | 84.05% | 85.21% |
| Mixed, near area <=20% | 437,494 | 11.73% | 14.09% | 87.62% | 83.01% |

The all-image domain includes out-of-ToF-FOV pixels and near floors. It is a
localization check, not a collision-task metric. The 500 test images contain
160,365 unknown-depth pixels (0.65%) excluded equally from both arms.

IoU improves in **48/57** test scenes with mixed-zone positives; nine of 66
scenes have no eligible positives. Scene-macro IoU change is +4.92 points.
All six held-out family aggregates improve:

| Family | Depth IoU | NFO IoU |
|---|---:|---:|
| ai_008 | 47.49% | 53.08% |
| ai_028 | 43.35% | 47.82% |
| ai_037 | 49.00% | 53.93% |
| ai_044 | 45.41% | 46.52% |
| ai_053 | 42.23% | 46.30% |
| ai_055 | 49.42% | 52.85% |

A descriptive 2,000-replicate paired family bootstrap gives a 95% percentile
interval of +2.65 to +5.06 IoU points; recall change is -0.50 to +4.28 points.
This was computed after evaluation, not used for selection or the pass criterion.
With six families and one seed, it does not establish a universal >=3-point gain.

## Weights, visible outputs and decision

`trained-depth.pt` and `trained-nfo.pt` are actual trained checkpoints, each about
1.06MB. Both loaded in independent inference processes using an NPZ containing
only RGB and public zone values, and produced finite `[4,192,256]` scores and
validation-calibrated nested masks on the RTX 5060 Laptop GPU.

`comparison.jpg` and `small-foreground.jpg` show RGB, <2m truth, depth prediction
and NFO prediction side by side. Green is true positive, red false positive,
blue missed near surface, purple UNKNOWN depth. Examples are the first eligible
frames per scene in the sealed identity order, chosen by truth strata, never
by model success. `validation-curves.png` shows the recall/error tradeoff;
`results.json` includes all domains, raw counts and all scene results.

The 32-frame fits reached 67.62%/89.06% IoU at >=95% recall for depth/NFO,
passing the implementation check. Full training took 127.29/134.85 seconds,
excluding data loading, validation, rendering and prior preparation. Actual CUDA
execution is recorded. A matched 8-image forward/backward probe measured
28.0ms GPU versus 401.4ms CPU; this is host training placement evidence, not
phone latency. Final inference output nesting is enforced; unconstrained logits
may violate ordering, with the raw violation count preserved in `results.json`.

Disposition: **COMPONENT_OR_CHALLENGER / CHALLENGER**, scoped to direct
near-surface supervision in this fixed synthetic Development comparison.
Keep the result and both controls; do not promote final alerts, claim thin-rod
retention, expand training, add ablations, or launch return-assignment here.
This run ends after the agreed comparison and delivery.

Experiment registration is blocked by the pre-existing
`experiments/index.jsonl:303` input-fingerprint mismatch. No ledger entry was
bypassed or rewritten. Local protocol, results, hashes and registration receipt
remain available; structured global registration/inheritance is pending.

## Data and comparison

Use 4,000 already prepared Hypersim RGB/axial-depth pairs: 3,000 training,
500 validation and 500 test, from 335/52/66 scenes. Keep the existing entire
`ai_XXX` family split, then select frames by a fixed identity hash. These are
training-unseen families but previously used Development data, not fresh blind
confirmation. No SANPO, UE data or test labels enter training.

The UE inventory found 2,500 actual RGB/SceneDepth pairs. All five regions share
the same CitySample map and repeat all 500 foreground parameter configurations;
a region split would measure background transfer, not unseen foreground layouts.
This is why the already available Hypersim source was selected instead.

Hypersim radial distance is converted to camera axial depth with per-scene
`M_cam_from_uv`. An independent audit recomputed one original HDF5 from each
split: all three matched prepared depth exactly. RGB and depth are 256x192;
depth uses nearest-neighbor resampling. NaN depth is UNKNOWN, excluded from
loss and known-pixel denominators. Dataset attribution: Apple Hypersim,
Mike Roberts et al., ICCV 2021, CC BY-SA 3.0. Raw data is retained locally;
this report does not redistribute the dataset.

Both arms receive identical RGB and fixed, already realized simulated ToF
returns. Six fields per 8x8 zone are normalized range, valid, missing, coverage,
and zone-center x/y. The public distance is an axial-depth proxy, not a
calibrated physical sensor: central 80% image footprint, dominant inverse-square
weighted 10cm bin, 1cm+2% Gaussian noise, 5% missing, 0.1–8m range. No native
return point, true near fraction or evaluation mixed-zone flag enters the model.

Both networks train their small U-Net RGB encoder, ToF encoder, local fusion
and pixel decoder from scratch with bit-identical shared initial weights. The
depth head has one log-depth output and SmoothL1 loss (0.1–80m target clamp).
NFO has four logits for 1/1.5/2/3m with BCE and 0.2 ordinal penalty; cumulative
maximum guarantees nested inference masks. The heads differ by 51 parameters
(258,605 versus 258,656 total). This compares supervision recipes, not merely
fusion modules or physical sensor accuracy.

First fit 32 training frames with at least 32 known <2m pixels for 512 updates
per arm. Both must achieve fit-set IoU >=65% at recall >=95% before full training.
Full training restarts from the original matched initialization: 12 epochs,
batch 24, AdamW lr0.002, weight decay0.0001, cosine learning rate to 10%, identical
batch sequence, last checkpoint only. No pretrained model or checkpoint sweep.

## Evaluation fixed before outcomes

Each arm selects one operating cutoff using validation 2m mixed known pixels:
maximize IoU subject to recall >=95%, on a fixed 0.001–0.999 grid. A depth score
is `sigmoid(log(threshold)-logdepth)`: it is a monotone score, not a calibrated
probability. Its cutoff corresponds to one global depth multiplier. The NFO
cutoff is applied to nested sigmoid scores. Each cutoff remains fixed across
all four distance thresholds and all test scenes.

The main mixed domain includes every zone with at least one known near and one
known far pixel, including zones whose ToF return is missing. No P20 exclusion.
Report full image, mixed, mixed with known/missing return, and mixed small near
area <=20%, plus scene metrics and raw counts. Small area is not a semantic
thin-rod annotation; subpixel structures and native-resolution coverage are
not established by this 256x192 experiment. Near floors and other surfaces are
included; these are surface-distance labels, not collision labels.

The continuation target is >=3 IoU points with no more than 1 recall point lost
on test mixed pixels, positive scene-macro IoU change, and improvements in at
least 60% of evaluable scenes. Both validation operating points must be feasible.
This pragmatic one-seed target is not a significance test or a safety claim.
No test scanning, rescue training, modality ablation, or successor is included.

## Reproduction and durable evidence

Entry: `research/active/dtr-r0/nearfield/ba_nfo_matched.py`.
Evidence: `artifacts.local/work/ba-nfo-matched-20260919/`.
Use the configured research GPU Python. Dependencies are Torch, NumPy, OpenCV
and Matplotlib; the run records the actual backend and package versions.

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_matched.py fit32
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_matched.py train
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_matched.py evaluate
```

Existing fitted checkpoints are protected from accidental overwrites. The
independent `infer` command needs only RGB uint8 `[192,256,3]` and public `values`
`[64]` with NaN for missing returns in an NPZ; it never reads dense truth.

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/ba_nfo_matched.py infer --checkpoint artifacts.local/work/ba-nfo-matched-20260919/trained-nfo.pt --sample YOUR_PUBLIC_INPUT.npz --output YOUR_PREDICTION.npz
```

`protocol.json` and `manifest.json` fix budgets and sample identities;
`content-admission.json` verifies every selected prepared-array hash and content;
`data-audit.json` retains independent source/geometry checks. A pre-evaluation
amendment requires validation calibration feasibility in the final gate;
`execution-code-receipt.json` explains its difference from the original code
hash. No model, loss, sample, budget or threshold choice changed in that repair.

Five focused tests cover missing inputs, tiny foreground inclusion, UNKNOWN
loss exclusion, calibration feasibility and nested thresholds. CUDA forward
and backward smoke checks passed. The independent implementation review found
and corrected the feasibility gate omission before test evaluation.
