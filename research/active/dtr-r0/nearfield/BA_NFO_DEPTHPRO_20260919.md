# Frozen Depth Pro structure and near-distance reference

Status: `NFO_DEPTHPRO_CONTOUR_GAIN_NO_NEAR2M_TRANSFER`. Completed one frozen
consumed-Development comparison, 1,036 new predictions, no training or tuning.
Retain original NFO. The native RGB reference improves spatial boundaries but
fails the joint near2m task; low RGB also fails. This result supports separating
shape from distance, not replacing NFO or claiming missing measurement is proven.

## Results on the unchanged500-frame task

| Metric | Original NFO | Depth Pro low256x192 | Depth Pro native1024x768 |
|---|---:|---:|---:|
| Far-small recall | 68.360% | 50.055% | 61.292% |
| Far-small IoU | 11.130% | 20.255% | 23.933% |
| Mixed recall | 95.000% | 74.892% | 81.642% |
| Pure-far FP pixels | 212,988 | 207,259 | 344,336 |
| All-small recall | 80.648% | 57.036% | 68.715% |
| Full-image recall | 96.982% | 87.205% | 91.100% |
| Full-image IoU | 61.621% | 71.264% | 70.911% |
| Outside-ToF FP pixels | 1,003,881 | 344,832 | 397,424 |
| Near-mask boundary F1,1px tolerance | 4.612% | 34.054% | 44.425% |
| Original four task gates passed | Reference3/4 | 2/4 | 1/4 |

Low/native far-small TP/FP/FN are9,617/28,267/9,596 and
11,776/29,992/7,437. Native rescues3,830 original FN but loses5,188 original
TP:1,358 net additional missed near pixels. Mixed regions lose105,467 TP and
rescue20,968 FN, while pure-far adds131,348 net FP. Better IoU and global FP
reduction therefore do not satisfy the recall/pure-far joint objective.

The frozen strata show useful heterogeneity without changing the primary task:
far-small<=1.8m recall rises60.23%->86.43% with native input, whereas1.8–2m
recall falls70.83%->53.66%. The existing depth-separated subset rises49.79%
->89.08%. Retain every near2m label and all500 frames; these subgroup gains do
not rescue the failed all-task result. Native far-small recall improves in3/6
families and declines in3/6; the full family table is in `results.json`.

## Contours improve only under the additional native-information condition

On the fixed matched70 subset, scale-independent directed ratio-edge F1 is
25.213% for cached UniDepthV2,20.266% for low Depth Pro and51.285% for native
Depth Pro. The native condition passes both frozen contour checks; low does not.
On all500, native ratio-edge F1 is53.203%; near-crossing discontinuity recall is
73.441%, versus38.969% with low input. These ratio measures do not depend on
predicted metric scale. Near-mask boundary F1 above does depend on it.

This supplies positive structure evidence for the native-input reference and
does not establish a same-information pretrained-model improvement. Extra image
detail, interpolation/alignment and architecture/calibration all differ from
the existing references. The cached UniDepth comparison has only70 rows and
estimated-camera inference. NFO has no comparable continuous-depth shape output.

## Separate old suspended-bar diagnostic

All18 old views were evaluated separately. On the unchanged1,054-pixel reference
support mask of `bar_near`, optical-z median is1.947m. Cached DA V2 predicts
14.205m; Depth Pro low predicts11.048m; native predicts2.875m. **All three predict
zero of these1,054 pixels below2m.** Native visibly recovers the bar as a separate
depth surface, but its absolute range still crosses the wrong side of2m.

Across all18 views, near2m recall rises81.46%->98.27%/98.97% for low/native, but
FP pixels rise350->193,304/329,260 and IoU falls81.36%->58.67%/46.04%. These are
optical-depth pixel diagnostics, not a re-evaluation of the old3m alert task.

The fixed bar visualization and two largest low-arm IoU gains/losses were
inspected; retrospective selection is recorded explicitly in `visual-selection.json`.
The images agree with the counts: clearer native bar shape, wrong range, and
mixed gains/regressions in the500-frame set. No images substitute for full counts.

## Decision

Close this exact frozen standalone Depth Pro<2m recipe as a replacement for NFO
(`NEGATIVE_CONTROL` for that role). Preserve its native contour predictions as
diagnostic evidence. Prioritize a source of near-layer distance or a separately
justified calibrated range mechanism before spending another round on a
segmentation head; this experiment does not prove those observations absent or
guarantee layered echo assignment will work. No ToF fusion, distillation,
alternate checkpoint, cutoff repair or automatic successor was started.

All500 predictions for3 evaluated arms and11 domains were independently
recounted with a separate joint-bin histogram implementation; every per-frame
and aggregate count matches. Original NFO counts are exact,350,140 UNKNOWN
pixels remain excluded, and all native output sampling checks pass. Both
input conditions and the18-view diagnostic complete without dropped frames.

Actual CUDA inference uses the host RTX5060 Laptop GPU. All1,036 calls plus
rectification, I/O and serialization take992.02s; scientific low/native call
p50 is764.53/768.03ms and p95 is810.30/809.85ms. These call timings include
preprocessing transform and output transfer but exclude CPU rectification and
file serialization. Peak allocated/reserved CUDA memory is3.979/7.116GB.
No phone/device latency, final-alert, natural-distribution or safety claim.

## Frozen question and contrast

Can one official detail-preserving reference improve near-mask localization and
actual optical-axis depth below 2m? Use the original 500 validation/Development
frames, all denominators and UNKNOWN pixels, original NFO cached masks at 0.081.
Depth Pro uses physical depth<2m without ground-truth scale alignment.

Official source: [Apple Depth Pro](https://github.com/apple/ml-depth-pro), revision
`9e65e4dbe9568d23c546fcec53302b10445e109e`, one official checkpoint SHA256
`3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce`.
Use official half-precision inference, as in its CLI. The public model is retrained
and does not exactly reproduce the paper; pretraining overlap is not excluded.

Two input conditions are frozen before scientific inference: original 256x192 RGB
and original 1024x768 RGB. Both use the same weight and inference configuration;
the native condition adds information and must be reported separately. Internal
1536x1536 processing means neither condition matches NFO's compute budget.

Hypersim public ray matrices include oblique projections; a scalar focal alone
does not describe every image. Rectify each RGB using the full public ray matrix
to a centered, square-pixel pinhole at the same dimensions. Choose focal from
all original rays with a two-pixel border; bilinearly sample RGB with zero padding.
Remap predicted inverse-z to the original rays, without rotating camera axes.
Sample native output at [::4,::4], matching the original point-ray label grid;
preserve native predictions. Calibration changes sampling, not source information.
The inherited input/label sampling also differs: a low RGB pixel averages a4x4
block centered at native4j+1.5, whereas the original nearest-depth label and
native prediction sample use native4j. Keep the original labels; native-versus-low
differences include this0.375low-pixel offset as well as added image detail.

The existing cached monocular reference is UniDepthV2 ViT-S, not DA V2, and covers
only 70 of these 500 rows. Compare it only on that matched subset. DA V2 belongs
to the separate old G5 diagnostic: all eighteen consumed views, both frozen
Depth Pro input conditions. Keep those optical-z<2m results separate from the
original <=3m heading-forward alert metrics and from the 500-frame task.

## Fixed decision rules

Keep the four existing joint targets: far-small recall>=75%; far-small IoU>=
original NFO; mixed recall>=94.5%; pure-far false-positive pixels<=original NFO.
Also expose full-image, outside-ToF, all-small, family and paired loss/rescue counts.

Measure two different boundary quantities. Near-mask boundary F1 uses one-pixel
tolerance on known adjacent pairs and depends on the 2m decision. Scale-independent
boundary P/R/F1 uses directed adjacent depth ratio>1.25 with exact edge matching;
compare Depth Pro and cached UniDepth on70, and report Depth Pro on all500.
NFO threshold probabilities have no equivalent continuous-depth boundary score;
do not invent pseudo-depth for them. Report near-crossing ratio-edge recall too.

Freeze contour support at near-mask boundary F1>=NFO+0.02 AND matched70
scale-independent F1>=UniDepth+0.02. These are diagnostic effect sizes, not
statistical significance. Joint task+contour success supports retaining this
reference for separately scoped ToF-assignment research. Contour-only success
prioritizes distance evidence. No contour gain closes this reference role.
Neither outcome launches fusion, distillation or another model automatically.
Interpret the two contour checks individually as well: only the depth-ratio
measure is independent of metric scale. Failure of near-mask boundary F1 may
follow a range error and cannot establish that spatial contours failed. The
frozen conjunction is a joint-usefulness requirement, not a pure shape test.
A positive matched70 shape result also cannot establish shape improvement
against NFO on all500 because NFO has no comparable continuous-depth output.

Implementation: [inference](ba_nfo_depthpro.py),
[saved-output evaluator](evaluate_ba_nfo_depthpro.py).
Evidence: `artifacts.local/work/ba-nfo-depthpro-20260919/`.
The sealed protocol, public observations, input identities and receipts precede
scientific outputs. Mechanical resume requires identical protocol/output hashes.

## Execution and validation

An independent geometry review and three geometry tests cover49 distinct public
camera matrices, the worst-oblique native view, inverse-z recovery of a synthetic
plane, RGB homography, bounds and the inherited sampling offset. Three evaluator
tests cover scale invariance, directed edges, strict ratio boundaries, UNKNOWN,
metric-versus-shape separation, one-pixel tolerance and empty denominators.

The official-model smoke uses only the upstream example image. Equivalent
cold CPU/GPU probes take66.93s/2.95s on this host and select CUDA. This includes
placement cost and is not the cohort steady-state latency. Peak smoke tensor
allocation is3.97GB. The existing runtime uses NumPy2.4.4 although upstream declares
NumPy<2; imports and actual CPU/CUDA inference pass, with no shared runtime change.

The initial pre-inference geometry guard correctly rejected the scalar-focal
assumption. It was replaced before sealing/scientific inference with the full-ray
rectification above. A launch guard then encountered only upstream Python bytecode
directories created by the smoke; tracked-source cleanliness plus every upstream
Python source hash are now checked. The pre-inference launch addendum binds source,
observation manifests, protocol, weight and upstream revision for resume integrity.

Registration was attempted and remains blocked by the pre-existing
`experiments/index.jsonl:303` input-fingerprint error. The failure receipt is
retained; no ledger change or bypass was performed.
The inheritance command separately reports an unknown terminal ID because this
new terminal is not registered. `disposition.json` records its exact scoped role
locally; global registration/inheritance remain incomplete, not silently repaired.
The hot-document index check passes10 files and481 local links. Existing unrelated
working-tree line-ending warnings were excluded from the scoped staged check.
All task inference processes have exited. Official weights/source,1,036 predictions,
protocols, input/output hashes, logs, metrics and review images are retained under
the evidence root for reproduction. Task-owned bytecode and the temporary current
snapshot were removed and absence verified (288,460 logical bytes); no worker or
paid allocation remains. The previous current-page working bytes were preserved.
