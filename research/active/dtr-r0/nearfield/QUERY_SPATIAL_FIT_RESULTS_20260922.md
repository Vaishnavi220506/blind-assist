# Tiny-training spatial query fit: query answers learned, added branch not retained

Decision: **NEITHER_PACKAGE_FITS** under the fixed joint fitting gate. The new
spatial branch does not improve macro mask localization over the original global
FiLM structure. Both structures can nevertheless learn image-dependent query
answers on these 84 training images. Invariant query ordering in the earlier
selected checkpoints is therefore not an unavoidable property of that structure.
This check does not isolate why the earlier training recipe failed.

[Protocol](QUERY_SPATIAL_FIT_PROTOCOL_20260922.md) was fixed before the two fits.
The old [held result](QUERY_OCCUPANCY_RESULTS_20260922.md) and
[frozen-state diagnosis](QUERY_OCCUPANCY_DIAGNOSTIC_RESULTS_20260922.md) remain
unchanged. This experiment is consumed training-only Development, not new held
evidence, an alert improvement or a safety claim. A and UNKNOWN are unchanged.

## Fixed comparison and result

Four metadata-selected training groups, three relations, seven poses: 84 images,
504 valid queries, 180 occupied/visible queries and 324 empty queries. Each family
contributes 21 images. Identical initial weights, batch schedule, new shared
fitting loss and optimization; one 100-epoch/700-update fit per arm. Score only
the final state with probability and mask cutoffs fixed at 0.5.

| Training measure | Original global FiLM | Added spatial branch |
| --- | ---: | ---: |
| Query TP / FP / FN / TN | 179 / 1 / 1 / 323 | 180 / 0 / 0 / 324 |
| Query precision / recall | 99.44% / 99.44% | 100% / 100% |
| Query FPR | 0.31% | 0% |
| Positive-mask macro IoU | **0.484823** | **0.481782** |
| Same-image correct query pairs | 540 / 540 | 540 / 540 |
| Lateral / height correct pairs | 216 / 216; 324 / 324 | 216 / 216; 324 / 324 |
| All / positive correct distance bins | 502 / 504; 179 / 180 | 504 / 504; 180 / 180 |
| Distinct descending query orders | 24 | 20 |
| Mask foreground-area precision / recall | 51.84% / 88.90% | 52.90% / 93.87% |
| Empty-query predicted area / valid area | 0.2474% | 0.2215% |
| Empty queries with any valid predicted area | 145 / 324 | 149 / 324 |
| Joint fitting gate | FAIL: IoU < 0.50 | FAIL: IoU < 0.50 |

Each arm passes the 95% query precision, recall and pair-order criteria, but
neither passes the separately declared localization criterion. The spatial arm
removes two query errors and increases aggregate foreground recall; that does
not establish improved localization. Its mean per-positive-query overlap is
slightly lower, and about half its predicted mask area is background.

| Family | Occupied queries | Global IoU | Spatial IoU |
| --- | ---: | ---: | ---: |
| body_protruding_plane | 60 | 0.3733 | 0.4168 |
| body_suspended_solid | 60 | 0.3968 | 0.4063 |
| head_hanging_plane | 30 | 0.6797 | 0.6316 |
| head_horizontal | 30 | 0.6891 | 0.6128 |

The spatial arm trades better BODY overlap for worse HEAD overlap in this tiny
cohort. With one initialization and four training groups, these are descriptive
effects, not statistically established superiority of either arm.

## What this resolves

Before fitting, all 720 constant query permutations were checked. The best
fixed order gets only 75.56% of all pairs correct (72.22% lateral, 77.78% height).
Thus the observed 100% pair scores cannot be explained by the earlier invariant
query order on this cohort. The original global structure is capable of fitting
query-dependent answers under the new small-cohort optimization conditions.
The new balanced loss, more updates per image, learning rates and reduced cohort
all differ from the historical recipe; this experiment does not separate them.

The oracle binary coarse-grid mask IoU averages 0.7450, so the 0.50 macro gate
is not ruled out by resolution. The thinnest single query has an upper bound
of only 0.1094; this is why the gate is a macro average, not an every-query rule.
Maps visibly locate the objects but often cover too much surrounding area or
misplace their boundaries. Mask precision and per-family overlap, rather than
query existence classification, remain the unresolved training-fit surface here.

The extra branch uses fixed ray/depth hypotheses and query boundaries, never
native depth or actor truth. It adds 768 effective weights, about 0.49% of the
original 158137 parameters. Both stored models have 158905 parameters because
the control retains but bypasses the branch. This is a package comparison, not
a strict isolation of geometry from additional effective capacity.

## Evidence, execution and disposition

Source root: `artifacts.local/evidence/ba-query-occupancy-20260922-prepared`.
Only selected train RGB/ToF rows and `labels/train.npz` were consumed. Dev and
held labels were not opened and their image rows were not inferred. Container
hashes validate immutable bytes; they do not confer fresh evaluation authority.
67 of the 84 frames retain baseline sensor UNKNOWN. Label-valid synthetic
geometry and fitted model scores do not turn those observations into measured
free space or independently supported metric range.

Output: `artifacts.local/evidence/ba-query-spatial-fit-20260922`.
The governed research-ue run succeeded and registered result lineage. Source,
input, initialization, batch, checkpoint and prediction seals are retained.
Equivalent cloned-model training probes measured CPU 305.33ms versus CUDA
32.45ms per batch. Actual execution used RTX 5060 Laptop GPU, Torch 2.11.0+cu130;
fits took 22.36s and 22.74s. There is no ongoing training worker.

Thirteen focused tests pass: initial pairing, true control bypass, spatial ray
geometry, arbitrary query permutation, projection gradients, missing-sensor
semantics, fixed metadata selection, fractional-area/empty/invalid losses,
oracle feasibility and cutoff counts. Independent output audit passes: raw
counts, fractional IoU, all query pairs, subgroup results, decision, admitted
inputs and output hashes reproduce; initialization and batch schedule regenerate
exactly. It also corrects one secondary reporting field: sealed
`any_predicted_area` counts any grid pixel, including predictions only in invalid
area. Its 159/158 counts become 145/149 when restricted to valid area, as reported
above. This additive correction is in `independent-audit.json`; sealed outcomes
are preserved. Area-weighted errors, IoU and the gate are unaffected.
Figures are `spatial-fit.png` and `spatial-fit-masks.png`;
examples use the fixed INSIDE/frame-5/centre query from each family, not a
selection by prediction quality.

Retain this exact two-arm fitting package as **NEGATIVE_CONTROL** for the joint
localization responsibility and retain its query-fitting evidence in the report.
Do not promote the spatial branch or infer generalization from memorized train
performance. No cutoff search, extra epoch, changed loss, dev/held access, video
fusion or automatic successor was run. Any later work needs a separately
justified premise addressing the remaining mask-boundary/false-area error.
