# Corridor-relative RGB/ToF representation: one Development pilot

Status: freeze before feature extraction/fit; EXPLORE, simulation only.

Question: can a range-interval-conditioned, position-independent local image
representation recover Boundary positives at a smaller added false-alert cost
than the retained Spatial-BCE supplement? A is unchanged strong + nonrecursive
one-sample hold. B comparison is the already frozen supplement at 7.6612162590026855.
Neither old negative terminals nor the demo are reopened.

The new package uses native 640x360 RGB, original 64 axial ranges and original
zone boxes. Each complete original interval defines nested corridor projections:
the common angular interior at the far in-corridor endpoint, an uncertain band
between near/far projections, and the angular exterior. These are image-query
regions, NOT object masks, ownership, certain clearance or measured surfaces.
An interval extending beyond the corridor depth has an explicit depth-fraction
feature; its common angular interior is not called physically definite support.
Within each zone, extract RGB mean/std, five grayscale quantiles, mean absolute
x/y grayscale gradients and band fraction (14 per band), two RGB mean contrasts,
eight zone-relative projected boundary coordinates, original interval endpoints,
depth fraction and validity: 60 features. Invalid zones remain explicitly masked.
No native depth, geometry truth, object identity, image component or label enters
feature extraction. The complete range support and independent A evidence remain.

One shared 60->32->32->1 ReLU MLP, fixed max over valid zones, no zone index or
Flatten. Max over no valid zones is -20; no missing range becomes a distant return.
No pretrained encoder; no metric-depth/mask output. This changes the complete
representation package; without a matched unaligned ablation no claim about the
isolated contribution of alignment, resolution or aggregation is permitted.

Train on the original 24 Spatial-BCE train layouts (1728 frames), select on its
8 dev layouts (576), evaluate once on all 16 previously consumed complement
transfer layouts (1152). The original 8-layout protected test stays closed.
Complete layouts and all pre-entry/post-exit negatives are retained. This is
disclosed reused Development, not fresh confirmation. One seed 20260921, ordinary
BCE, AdamW lr .001 / wd .0001, batch64, 1200 updates, last checkpoint only;
no augmentation, architecture/seed/loss/threshold retries. Same-workload CPU/GPU
probe chooses placement using research_backend; actual backend/timing recorded.

Candidate R_current = A_current OR new logit >= t; R_hold uses the unchanged
nonrecursive .2s hold. UNKNOWN always equals A, including learned-only positives.
OR cannot remove existing A false alerts. Goal is bounded ADDED cost, not removal.
New disclosed tradeoff contract (not a retroactive relaxation of any old gate):
for Core and Boundary separately, additional current FP <= floor(.01*Nnegative),
additional hold FP <= floor(.02*Nnegative), and added false segments <= floor
(.125*Nclips) for each readout. On dev choose the lowest inclusive unique score
meeting every cost cap (also include a disabled score above max); this monotone
choice maximizes coverage within the fixed budget. Scores must be finite.
Apply that single dev threshold unchanged to transfer; never use transfer to tune.

Keep as research challenger only if transfer meets all the same cost formulas,
retains every A positive decision, Boundary hold recall >= .50, and gains positive
Boundary frames in >=8/16 layouts. Compare frozen old B at its own old operating
point and report both its greater coverage and costs; do not imply equal operating
points. Report TP/FP/FN, precision/FPR, event coverage/timing/release/segments and
BODY/HEAD/background/layout groups. Also disclose new TPs without native-return
target-corridor contributors using the already sealed evaluator admission only
AFTER predictions are sealed. These are classifications, not measured ranges.

Failure closes this exact package/fit/selection recipe; no successor or test opening.
Success retains a research component only, with no app/default promotion. Finish
scoped validation, provenance, result/current notes, commit/push and process release.
Mechanical faults may be fixed with preserved receipts before any changed recipe;
no silent overwrite or second fit. Existing ledger303 failure is recorded, not fixed.

Feasibility: Spatial-BCE already shares convolutions; pixel-query already used native
RGB and corridor queries. New here is explicit pre-learning interval-relative band
statistics plus permutation-invariant aggregation. MZ136/MZ137 show why image edges
must not shrink range support; RGB-multizone's whole-zone anchors are not reused.
Frustum PointNets motivates geometric canonicalization conceptually, not our results:
https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf
It uses real point-cloud proposals; this pilot has coarse ambiguous ToF supports.
