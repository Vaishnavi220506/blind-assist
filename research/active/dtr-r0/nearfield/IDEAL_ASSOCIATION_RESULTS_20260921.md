# Whole-target ideal-association diagnostic is not evaluable under its contract

2026-09-21. **NOT_EVALUABLE_WHOLE_TARGET_ORACLE.** The authorized feasibility
preflight completed on all 1152 consumed transfer frames. The proposed ideal
association does not supply enough legally admissible surface-depth constraints
to define the promised whole-obstacle geometric test. This is a limitation of
the proposed diagnostic contract, not a negative result for RGB, segmentation,
ToF, the original supplement, or all possible geometry methods.

[Frozen preflight](IDEAL_ASSOCIATION_PREFLIGHT_20260921.md),
[runner](audit_ideal_association_preflight.py),
[focused checks](test_ideal_association_preflight.py).

## What can actually be reconstructed

All 1152 saved native depth files passed their recorded SHA-256 checks. The
original source's visible target counts were reproduced exactly in all frames.
The 64 public ranges, observed/missing states and winning contributor pixel IDs
agree with saved lineage; no new sensor data or return is synthesized.

The source target mask is reconstructed by native-depth backprojection into the
authenticated target bounding box with the existing 2 cm tolerance. It is not an
independently rendered per-pixel instance-ID mask, and it does not cover occluded
surfaces. Reproducing source admission does not upgrade it to perfect semantic
segmentation. This provenance alone limits the planned "perfect contour" claim.

More fundamentally, winning-bin contributor IDs only associate an existing
return with some sampled rays. The original interval may describe those rays
under the inherited noise/scatter assumptions. It provides no depth bound on
other parts of the same object. Neither the native inverse-depth weights nor
the winning-bin number may repair this: both encode privileged distance.

The coverage function receives only a 2D mask, public boxes/ranges, zone IDs,
observed states and contributor indices. A strict field whitelist excludes
native depths, true 3D bounds, winner-bin indices, weights and labels. The mask
adapter alone sees evaluator geometry/depth. Coverage was sealed before saved
alerts and truth were joined for the following subset summaries.

## Coverage on the exact added-error/rescue populations

Counts are sample incidences across frames, not unique objects or obstacles.
Subsets use original B current additions relative to A current.

| Subset | Frames | Target samples on existing lattice | With observed-return association | Without association | Frames with complete sampled coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| All cohort | 1152 | 459427 | 387559 | 71868 | 108 |
| Added depth-valid lateral FP | 13 | 6314 | 5511 | 803 | 1 |
| Added distance FP | 15 | 6216 | 5502 | 714 | 1 |
| Rescued Boundary TP | 118 | 65692 | 55778 | 9914 | 0 |
| Rescued Core TP | 2 | 582 | 432 | 150 | 0 |

The 803 missing lateral-FP target samples divide into 310 in zones without an
observed return and 493 in observed zones outside the winning contributor set.
None are outside the ToF footprint. For the 118 Boundary rescues, the 9914
missing samples divide into 715 outside the footprint,3743 in unobserved zones,
and5456 noncontributors in observed zones.

Across the complete cohort, these three missing categories contain6602,24970,
and40296 samples respectively. Only two cohort frames have no target-associated
return at all; none of the incremental error/rescue subsets lacks all target
returns. Thus "some target return exists" is different from "the obstacle's
spatial extent is constrained" and from "a returned target point is in-corridor".

Native visible masks contain2141763 pixels versus387559 target contributor
sample locations. Zero frames have complete native-pixel coverage. That latter
fact is expected from sparse sampling and is **not** by itself an information
limit or a requirement that every pixel be measured by a practical model.
Even complete coverage of the sampled lattice would not certify continuous or
occluded obstacle extent. The one fully sampled lateral-error frame therefore
cannot be declared removable from this count alone.

## Why the planned geometric classification did not run

A full contour plus a return is not a metric shape. Associating the return
correctly does not authorize copying its range over the whole silhouette.
The promised test forbade constant-depth/planar assumptions, unmeasured-depth
interpolation and use of evaluator 3D extent as the answer. With those rules,
outside-only returned support cannot certify that the whole target stays outside.

The protocol therefore stops at source/coverage feasibility. No frames were
declared removable FP or preserved TP by a new oracle classifier. No alert,
threshold, hold, UNKNOWN state, model or demonstration changed. No supplemental
capture, protected test, new training or successor was started.

This does not show that accurate contours cannot help a learned decision, that
additional observations are necessary, or that RGB plus 64 ranges intrinsically
lacks separable information. It shows that this specific proposed whole-target
oracle is underspecified for an admissible upper-bound claim. A future method
would need an explicit, separately evaluated rule for unmeasured surface depth;
such a rule was not introduced in this check.

## Validation and disposition

Three focused checks pass: missing categories form a nonoverlapping partition;
sampled coverage does not imply native coverage; an unobserved zone cannot
supply contributors. System Python initially lacked NumPy; the check and run
used the existing research environment without installing packages. The actual
run used CPU NumPy mask/metadata reductions (TASK_NOT_GPU_SUITABLE), not a GPU
or model. All parent observation-seal files,1152 native hashes, source mask
counts and range/lineage identities passed. No task-owned persistent resource
remains.

Evidence under `artifacts.local/work/ba-ideal-association-preflight-20260921/`
contains the input protocol/hashes, sealed per-frame coverage, post-seal subset
joins, per-layout totals and final results. Retain COMPONENT evidence for the
coverage/provenance blocker; do not assign a method NEGATIVE_CONTROL from a test
that was not evaluable. Prior A/B/R/U/G dispositions remain unchanged.

Independent read-only review passes the input whitelist, source-mask formula,
1152 per-frame partitions, five subset summaries,16 layout totals and output
hashes. Supported registration still fails at the existing ledger303 fingerprint
mismatch and inheritance reports an unknown terminal. Local disposition and
command receipts preserve this metadata gap without editing the global ledger.
