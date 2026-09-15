# MZ139: observable regional-ToF constrained finite-surface fitting

EXPLORE, 2026-09-15. Implement one observable method, check/fix on TRAIN,
then freeze it before the **already consumed dev48**. No oracle extension,
new capture, edge-parameter search, alert-head training or original-test scoring.

**Completed: KEEP_MZ129_NO_CONSUMED_DEV_JOINT_GAIN.** The runnable observable
slab estimator leaves dev48 at24TP/13FP/0FN, with5/5events and unchanged times.
It supports one of the three unsampled intrusions, but has limited coverage,
unresolved extent ambiguity and a confidently wrong accepted surface on a TP.
Retain this exact estimator as a negative control, not an alert challenger.

## Hypothesis and boundary

MZ138's true-point geometry removed11FP while losing3TP; complete true surface
extent restored those3TP without addingFP. This does not attribute all11FP
removals to completion or prove observable recovery. MZ139 asks whether a small
finite-surface hypothesis can jointly explain measured zonal returns and RGB
bounds and improve alerts, including the three unsampled intrusion cases.

The shape prior is an upright, horizontally oriented finite cuboid/slab, eight
parameters: center xyz, three log half-extents, yaw and log constant reflectance.
All parameters are fit from RGB and public returns. No native position, face,
size, actual ray/lineage, evaluator pose, per-point range or histogram is input.
The prior does not assume the native cuboid's orientation, size or boundaries.

## Fixed implementation before TRAIN check

Use existing MZ125 observed boxes and optional MZ136 edge estimates. Fine-edge
availability is not a gate. Public near-return cohorts and box intersection
nominate up to two seed/return combinations, requiring at least two distinct
single-return VALID zones; no source identity or label chooses the association.

For each candidate geometry, generate3x3 midpoint rays from each public zone's
angular bounds under the known hypothetical sensor model. Transform using
public pitch, integrated IMU yaw and camera offset. Ray/slab first intersection
within4m predicts hits; constant fitted reflectance supplies inverse-square
weights. A batched single-component expected-mean/strength surrogate plus
projected silhouette bounding-box loss drives derivative-free population search.
This is a stated single-component surrogate, not hidden access to actual returns'
histograms or an exact differentiable firmware model.

The final candidate must pass the original `measure_zone` histogram/peak/merge
reduction on **hypothesis-generated hits**, with no new random noise. Compare
expected mean, public strength and status against each observed single VALID
return. MERGED and multi-return support is not explained by this first model
and retains incumbent treatment. Missing detections/packets supply no absence
penalty or free-space evidence. Optimized reflectance is a nuisance estimate.

Initial search settings:192candidates,64iterations,24elites, fixed seed139015.
Loss is mean squared range residual scaled by reported sigma (floor.04m),
plus.25 times mean squared log-strength residual/.35, plus mean squared RGB
box residual/2.5px. Missing model hits are penalized; no ground-truth labels enter.
Final loss<=4, maximum box residual<=5px, maximum range residual<=3sigma.
Exact reduction additionally requires a single VALID predicted component,
mean within3sigma+.01m, and absolute log-strength ratio<=.7.

Keep elite solutions plus fixed perturbations in depth/extent/yaw/reflectance;
loss<=best+1 defines a finite plausibility check. Across plausible solutions and
competitive seed fits, any disagreement about corridor intersection forces
incumbent fallback. This finite search is not a calibrated/global uncertainty
bound. It must not be described as proof of unique surface recovery.

Compute continuous oriented-slab/corridor intersection by a separating-axis test,
not by testing only the generated ray hits. Replace only exact-model-explained
single VALID slots whose complete original zone footprint is inside RGB, with
at least two such zones. Unmatched, MERGED, missing, and out-of-RGB evidence
retains its original vote. No complete-incumbent OR is added on accepted slots.
Keep binary-four zone weights, per-zone deduplication, certainty semantics and
the complete frozen MZ129 Radar-only branch/guards unchanged.

## Acceptance and evidence

One TRAIN pilot checks implementation and settings before full dev48. Any
TRAIN revisions are disclosed and prior attempts retained. The final model and
settings are frozen before dev prediction/scoring; no dev-driven retry follows.
Required dev target stays24TP,FP<=10,5/5events and each additional delay<=.25s.

Report all48frames: TP/FP/FN/TN/UNKNOWN, precision/recall, families, shallow vs
other families, native gap/overlap strata, events/timing and false duration,
candidate/exact-explanation/ambiguity coverage, removed/new errors, and whether
the new surfaces themselves support the three known unsampled intrusion cases.
Audit replaced native contributors at numerical tolerance and separately at
.04m sigma-scale proximity; proximity is not exact native support retention.
Native truth is evaluator-only after observation predictions are sealed.

GPU-first batched geometry is benchmarked against equivalent CPU work before
choosing placement, using `research_backend.py`; actual device/timings retained.
No persistent allocation remains after execution. Artifacts stay under
`artifacts.local/work/mz139-surface-fit-20260915/`. Ledger303 remains a separate
registration/inheritance blocker if supported commands still fail.

## TRAIN checkpoint before dev access

The fixed12-frame TRAIN pilot completed with no parameter or model revision:
2accepted,3corridor-ambiguity fallback,4measurement/RGB-misfit fallback and
3without an admissible candidate. Both arms9TP/3FP/0FN on these selected fragments.
Four returns/nine contributors were replaced;7contributors were outside the fitted
volume at1e-6m, including3corridor contributors (also outside at.04m proximity).
No corridor-positive original return was replaced by a nonalerting surface.
Thus low residual does not establish physical recovery. Keep settings and record
the same audit on dev; do not relax fallback to create apparent coverage.

The real192-candidate regional-render/loss benchmark used16forward calls per
probe: CPU median.0143163s, CUDA.0304400s. Backend selector recorded
`CPU_FASTER_MEASURED`; dev uses CPU. Two initial benchmark harness errors
(cache index and a first frame without candidate) were corrected before the
successful comparison; no fitting settings or native inputs were involved.
Pilot freeze and its completed receipt are required by the dev invocation.

## Consumed dev48 result, no subsequent fit or threshold change

| Arm | TP / FP / FN / TN | Precision / recall | Events | Extra event delay | False-alert bins |
| --- | --- | --- | --- | --- | --- |
| Frozen MZ129 |24 /13 /0 /11 |64.86% /100% |5/5 |reference |3.25s |
| Observable surface fit |24 /13 /0 /11 |64.86% /100% |5/5 |0s for every event |3.25s |

All48final decisions and all48ToF-only decisions are unchanged. The ToF branch
is22TP/12FP/2FN in both cases; Radar supplies the same missing alerts. Both arms
have11UNKNOWN/no-alert outputs, not certified clear space. Joint target fails
because13FP>10. Both-correct paired decisions remain11/24; binary ordering is
not a continuous-score ranking result. There are no removedFP, newFP or lostTP.

Both arms have the following identical TP/FP/FN by family: suspended6/0/0,
substantial6/1/0, rod/wall6/6/0, shallow6/6/0. The4frames with absolute lateral
gap/overlap<1cm give2/2/0; the other44give22/11/0. Thus the unchanged result
cannot be attributed only to sub-centimeter cases. All strata and event records
are retained, including24paired comparisons and all48frame outcomes.

### What actually reached the backend

25/48frames had at least one fit attempt.10/48accepted a surface;23had no
admissible candidate,3failed measurement/RGB residual checks,4lacked enough
exactly explained fully visible returns, and8had corridor ambiguity. Fine-edge
failure was not a hard gate.30/653returns were replaced across10frames.

The accepted surface alone has3TP/0FP/1FN/6TN on its10available frames;
the other38include20positives and must not disappear from this denominator.
Across4frames it removes12old possible-support votes, adds0votes, changes
0ToF decisions and0final alerts:

| Frame suffix | Removed zone IDs | Why final alert remains |
| --- | --- | --- |
| near_rod_farwall_scene4_out_02 |45,53 |Unexplained21,29,37 and outside-RGB61 |
| near_rod_farwall_scene4_out_03 |45,53 |Same remaining ToF zones |
| shallow_boundary_stress_scene4_enter_00 |26,34,42,50 |Outside-RGB58 alone suffices |
| substantial_body_scene4_in_00 (true positive) |29,37,45,53 |Other independent ToF support and Radar |

The first three are3existingFP with some support correctly classified outside;
the last is a wrong local rejection on a genuine TP. The integration does permit
old votes to be withdrawn, but unmodeled evidence still owns the resulting alarm.
This is not evidence that withdrawing that remaining evidence would be valid.

### Three unsampled intrusion cases

All IDs below have prefix `mz136_shallow_boundary_stress_scene4_`. All three
have zero native returned points inside the corridor and frozen Radar=false.

| Suffix | True intrusion | Accepted continuous surface | Range max residual | RGB box max residual | Final |
| --- | --- | --- | --- | --- | --- |
| enter_05 |31.472mm |Yes, intersects corridor |0.96sigma |2.13px |TP |
| exit_01 |18.883mm |Unavailable: plausible surfaces disagree |0.77sigma |1.08px |TP via incumbent ToF fallback |
| exit_02 |6.294mm |Unavailable: plausible surfaces disagree |1.43sigma |1.00px |TP via incumbent ToF fallback |

For enter_05 the estimated continuous slab itself supplies an intersection,
not only Radar or another branch. This is one mechanism witness, not an added
TP versus MZ129: other original ToF votes already warned. Its alternative
surfaces are a finite search ensemble, not certified geometry or unique recovery.
The two other cases remain unresolved despite small observable residuals.

### Native-support audit exposes accepted-model errors

Of4187native contributors (1046corridor),4018keep original support treatment;
169belong to replaced returns.104/169are outside every retained estimated slab
at1e-6m numerical tolerance, including24corridor contributors that were
previously contained. At the separately labeled.04m proximity tolerance,
64remain outside, including18corridor contributors. Proximity is not exact
support retention, and neither tolerance changes the alert geometry.

12native corridor contributors belong to old positive-support returns replaced
by a nonalerting surface. They occur on `substantial_body_scene4_in_00`;
that accepted fit has loss1.80, yet gets the relevant corridor relationship wrong.
The final zeroFN is partly protected by other evidence. Small range/box residual
plus the finite ambiguity test does not establish correct ownership or extent.

Read-only independent audit reproduced all48aggregations, with no invalid slot
replacement or lost Radar vote. The wrong BODY fit explains a narrow observed
stripe `[397,41,402,360]` with a slab centered at y=.416m, halfwidth=.0102m
and fitted reflectance=1.0. The other12excluded corridor contributors occur in
`suspended_head_scene4_in_04`, whose surface still intersects the corridor.
This separates wrong local geometry from final binary alert behavior.

The current limitations are concrete: box/cohort proposal coverage, truncated
RGB boundaries, one constant-reflectance upright slab, limited local search,
single-component surrogate fitting, and incomplete attribution of other returns.
No result here establishes that observable joint surface estimation is impossible.
It rejects this exact first estimator/readout as a demonstrated alert improvement.
MZ138 remains a diagnostic headroom result; MZ129 remains retained core.

## Evidence, verification and disposition

Artifact root: `artifacts.local/work/mz139-surface-fit-20260915/`.
Entrypoint: `run_mz139_surface_fit.py --split train --pilot --device cpu --output
<new-artifact-directory>`, then `--split dev --device cpu --require-train-freeze
<completed-train-directory> --output <new-dev-directory>`. Use the recorded
Python runtime and source snapshot for this version. Existing evidence directories
are immutable outputs; the runner refuses to overwrite them.
`train-pilot-v1/` and `dev-v1/` contain source snapshots, pre-fit protocol,
hash-frozen inputs/settings, observation-only predictions sealed before selected
evaluator parsing, complete case/contributor audits, summaries and PASS receipts.
`dev-v1/geometry-chain.json` is a post-seal audit of existing decisions, not a
new fit. The predictor does not load native actor geometry, private rays or
lineage; those appear only after sealing in the evaluator audit.

Seven focused tests pass: regional weighted forward measurement versus zone
center, finite boundaries, continuous unsampled support, orientation/camera
offset, split selection, numerical/proximity separation and replay payload rules.
Source/input hashes remain unchanged. Full inference replay was **not** performed;
testing a replay helper does not establish full-run reproducibility. No test split
was scored. Incumbent cached results are reused with their seals, not recomputed
under the current cv2runtime.

CPU fitting took5.5351s for48frames (mean115ms including frame processing and
progress writes, excluding initial input/cache loading and scoring). This is an
offline timing, not Android latency. Unchanged simulated event timestamps do not
include computation delay. CPU/CUDA renderer equivalence and timing are recorded
in `backend-equivalence.json` and `backend-selection.json`; no persistent worker,
GPU allocation or paid instance remains.

Dev prediction SHA256 is recorded in `dev-v1/prediction-seal.json`; summary
SHA256 is `8b19908e2ae1c30024316a833f235804083a67529f460c3030993dbde7f61ce9`.
Intended terminal `MZ139_OBSERVABLE_SURFACE_NO_ALERT_GAIN`, role
`NEGATIVE_CONTROL`, scope this frozen estimator/readout on consumed dev48.
Retain executable forward fitting and failed-transfer evidence for comparison;
do not promote or automatically launch a successor. Registration/inheritance
are attempted through supported commands separately from technical completion.

Supported registration failed at existing `experiments/index.jsonl:303` because
its fingerprint does not match inputs; inheritance failed with unknown terminal.
Logs are retained as `registration.log` and `inheritance.log`. Both remain
**pending metadata**, not registered or inherited successfully. The ledger was
not rewritten or bypassed; this does not alter the completed technical result.

## Mechanism references, not transferred performance

The two original project pages were fetched through Exa. [DELTAR](https://zju3dv.github.io/deltar/)
treats light-weight ToF as regional depth information. [ToF-SLAM](https://zju3dv.github.io/tof_slam/)
optimizes scene geometry by comparing rendered RGB/zone signals to observations.
MZ139 borrows the measurement-explanation principle, not their neural systems,
hardware model, depth metrics or performance claims. Only these two project
pages were inspected; no broader literature search or paper-PDF review occurred.
