# MZ138: evaluator-only ToF support ceiling

**Result: true complete-surface extent reaches24TP/2FP/0FN versus MZ12924/13/0,
with all five event times unchanged. ToF geometry has useful headroom; retain
MZ129 and do not promote an oracle. The bottleneck exposed here is completing
unsampled surface extent, not further edge MAE tuning or demonstrated gains
from splitting already sampled ownership.** The complete-surface arm corrects
an initially too-narrow sample-envelope definition, disclosed below.

EXPLORE, 2026-09-15. One fixed diagnostic on all **48 consumed MZ136 dev
frames**. No training, thresholds, edge refinement, capture, original-test
scoring or new observable algorithm. MZ137 rejected a specific cohort-plane
replacement, not every ToF support representation. This run asks whether
better ToF support can change final alerts with the rest of MZ129 frozen.

## Initial frozen definitions (full-surface scope correction below)

Use only native contributors linked to an actually returned ToF slot through
`returned_lineage.target_index -> hit_indices -> private_rays.subray`. Never add
undetected rays/components. All oracle outputs are explicitly evaluator-only;
none may enter the observation-only predictor or be promoted as an algorithm.

| Main arm | Replaces per-return ToF support |
| --- | --- |
| Angular oracle | Native contributor theta/phi min/max envelope, original measured range interval and original public pitch/yaw uncertainty; same interval enclosure routine as MZ129. No RGB viewport clipping. |
| Initial sampled ownership + extent | Group linked native points by actor and inferred axis-aligned box face; use union of each group's true XYZ sample envelope. This was insufficient for the requested complete-surface arm; see correction below. |
| Full native support oracle | Direct corridor membership of every linked native point; no bounding rectangle or plane. |

An additional **pooled native XYZ envelope** readout uses the same true points
and depth as ownership+extent but no grouping. It is an attribution bridge,
not a fourth proposed algorithm. Only grouped-versus-pooled gain isolates
grouping. Angular-versus-native additionally changes range/orientation authority.

These are ceilings for **saved discrete returned contributors**, not complete
continuous surfaces, all sensor information or hardware. Angular min/max plus
the existing Cartesian interval enclosure can include unobserved combinations;
its failure would reject neither an exact angular set nor every frustum model.
Native extent does not extend a sampled face into an unobserved full object.

Infer a face only when a native hit is within 1e-5 m of a native box min/max
face and lies within its bounds. Tag multiple-face ties explicitly. For an
unresolved face/actor, retain an unsplit UNKNOWN-face group and count it; no
ground-truth source identity is invented. Missing lineage/points: keep the
entire incumbent return support and count fallback. Zero public returns remain
zero; missing data never becomes CLEAR. Retain VALID and MERGED denominators.

## Frozen final aggregation and checks

One original return remains the unit of certainty: possible=ANY sub-support
intersects corridor, certain=ALL sub-supports lie wholly inside, nonempty.
Full-native support uses ANY/ALL linked points. No split creates a new return
or an unweighted certainty vote. Use the original binary-four weights with
zone deduplication, score>=1, OR any per-return certainty, OR the **unchanged
MZ129 Radar-only candidate including all its guards/raw/carry**. Do not OR
the complete old alarm or pre-MZ129 Radar field back in. Reconstruct baseline
exactly before interpreting contrasts. Report ToF-only and fixed-Radar results
as well as final alerts and per-FP residual causes.

Retain original dev acceptance: FP<=10 (>=20% reduction from13), recall within
2pp (24 positives requires zero FN), all five events, each relative first alert
delay<=.25s. Report all TP/FP/FN/TN/UNKNOWN, precision/recall, family and shallow
versus other-family strata, <1cm versus >=1cm native target gap/overlap, false
segments/duration, binary pair both-correct, changed return/zone/frame support,
native corridor contributor retention and each removed/new error.

Freeze input/method/source receipts before computing this one oracle contrast.
Outputs necessarily consume evaluator truth; sealing them before frame scoring
does not make them observation-only. Replay exact outputs; verify input hashes.
CPU scalar geometry/metadata uses TASK_NOT_GPU_SUITABLE. No persistent workers.

## Decision rules and stop

- Angular meets joint target: retain evidence of angular headroom; a subsequent
  observable edge-clipped frustum could be worth proposing, not executing here.
- Only native extent meets target: use pooled/grouped difference to distinguish
  range/geometry from ownership. Do not infer ownership benefit without it.
- Full native cannot meet target: inspect fixed Radar floor, fallback and missing
  contributors. If fixed Radar alone prevents FP<=10, close ToF-only refinement
  for this fixed panel/aggregation goal. Failure alone is not a global ToF ceiling.
- All results remain diagnostics. Keep MZ129 and existing component authority;
  no successor algorithm, tuning or new validation is included in this run.

Registration/inheritance are separate supported-command attempts. Preserve
ledger303 pending metadata and completed technical evidence independently.

## Scope correction before completing the requested full-surface oracle

The initial sealed diagnostic implemented ownership+extent as sampled-point
envelopes, which is narrower than the user's requested **true surface extent**.
It produced angular22/3/2 and sampled-native21/2/3; three positive frames have
no returned point inside the corridor. Do not use those sample-only misses to
conclude that complete surface knowledge cannot retain them.

The existing native source supports full-face truth: `mz115_zonal_capture.py`
loads `/Engine/BasicShapes/Cube`, uses unrotated collision actors scaled to
`size_m`, and non-colliding texture tiles. `mz113_dynamic_sensors.py` records
the base actors' native bounds. Complete the requested second oracle with the
**union of verified complete native cuboid faces** attached to each existing
return. Each face fixes one native coordinate and uses both full other native
bounds. Do not replace the union by a full-object volume or clip it to samples,
zones or RGB; this is explicitly evaluator-only unsampled surface completion.

Keep per-return ANY-possible/ALL-certain, original zone weights and Radar/guards.
Any unknown/ambiguous face gets incumbent-return fallback. This correction is
made after observing the initial results, with no tuned parameters. Preserve
`ceiling-v1` verbatim and freeze the separate `full-surface-extent-v1` completion
before calculating its outputs. It completes the requested oracle definition;
it is not an independent confirmation or an observable successor algorithm.

## Final comparison: same consumed dev48 and frozen MZ129 aggregation

The requested second main arm is **complete native surface extent**. Keep the
initial sampled-only arms as separate diagnostics; do not overwrite their
outputs or mislabel their original definition as complete geometry.

| Final-system arm | TP / FP / FN / TN | Precision / recall | Joint dev criterion |
| --- | ---: | ---: | --- |
| MZ129 | 24 / 13 / 0 / 11 | 64.86% / 100% | Baseline |
| Oracle angular envelope + original range intervals | 22 / 3 / 2 / 21 | 88.00% / 91.67% | Fail recall |
| **Oracle ownership + complete native face extent** | **24 / 2 / 0 / 22** | **92.31% / 100%** | **Met as oracle only** |
| Oracle full native returned points | 21 / 2 / 3 / 22 | 91.30% / 87.50% | Fail recall |
| Attribution: pooled native sample XYZ | 21 / 2 / 3 / 22 | 91.30% / 87.50% | Fail recall |
| Attribution: actor/face-grouped native sample XYZ | 21 / 2 / 3 / 22 | 91.30% / 87.50% | Fail recall |

Angular removes10/13FP (76.92%) but loses two positives. Complete faces remove
11/13FP (84.62%), lose no positive and add no false positive. Every arm retains
all5/5events with **identical first-alert timestamps**, but event retention does
not excuse the sampled/angle arms' frame misses. No thresholds were selected.

False segments/duration are4/3.25s for MZ129,1/.75s angular, and1/.5s for all
native-coordinate arms. Duration counts .25s sampled bins, not user interruption
time. No-alert/UNKNOWN counts are11 baseline,23 angular,22 complete-face and25
sampled-native. Binary pair both-correct is11/24 baseline,19/24 angular and
sampled-native,22/24 complete-face; it is not learned continuous-score ranking.

### Where the useful change comes from

All653 returned slots are evaluable:637SIM_VALID and16SIM_MERGED, with4,187
linked native contributors including1,046actual corridor points. There are no
fallbacks, unknown faces or ambiguous face ties. All three main oracles retain
every recorded native point and every recorded corridor point; complete-face
containment uses the disclosed1e-5m face-match/audit tolerance, without widening
the faces for alert geometry. This avoids MZ137's native-support loss.

There are **zero multi-actor returns** and15multi-face returns on this panel.
Pooling versus splitting the same native points changes **zero per-return
possible bits and zero final alerts**. It therefore supplies no evidence that
ownership splitting is the missing gain here. This does not establish return
purity or trivial association on other sources, including other MERGED returns.

Angular changes120return decisions and17frame zone sets, removing10FP and2TP.
Sampled-native arms change187return decisions and29frame zone sets, removing
11FP and3TP. Complete faces then restore exactly the following three positives,
without restoring any FP:

| Shallow-boundary frame suffix | Native target intrusion | Returned corridor contributors | Point oracle | Complete-face oracle |
| --- | ---: | ---: | --- | --- |
| `scene4_enter_05` | 31.472 mm | 0 | No alert | Alert |
| `scene4_exit_01` | 18.883 mm | 0 | No alert | Alert |
| `scene4_exit_02` | 6.294 mm | 0 | No alert | Alert |

All have real returned points on the target's native front face, but none on
the part entering the corridor. Their five returned slots are zones26/34/42/50/58;
native face identity is `shape0 / 0:MIN`. The full-face oracle knows that this
surface continues into the corridor. The point oracle cannot create that
unsampled part. Neither case is a lost sampled contributor or a zero-weight
aggregation rejection; the missing support is between/outside sampled hits.

Consequently, the returned-point oracle is **not a global perfect-geometry
upper bound**. It is exact only about the returned samples. Inferring object or
surface extent can legitimately exceed that sampled-point recall. Do not close
the ToF refinement route because its point-only oracle loses these positives.

The complete-face result is not a rescue by moving a fitted plane or tuning a
padding margin: the collider's actual fixed face and full native bounds are
used. Unlike MZ137's cohort-median artificial plane, these are known surface
poses/extents. This privileged information is absent at inference time.

### Remaining false alerts and independent sensors

The fixed Radar-only branch is13TP/2FP/11FN. Its two FPs are
`mz136_shallow_boundary_stress_scene4_exit_03` and `_exit_04`; both have raw
Radar and corrected-current Radar support, with no carry or guard vote.
Complete-face and point-oracle ToF are false on both. Thus2FP is the fixed-Radar
floor for this comparison, not13FP; it does not prevent the <=10FP target.

ToF-only counts further separate roles:

| ToF readout only | TP / FP / FN |
| --- | ---: |
| MZ129 | 22 / 12 / 2 |
| Angular | 20 / 2 / 4 |
| Sampled native (pooled/grouped/points) | 18 / 0 / 6 |
| Complete native faces | 22 / 0 / 2 |

Radar still supplies the two positives absent from complete-face ToF and the
earliest shallow-entry warning. Complete-face ToF alone delays that event .25s;
the unchanged final system has zero relative delay. Sensor complementarity is
preserved rather than credited to the oracle alone.

### Pressure and non-pressure breakdown

| Stratum | Frames | MZ129 TP/FP/FN | Angular | Native points | Complete faces |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shallow-boundary family | 12 | 6/6/0 | 4/3/2 | 3/2/3 | 6/2/0 |
| Other three families | 36 | 18/7/0 | 18/0/0 | 18/0/0 | 18/0/0 |
| Native absolute gap/overlap <1cm | 4 | 2/2/0 | 1/1/1 | 1/1/1 | 2/1/0 |
| Native absolute gap/overlap >=1cm | 44 | 22/11/0 | 21/2/1 | 20/1/2 | 22/1/0 |

For HEAD/BODY/rod individually, every oracle ends6/0/0 in each12-frame family;
MZ129 was6/0/0,6/1/0,6/6/0 respectively. Improvements are not limited to
millimetre boundary errors. The original full48 denominator and labels remain
unchanged; no subset replaces the joint acceptance gate.

## Disposition and next decision boundary

Retain **oracle evidence of ToF surface-extent headroom**, not an alert algorithm.
Keep MZ129 as baseline, MZ136's conditional edge component and MZ137's scoped
negative result. The executed evidence favors a question about **how measured
support extends over a continuous surface beyond its sampled hits**. It does
not demonstrate a gain from native ownership splitting on this pure-return panel.

A later observable proposal could use RGB extent and retained ToF range volume,
but a frustum that only shrinks to sampled angular support cannot be assumed to
retain the missed surface portions: even this angular oracle loses two frames.
It must explain supported surface extent and preserve UNKNOWN and independent
out-of-RGB evidence, rather than optimize MAE or simply tighten every volume.
This is a future decision boundary, not authorization exercised here for another
algorithm. No edge fitting, head training, capture or original-test run followed.

## Verification and registration

Artifact root: `artifacts.local/work/mz138-support-ceiling-20260915/`.
`ceiling-v1/` contains the initial freeze, native oracle outputs, all48cases,
return/contributor records, full metrics, CPU record and exact replay receipt.
`full-surface-extent-v1/` contains the explicitly posthoc scope-correction freeze,
source snapshots, complete-face per-return outputs, metrics and replay receipt.
Both sources and original caches are unchanged; MZ129 is reproduced48/48.
Both output sets replay exactly. Six focused MZ138 tests pass, covering
group-bridge removal, per-return certainty, zone deduplication, frozen Radar,
lineage/fallback coverage, face ambiguity and complete unsampled face extent.

Initial oracle computation took0.2166s; complete-face computation0.04256s,
excluding input IO, cached MZ129 computation and scoring. These are scalar CPU
diagnostic timings, not live system latency. No worker or paid allocation remains.
The backend selector records `CPU_TASK_CLASS_SCALAR_SCORING` for the requested
`TASK_NOT_GPU_SUITABLE` workload; no GPU inference is claimed.

Initial output SHA256:
`175c1618b891ddbdff18edfb933ff48815e62e45245d288afe672af75086d052`.
Complete-face output SHA256:
`264340afc72beba34a55dc633cd2a82e90071d91e04df7d8eda722285c068fea`.

Intended terminal `MZ138_TRUE_SURFACE_EXTENT_HEADROOM_ORACLE`, intended role
`COMPONENT_OR_CHALLENGER / COMPONENT`, **diagnostic evidence only** on consumed
dev48. Supported registration and inheritance are attempted separately; pending
metadata must not be called completed registration or algorithm promotion.

Registration failed on the existing `experiments/index.jsonl:303` fingerprint
mismatch; inheritance then failed with unknown terminal. Both outputs are
retained in the artifact root. Registration and inheritance remain **pending
metadata**; no shared-ledger edit or bypass occurred. This does not change the
completed oracle result or supply observer-side algorithm authority.
