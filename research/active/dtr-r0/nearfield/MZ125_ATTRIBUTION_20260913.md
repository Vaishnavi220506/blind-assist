# MZ125 incumbent alert-source attribution

Date: 2026-09-13. `EXPLORE / CONSUMED_EVALUATOR_DIAGNOSTIC`.

The focused question is which observable support branches produce the typical
MZ116 false alerts and which true alerts would be exposed by correcting those
branches. This audit does not propose a runtime identity oracle, train a model,
alter the frozen baseline, or establish fresh validation. The root's observable
correction contrast is separate.

## Branch attribution

All 288 MZ123 frames were matched by ID across raw observations, native geometry,
Radar provenance, sealed MZ116 cache and original frame report. Native AABB
labels reproduce `139 TP / 116 FP / 5 FN / 28 TN`. For every frame the incumbent
flag equals the OR of common Radar, resolution guard and active ToF support.
All 288 resolution guards are inactive.

| Active branch combination | TP | FP |
| --- | --- | --- |
| ToF only | 15 | 80 |
| Common Radar + ToF | 100 | 25 |
| Common Radar only | 24 | 11 |
| Total | 139 | 116 |

Of the 80 ToF-only FP, 25 are rod/farwall, 30 BODY, 12 HEAD and 13 boundary
stress. The 15 ToF-only TP are 3 rod/farwall, 2 BODY, 1 HEAD and 9 boundary
stress. Correcting ToF is a substantial opportunity, but turning it off or
requiring current Radar would remove independently supported true frames.

The sealed common-Radar cache is a frame-level aggregate; it does not retain
the precise triggering slot and decision path. This audit records every valid
slot's observed range/bearing, nominal point and evaluator provenance, but does
not claim an individual slot triggered the aggregate merely because it exists.

## Native returned-ToF attribution

The audit follows native `returned_lineage.target_index` and private hit indices
for all **3,954** ToF return records, verifying packet status/range and allocation
contract consistency. Source identity comes exclusively from native actor IDs;
RGB box overlap does not assign identity. All checks report zero violations.

| Active supporting returns | In TP frames | In FP frames |
| --- | --- | --- |
| Total | 962 | 586 |
| SIM_VALID | 926 | 356 |
| SIM_MERGED | 36 | 230 |
| Whole-zone unresolved | 455 | 434 |
| RGB signal association proxy | 507 | 152 |
| Contains native hazard-actor contributor | 879 | 0 |
| Contains native hit point inside corridor | 620 | 0 |
| Contains multiple native actor IDs | 0 | 7 |

The actor-intersection and actual-hit labels are distinct: an obstacle volume
can intersect the corridor while its sampled return hits a surface outside it.
These controlled native rays are not an enumeration of physical optical paths.

Most merged nuisance returns are **single-actor**, not multi-object: 223 of 230
active merged FP returns have one native actor ID. A merged-status gate cannot
be interpreted as removing only confused object associations. Nor does being
RGB-associated certify correctness: 152 active FP returns already have that
association state.

Three nominal TP frames have **only farwall-derived active ToF evidence** and
no common Radar support: `mz123_near_rod_farwall_pair0_in_03`,
`mz123_near_rod_farwall_pair1_in_06`, and
`mz123_near_rod_farwall_pair2_in_08`. Their active farwall-return counts are
5, 2 and 1. An additional 23 joint Radar+ToF TP frames have no hazardous-actor
ToF contributor. Thus some correct frame alarms have the wrong ToF source.
A source-correcting method could lose those incidental TP; frame recall alone
must not be confused with correctly identified obstacle evidence.

## Typical false frames and matched positive evidence

Examples are selected by the middle ordered FP in each existing family/branch
group, using evaluator metadata solely for explanation. Every negative frame
has its exact same-time positive pair in `examples.json` and `pairs.json`.

| False frame | Active reason and native source | Same-time positive |
| --- | --- | --- |
| `near_rod_farwall_pair1_out_04` | ToF-only; 9 valid whole-zone supports: 6 from off-corridor rod and 3 from farwall | `pair1_in_04`: Radar + ToF |
| `substantial_body_pair1_out_07` | ToF-only; 10 merged whole-zone supports, all from the single off-corridor BODY actor | `pair1_in_07`: Radar + ToF |
| `suspended_head_pair0_out_06` | Common Radar only; no active ToF; native provenance identifies off-corridor HEAD, nominal lateral point .456 m | `pair0_in_06`: common Radar only |
| `shallow_boundary_stress_pair1_enter_02` | Radar + ToF; 6 valid RGB-associated ToF returns from off-corridor boundary actor; Radar slots include a transient and the actor | `pair1_exit_02`: Radar + ToF |

All IDs in this table have the `mz123_` prefix. Nine family/branch examples are
fully recorded, including source roles, active returns and paired positives.
For HEAD and boundary examples, the observed nominal Radar point can be outside
the corridor while the inherited aggregate still alerts; exact internal
common-Radar attribution needs its detailed predecessor outputs, absent from
this sealed cache. No actor identity is available to a runtime correction.

## Exact observable collisions

The explicit packet signature excludes frame/episode IDs, time, paths, family,
native truth and source identities. It includes all zone bounds, valid ToF
range/status/signal/noise fields, packet flags, valid Radar range/angle/velocity,
IMU deltas/validity, rig calibration and RGB intrinsics. Invalid/unreceived
slots contribute availability only. RGB pixel content is not part of this
sensor-packet comparison.

| Signature | Unique packets | Duplicate groups | Mixed-truth groups |
| --- | --- | --- | --- |
| Sensors + IMU + calibration | 288 | 0 | 0 |
| Sensors + calibration, without IMU | 280 | 4 | 2 |
| Radar packet only | 204 | 39 | 7 |

One non-IMU mixed-truth group includes HEAD `pair0_in_02`, `pair0_in_07`,
`pair0_out_01`, `pair0_out_09`, and `pair2_out_01`: two positive and three
negative frames have identical sensor/calibration signatures without IMU.
Their differing IMU values eliminate exact full-signature duplicates. This
does not establish that the differences contain useful obstacle information.
Conversely these partial-signature collisions do not establish a full RGB +
sensors information ceiling. The complete collision membership is archived.

## Verification and reproduction

Code: [mz125_evidence_attribution.py](mz125_evidence_attribution.py).
It reuses the evaluator-only native-return auditor
[mz115_allocation_audit.py](mz115_allocation_audit.py).

```powershell
python research/active/dtr-r0/nearfield/mz125_evidence_attribution.py --source artifacts.local/work/mz123-frozen-early-20260913/returned-v1 --output artifacts.local/work/mz125-observable-correction-20260913/attribution-v1
```

Output must be new; use a new revision when reproducing. Artifacts include
`frames.json`, `returns.json`, `pairs.json`, `examples.json`, `collisions.json`,
`summary.json`, `backend.json` and `completion.json`. All consumed source hashes
are verified against receipt/seal and checked unchanged after attribution.
Measured execution was 2.074 seconds, Python 3.14.5 on CPU,
`TASK_NOT_GPU_SUITABLE` for scalar metadata. No model runtime, worker or GPU
allocation was started.

The audit favors examining the geometry/uncertainty of ToF-supported nuisance
alerts while preserving independently credible evidence. It does not support
dropping all merged returns, trusting association state as correctness, using
source identity as a feature, or interpreting unique noisy packets as proven
separability. Registration and route decisions remain with the root task.
