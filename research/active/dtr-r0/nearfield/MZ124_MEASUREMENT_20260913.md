# MZ124 measurement-centered geometry and accountable correction

Consumed Development, all 288 MZ123 frames. These are fixed geometry contrasts
under the [MZ124 protocol](MZ124_PROTOCOL_20260913.md), not a new trained model
or a replacement for MZ116. [Code](mz124_measurement_geometry.py) and
[six focused checks](test_mz124_measurement_geometry.py) retain the original inputs.

## Measurement hypothesis

Cartesian interval boxes can include locations that the original finite-angle
range return cannot reach. Rather than assigning a return to a single visual
object or predicting an unconstrained occupancy grid, subdivide its angular
footprint and test whether any complete range/ray enclosure still intersects
the BODY/HEAD corridor. Six binary subdivision levels are fixed; an unresolved
leaf remains POSSIBLE. An OUT result requires every child to be disjoint.
No sampled point or missing return establishes exclusion.

Use the inherited ToF slant range bounds: valid range plus/minus three reported
noise sigmas, and full .02–4 m support for a merged return. Keep pitch/yaw working
uncertainty. For Radar use horizontal range plus/minus .15 m, inherited 12-degree
association tolerance plus yaw uncertainty, and unknown height. These are working
model bounds, not calibrated sensor confidence regions or physical volume bounds.

The three-sensor arm uses ToF interval support plus raw Radar possible support.
The RGB-assisted arm adds every bearing-compatible visual-box shell hypothesis;
it retains a null explanation rather than assuming the proposals are complete.
The sealed MZ116 proposals are used; no new detector is fitted or RGB payload
re-read for this branch. Thus this is a sparse measurement-support experiment,
not a learned echo-association network or a 45-cell occupancy reconstruction.

## Results and the correction contract

| Same 288 frames | TP / FP / FN | Precision | Events | False seconds / segments |
| --- | --- | --- | --- | --- |
| MZ116 | 139 / 116 / 5 | 54.51% | 15/15 | 29.00 / 27 |
| ToF + Radar + IMU measurement support | 140 / 129 / 4 | 52.04% | 15/15 | 32.25 / 26 |
| Same support plus all RGB hypotheses | 140 / 131 / 4 | 51.66% | 15/15 | 32.75 / 26 |
| Correction inheriting MZ116's Radar decisions | 139 / 113 / 5 | 55.16% | 15/15 | 28.25 / 28 |
| Strict correction preserving raw possible Radar support | 139 / 116 / 5 | 54.51% | 15/15 | 29.00 / 27 |
| Inherited correction plus sensor recovery | 140 / 133 / 4 | 51.28% | 15/15 | 33.25 / 24 |

Among 3,954 ToF returns, 1,565 coarse boxes intersect the corridor. Subdivision
proves 62 of those original ray/range supports disjoint, using 13,038 interval
nodes overall. This is a real tightening of the representation under its model;
it does not imply 62 removable alerts. Other returns often still support the frame.

Only 59 incumbent-positive frames qualify for ToF-only correction after retaining
MZ116's current Radar/guard decisions and excluding missing packets, invalid IMU,
merged returns and malformed support. Three complete ToF exclusions occur,
all false frames in suspended HEAD pair2_out, indices 03, 05 and 09. They lose
no baseline TP, but false segments increase from 27 to 28. Even that conditional
gain is not an unqualified reduction in interruptions.

An independent audit exposed a further distinction: `common_radar=False` means
the incumbent declined its Radar branch; it does not mean the new raw Radar
possible-support branch is absent. All three frames still have that ambiguous
raw support. The strict correction therefore retains all three alerts and is
identical to MZ116. This is the accepted contract for accountable correction.
The 139/113/5 arm remains a disclosed inherited-Radar diagnostic only.

Version `measurement-v1` retains the initial conditional interpretation. Version
`measurement-v2` explicitly splits inherited and strict correction after this
audit; no uncertainty bound, depth budget or truth-dependent decision was tuned.
Both versions and their hashes are preserved. The recovery arm's fewer segments
also coexist with longer false duration; no segment count is a real notification
rate or proof of useful spatial disambiguation.

## Decision and evidence

Retain interval subdivision as an explanatory Development geometry component.
It demonstrates excess enclosure volume in the earlier Cartesian bounds.
Do not promote any tested measurement/readout arm: the broad evidence union
increases false alerts, strict correction supplies no alert gain, and the small
conditional reduction inherits an unresolved Radar rejection. Positive ToF
surface support and native obstacle-volume labels remain distinct authorities.

The source prediction seal authenticates the raw packet stream and incumbent
cache. New predictions are serialized before frame truth is read. The six
focused checks cover direct interior witnesses, far exclusion, missing support,
independent Radar retention, truth-metadata independence and the audited
incumbent-versus-raw-Radar distinction. The final CPU backend and source snapshots
are retained under `artifacts.local/work/mz124-all-directions-20260913/measurement-v2/`;
independent read-only checks are in sibling `measurement-audit-v1/`.
That audit independently sampled 3,135,650 rays across all 62 exclusions with
no corridor hits or coarse-enclosure violations. This is a numerical check,
not a continuous-space proof; the interval implementation uses ordinary floating
point, not directed-rounding certified arithmetic. All three nominal Radar
points are outside the corridor; their expanded working intervals remain possible.
Scalar interval work is `TASK_NOT_GPU_SUITABLE`. No owned persistent process or
accelerator allocation is created. Original MZ123 payloads remain unchanged.
