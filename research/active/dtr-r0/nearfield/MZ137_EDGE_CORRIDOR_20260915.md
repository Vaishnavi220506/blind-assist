# MZ137: one coarse/fine edge to corridor contrast

EXPLORE, 2026-09-15. Use the **already consumed MZ136 dev48**, all four
families and all missing-output frames. No new capture, TRAIN fitting, original
test scoring, threshold search or edge tuning. Stop after this fixed comparison.
MZ136's preceding TRAIN-only edge result **has not demonstrated alert benefit**;
it did not evaluate alert improvement or establish the absence of improvement.

## Question and fixed intervention (before outcomes)

Does the retained MZ136 refinement change selected spatial supports, correct
corridor decisions and reduce final false alerts relative to identical coarse
readout? Compare frozen MZ129, shared readout with coarse seeds, and the exact
same readout with fine horizontal edges (coarse fallback when unavailable).

Select the first frozen MZ136 proposal by observed contrast support. Identify
its unique public SIM_VALID ToF cohort through the proposal's zone list; reject
duplicate zones/ambiguous cohorts. Convert each measured slant range along its
zone-center ray through the public camera pitch/integrated IMU yaw, and use
the median body-forward depth. Assume the silhouette and cohort share a plane
perpendicular to body forward. Both arms use that same depth and the same
rectified coarse vertical extent; only horizontal edges differ. No evaluator
identity, native geometry or label enters prediction. This is an uncalibrated
surface hypothesis, not proof that returns and silhouette share a surface.

Replace incumbent ToF alert supports only for selected single-target VALID
slots whose **entire zone footprint** lies inside RGB and intersects the coarse
seed. Unmatched, MERGED and out-of-view evidence retains its old alert vote.
Keep the frozen Radar branch and guards. Eligible old coarse-certainty votes
are replaced with the plane readout too; the complete MZ129 alarm is not ORed
back in. Thus ToF false alarms can disappear, while residual independent
evidence can still sustain an alert. Trace that obstruction explicitly.

No valid association: exact MZ129 fallback. No fine edge: coarse fallback.
Raw packets are unchanged. No-alert remains UNKNOWN, never certified clearance.

## Acceptance and diagnosis

Retain the original Development criterion: >=20% FP reduction (13 to <=10),
recall within 2 percentage points (24 positives therefore no FN), all five
incumbent events and per-event delay no more than .25 seconds. Improvement of
fine over coarse identifies the edge contribution; baseline-to-coarse changes
identify shared association/readout effects. Passing this consumed panel would
justify only a Development challenger, not independent validation/promotion.

Report full 48-frame confusion, precision/recall, families, shallow-boundary
versus other families, event times, false segments/duration, pair both-correct
and binary ordering, availability/fallbacks, changed edges/support bits/plane
decisions/final flags, removed/new errors and unchanged independent evidence.
Evaluator-only signed native corridor overlap stratifies <1cm shallow overlap
or gap versus >=1cm; this is descriptive and does not change labels or gates.
Audit native contributor directions against coarse/fine angular support and
native corridor contributor retention; a point plane is not a volume enclosure.

Seal method/source/input identities and predictions before parsing selected
evaluator rows. Verify exact prediction replay and input immutability. CPU
OpenCV/NumPy has no equivalent implemented GPU backend; record actual runtime.
Source and payloads stay under artifacts.local. Registration/inheritance use
supported commands; the pre-existing ledger303 mismatch remains separate.

## Result: edge changes reach one plane decision, no final alert changes

All 48 rows remain in every denominator. The frozen source/method ran once;
the second execution was an exact replay check, not a candidate retry.

| Consumed dev48 | MZ129 | Coarse plane | Fine plane + fallback |
| --- | ---: | ---: | ---: |
| TP / FP / FN / TN | 24 / 13 / 0 / 11 | 24 / 13 / 0 / 11 | 24 / 13 / 0 / 11 |
| Precision / recall | 64.86% / 100% | 64.86% / 100% | 64.86% / 100% |
| No-alert / UNKNOWN | 11 | 11 | 11 |
| Events detected | 5 / 5 | 5 / 5 | 5 / 5 |
| Maximum first-alert delay | 0 s | 0 s | 0 s |
| False segments / sampled-bin duration | 4 / 3.25 s | 4 / 3.25 s | 4 / 3.25 s |
| Pair both-correct | 11 / 24 | 11 / 24 | 11 / 24 |
| Original joint Development target | Not a candidate | Failed | Failed |

Every first-alert timestamp is unchanged. Removed baseline FP, new FP and lost
baseline TP are each zero. Strict ordering of **binary decisions** is 11/24;
this is not comparable to the prior learned continuous-logit ranking.

All three arms have the same family outcomes: HEAD 6/0/0, BODY 6/1/0,
rod/far-wall 6/6/0, shallow-boundary 6/6/0 (TP/FP/FN). Shallow-family pressure
frames are 6/6/0 over 12; other families 18/7/0 over 36. The prespecified native
<1 cm gap/overlap stratum has four frames, 2/2/0; the remaining 44 have 22/11/0.
Thus this panel's remaining false alarms cannot all be attributed to millimetre
boundary cases. These are descriptive strata, not revised labels or gates.

### Coverage and causal trace

| Family (12 frames each) | Coarse available | Fine available | Shared association usable | Actual coarse/fine geometry change |
| --- | ---: | ---: | ---: | ---: |
| HEAD | 9 | 4 | 5 | 3 |
| BODY | 9 | 0 | 9 | 0 |
| Rod/far-wall | 3 | 0 | 3 | 0 |
| Shallow boundary | 10 | 10 | 10 | 10 |
| Total | 31 / 48 | 14 / 48 | 27 / 48 | 13 / 48 |

There are 17 frames without a coarse proposal and another four without an
eligible association: 21 exact incumbent fallbacks. Fourteen of the 27 usable
associations use coarse fallback because refinement is unavailable. One of the
14 fine outputs has no usable association. No missing result disappears.

**13 edge changes -> 13 spatial-support changes -> 1 plane decision correction
-> 0 final alert changes.** Four individual ToF support bits change in that
one frame. The coarse-to-fine gain ends at the final evidence aggregation.

The exact case is `mz136_shallow_boundary_stress_scene4_enter_02`:

- Native target is outside the corridor by 6.294 mm.
- Shared public body-forward depth is 3.218471 m, cohort spread 101.882 mm.
- Coarse inferred overlap is +2.853 mm; fine is -2.876 mm. Thus fine changes
  this selected plane from in-corridor to out-of-corridor, correctly.
- Selected zones 26/34/42/50 lose their possible-support votes. Zone 58 has
  independent support outside full RGB coverage; it keeps score 1 and the
  alarm remains. Radar and its guards are false in this frame.
- Inner-edge absolute error actually worsens from 0.780 to 1.594 pixels here.
  The locally correct crossing therefore does not prove better surface depth
  or edge accuracy; signed errors in different coordinates can compensate.

Do not delete zone 58 merely to turn this diagnostic into a positive result.
That would require a justified model of its unobserved extent. Also, preserving
this branch is not a full-incumbent OR: the focused logic test demonstrates
that replaced ToF false alarms can disappear when no independent vote remains.

### Native support loss is material even with unchanged frame recall

There are 4,187 returned native ToF contributor records, including 1,046 actual
corridor points. Both arms replace slots containing 503 contributor records.
The evaluator projects actual points into the same body-aligned angular support;
it does not pretend a zero-thickness point plane encloses a solid object.

| Native contributor audit | Coarse | Fine |
| --- | ---: | ---: |
| Replaced contributors outside proposed angular extent | 378 | 362 |
| Actual corridor contributors excluded angularly | 117 | 105 |
| Previously enclosed corridor contributors now excluded angularly | 117 | 105 |
| Actual corridor contributors whose old possible alert support is removed | 90 | 90 |

Fine retains 12 more actual-corridor angular samples than coarse, but still
excludes 105 that the incumbent support enclosed. The unchanged final recall
is sustained by residual support, not evidence that the new plane is reliable.
Angular exclusion and removal of a possible alert vote are different counters;
neither equals a lost frame/event. Unchanged raw packets do not erase this loss.
The 90 alert-support losses are 30 returns across all six BODY-positive frames;
they are not 90 missed frames. No replaced record loses a coarse-certainty-only
vote in this run, so the possible-support audit misses no such loss here.
Radar predictions and their support model are reused unchanged, not re-associated
to a fine edge. No new Radar-geometry retention claim is made.

## Decision, verification and evidence

Keep MZ129 as the system baseline and MZ136 as a conditional geometry-input
component. Reject **this fixed cohort-plane replacement rule** as an alert
improvement on consumed dev48. The experiment now establishes no final alert
gain for this exact contrast; it does not establish that every future edge
integration is ineffective. No pure edge tuning or alert-head fitting follows.

The gap is both limited coverage and a surface/extent assumption: only one
eligible change corrects a plane decision, independent support sustains its
alert, and the replacement also loses real native corridor support. A subsequent
proposal would need to explain that support, not merely lower mean pixel error.
This run stops without new capture, original-test access, shifted-test scoring
or another operating-point search.

Evidence root: `artifacts.local/work/mz137-edge-corridor-20260915/contrast-v1/`.
`protocol-before-outcomes.md`, `freeze.json` and `source-snapshot/` preserve
the fixed method. `predictions.json` and `prediction-seal.json` precede selected
evaluator parsing. `cases.json` retains every frame's signed error/overlap and
chain; `native-contributors.json` retains per-return/per-ray evidence; `summary.json`
contains all event times and strata. `completion.json` verifies 48/48 exact
prediction replay and input/source immutability. Prediction SHA256:
`329571212c620a5bf3dc86ac1bebf574d607ad20ca3e8b8a677e0d829ee683ba`.

All 36 focused `test_mz13[67]*.py` tests pass (31 inherited, five new). New checks
cover a real false-alarm removal opportunity, fine-induced boundary crossing,
same-zone residual/Radar preservation, missing fallback/input immutability,
replacement of old certainty, outside-RGB retention and ambiguity rejection.
CPU OpenCV 4.10.0 / NumPy 2.4.4 executed the 48-frame prediction loop in 0.2002 s;
this excludes cached incumbent processing, file IO and scoring and is not a
full-system latency benchmark. The backend record is `GPU_BACKEND_UNAVAILABLE`.
No persistent process, worker or paid allocation remains.

Intended terminal: `MZ137_EDGE_CORRIDOR_NO_JOINT_GAIN`; intended role:
`NEGATIVE_CONTROL`, scoped to this fixed public-cohort plane and support-replacement
rule on consumed dev48. Supported registration and inheritance are attempted
separately; their status is recorded below, without editing the shared ledger.

Registration failed at the existing `experiments/index.jsonl:303` fingerprint
mismatch; the inheritance command then reported the unknown terminal. Both
outputs are retained in the parent artifact directory. Registration and structured
inheritance remain **pending metadata**, not completed registration. The ledger
was not repaired, appended manually or bypassed. This is separate from the
completed technical comparison and its scoped negative result.
