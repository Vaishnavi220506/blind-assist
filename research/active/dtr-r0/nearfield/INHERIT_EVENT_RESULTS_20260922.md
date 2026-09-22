# Original-support event inheritance: exact recipe negative control

2026-09-22. Completed one authorized EXPLORE candidate on all 1,728 consumed
query-occupancy frames. **Terminal recommendation: NEGATIVE_CONTROL for this
exact recipe.** No threshold, interval, association or expiry variant follows
this run. The result does not reject temporal information or event inheritance
in general.

## Decision and mechanism

The frozen [protocol](INHERIT_EVENT_PROTOCOL_20260922.md) separated alert birth
from continuation eligibility. Birth stayed the exact A current decision, with
threshold `0.4071309640537889`, its definite-support bypass and original UNKNOWN.
The candidate retained every birth and allowed only the immediately following
sample, within 0.2 s, to inherit from that birth. Inheritance required an observed
return in an original decisive zone, positive overlap with that original range
support interval, continued query possibility, and a measured range centre in
the fixed `[0.3, 3.0] m` slab. A held alert never renewed the seed. There was no
score-cutoff selection, recursive keep score, learned model or truth identity.

This differs from the consumed 2026-09-21 conditional-hold recipe, which used a
lower original-score cutoff for recursive continuation. Here the hypothesis was
that observable agreement with original support could qualify finite inheritance.
On the primary 576 frames it qualified no continuation. It removed all 20 FP
added by old hold, while losing all 32 TP added by that hold.

## Primary complete-clip result

The original evaluation role contains 576 frames, 48 complete 12-frame clips,
16 base layouts, 256 positive and 320 negative frame labels, and 32 positive
events. This consumed role is Development evidence in this experiment.

| Arm | TP / FP / FN | Recall | Precision | FP / 320 negative labels | Events detected | False segments / sampled seconds |
| --- | --- | --- | --- | --- | --- | --- |
| A current | 193 / 6 / 63 | 75.39% | 96.98% | 1.88% | 32 / 32 | 6 / 1.2 s |
| Original nonrecursive A hold | 225 / 26 / 31 | 87.89% | 89.64% | 8.13% | 32 / 32 | 20 / 5.2 s |
| Support inheritance | 193 / 6 / 63 | 75.39% | 96.98% | 1.88% | 32 / 32 | 6 / 1.2 s |

The candidate's evaluation decisions are identical to A current. Its retained
fraction of hold's added TP is `0 / 32`; its removed fraction of hold's added FP
is `20 / 20`. It adds zero TP and reaches zero gain layouts, failing the frozen
75% retention, at-least-8-TP and at-least-4-layout criteria. It preserves all
A current true decisions, events and onset times. These local usefulness gates
do not confer safety authority.

UNKNOWN remains identical for all arms: 487 / 576 frames, comprising 167 positive
and all 320 negative frame labels. Alerts may coexist with UNKNOWN. Withheld
unknown-negative decisions are abstentions, not certified true negatives
(`TN = 0` under the frozen metric semantics). IoU does not apply to this
alert-only decoder.

| Timing diagnostic, primary role | A current / candidate | A hold |
| --- | --- | --- |
| Initial silent positive samples | 30 | 30 |
| Internal silent positive samples | 15 | 0 |
| Terminal silent positive samples | 18 | 1 |
| Internal interruption episodes | 15 | 0 |
| Maximum event onset delay | 0.4 s | 0.4 s |
| Pre-entry false samples | 0 | 0 |
| Post-exit false samples | 1 | 16 |
| Post-exit sampled false duration | 0.2 s | 3.2 s |

Durations are sample counts multiplied by 0.2 s, not measured physical warning
latency or continuous walking exposure. Full per-event coverage, first detection,
internal and terminal gaps, pre-entry and exit-tail data are saved; neither event
boundaries nor their object identities were used by inference.

## Where continuation disappeared

There are 52 primary frames alerted only by old hold: 32 positive and 20 negative.
Each frame may test more than one original decisive zone; the following are
support-pair checks, not mutually exclusive frame counts.

| Original-support check outcome | Hold-only positive frames: 76 checks | Hold-only negative frames: 39 checks |
| --- | --- | --- |
| Current zone return missing or invalid | 5 | 1 |
| Current support no longer query-possible | 39 | 24 |
| Original/current support intervals disjoint | 32 | 0 |
| Measured centre outside query slab | 0 | 14 |
| Compatible current support | 0 | 0 |

Thus removing the centre-in-slab qualifier alone would not recover the lost
32 TP in this run. Their available original-zone returns already failed the
query-possibility or interval agreement checks, or were absent. This is evidence
against this particular same-zone support test at this sampling rate and sensor
representation. It does not show that another independently observed association
is unavailable. No alternative association or weakened qualifier was tested.

Across all 1,728 frames, only one inheritance occurred. It was a train-role FP:
`query_occupancy_body_suspended_solid_g10_outside_09`, inherited from `_08` through
zone 53. Same-zone compatibility cannot identify an object; a second object with
indistinguishable returns remains unresolved, as the focused control demonstrates.

## Full-cohort and subgroup disclosure

Train/dev roles are descriptive diagnostics, not used to choose the mechanism.
All complete clips were processed with one fixed candidate and one fixed A.

| Role | Frames | A current TP / FP / FN | A hold TP / FP / FN | Candidate TP / FP / FN | Original UNKNOWN |
| --- | --- | --- | --- | --- | --- |
| Train | 864 | 288 / 17 / 96 | 332 / 54 / 52 | 288 / 18 / 96 | 725 |
| Dev | 288 | 98 / 3 / 30 | 114 / 13 / 14 | 98 / 3 / 30 | 246 |
| Evaluation | 576 | 193 / 6 / 63 | 225 / 26 / 31 | 193 / 6 / 63 | 487 |

All 48 / 16 / 32 positive events were detected in train / dev / evaluation for
every arm. Primary strata below give A current, which is also the candidate,
against old hold; individual layout strata and every paired frame gain/loss ID
are present in `metrics.json` and `frame-results.json`.

| Primary stratum | A current / candidate TP / FP / FN | A hold TP / FP / FN |
| --- | --- | --- |
| body_protruding_plane | 49 / 4 / 15 | 56 / 11 / 8 |
| body_suspended_solid | 50 / 1 / 14 | 60 / 6 / 4 |
| head_hanging_plane | 49 / 1 / 15 | 56 / 6 / 8 |
| head_horizontal | 45 / 0 / 19 | 53 / 3 / 11 |
| BODY | 99 / 5 / 29 | 116 / 17 / 12 |
| HEAD | 94 / 1 / 34 | 109 / 9 / 19 |
| BOUNDARY | 71 / 0 / 57 | 99 / 0 / 29 |
| INSIDE | 122 / 1 / 6 | 126 / 16 / 2 |
| OUTSIDE | 0 / 5 / 0 | 0 / 10 / 0 |

## Execution, audit and preserved evidence

Inputs were the exact admitted prepared ToF and identity files, materialization
metadata and three evaluator label NPZs from
`artifacts.local/evidence/ba-query-occupancy-20260922-prepared`. Observation and
source hashes were bound before execution. All predictions were sealed before
label arrays were opened; stored baseline labels were used only for post-seal
parity checks. No native contribution, RGB, protected data, model weight or real
detector track was read. No UE policy edit was required.

Both governed entry points succeeded with UE preflight PASS, native input count
0, asset-fabric verification PASS and recorded lineage:

```powershell
pwsh -NoProfile -File tools/ba.ps1 run research-ue -RunSpec artifacts.local/evidence/ba-inherit-event-20260922/run-spec.json
pwsh -NoProfile -File tools/ba.ps1 run research-ue -RunSpec artifacts.local/evidence/ba-inherit-event-20260922/audit-run-spec.json
```

Run receipts:

- `artifacts.local/evidence/resource-fabric/runs/ue-inherit-event/inherit-event-20260922-run-v1.json`
- `artifacts.local/evidence/resource-fabric/runs/ue-inherit-event/inherit-event-20260922-audit-v1.json`
- Specs, console logs and focused test output:
  `artifacts.local/evidence/ba-inherit-event-20260922/`.
- Frozen input/source hashes, protocol copy, backend receipt, per-frame supports,
  predictions, prediction seal, full paired frame/event metrics, evaluation seal
  and terminal summary:
  `artifacts.local/evidence/ba-inherit-event-20260922-run/`.
- Independent audit receipt:
  `artifacts.local/evidence/ba-inherit-event-20260922-audit/result.json`.

Key saved hashes:

- `metrics.json` SHA-256:
  `0621c4f3a39952fbc0cd10815476844b50d52549ba3111d64cd210a5284268b4`.
- `prediction-seal.json` SHA-256:
  `f132d14c43a026c6700b0d8fcb8ebbeb4667d4ee498871513c094863a872983e`.

Eight focused tests passed: exact-threshold birth, nonrecursive inheritance,
miss/reappearance, different-zone second target, indistinguishable same-zone
limitation, interval/nominal exit, clock/clip expiry and nonmonotonic time rejection.
The independent auditor imports neither candidate core nor runner. It reconstructs
the inheritance equations from public supports, verifies all 1,728 decisions and
UNKNOWN values, checks hashes and baseline parity, and recalculates frame counts,
paired IDs, event timing, false segments and all terminal gates. It reuses the
unchanged A scoring function; this is not an independent validation of A physics.

Scalar processing ran on CPU with `TASK_NOT_GPU_SUITABLE` recorded. Mean per-frame
score/decoder time was 2.340 ms (p50 2.514 ms, p95 2.960 ms); script work took
5.427 s, excluding governed-entry overhead. This is a host implementation timing,
not an Android/device latency claim. No capture, training, new solver run or
continuing worker was needed. No failure or mechanical retry occurred.

## Evidence boundary and retained engineering value

This is same-generator controlled-object simulation Development: 48 layout groups,
144 clips and 1,728 frames, with the consumed 576-frame role explicitly primary.
It provides no natural-scene, physical ToF, product, Android, user-benefit or safety
evidence. No fresh confirmation set was consumed.

The event seed lifecycle, original-support trace, expiry semantics and frozen
baseline comparisons remain useful auditable components. The candidate's effect
does not meet its declared usefulness condition. Historical seven-frame bbox
three-state motion is **NOT_EVALUABLE** in this run because no real detector tracks
are present; evaluator object IDs must not replace those observations. The
mechanical controls establish semantics only, not algorithm effect. No successor
is automatically started, and no shared route state is changed by this report.
