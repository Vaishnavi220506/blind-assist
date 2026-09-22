# Public axis audit: distance signal exists, boundary and lateral ambiguity remain

2026-09-21. Consumed same-simulator Development. **Useful diagnostic evidence:**
none of 633 distance-negative frames has an observed interval wholly inside
[0.3,3] m, while 273/346 positives do. However, the same property is absent
at the first and last positive samples of every one of the 32 positive clips,
and present in 137/173 distance-valid lateral negatives. This is evidence for
separating distance evidence from spatial qualification, not a validated gate.
Keep the original A/demo and these diagnostics; do not change alerts, fit,
tune thresholds, narrow intervals, run conditional hold or activate the test.

## Fixed inputs and observable semantics

[Protocol](AXIS_EVIDENCE_PROTOCOL_20260921.md) and
[runner](audit_axis_evidence.py) cover all 16 transfer layouts, 1152 frames and
73,728 zone positions. Prior SELECTION g00/g01 and EVALUATION g02/g03 remain
separate (8 layouts / 576 frames each). No original FIT or protected-test values
are accessed. Public descriptors were sealed before evaluator geometry, saved
scores or native contributor values were parsed. Parent seals and input hashes
are checked; source private-lineage payload is hashed, never used as a feature.

The public payload contains axial proxy ranges and fixed boxes only. Validity
is finite and 0.1 <= r < 8. Existing uncertainty is retained exactly:
radius=.1+3*(.01+.02*r), interval=[max(.1,r-radius),r+radius]. Full zone footprint
and inherited d/o/joint scores are retained. The conditional angular factor
combines horizontal and vertical overlap; it is not solely lateral or a
probability. Exact contact can be possible with zero overlap measure.

Frame distance states distinguish no valid returns, valid outside-only
intervals, partial boundary intervals only, and at least one wholly-contained
interval. Three fixed evidence-presence flags describe any possible-depth,
any contained-depth, and any contained-depth interval whose full zone support
can intersect the corridor. They do not identify the target or certify that
the actual returned surface occupies the corridor. Absence remains UNKNOWN.

## Complete-cohort axis separation

Actual rendered full target bounds independently reproduce every saved truth
label. Categories prioritize distance invalidity; independent X/Y/Z overlap
flags remain in every record, so simultaneous lateral invalidity is not lost.
Every distance-negative case is beyond 3 m; this cohort has no below-0.3 m or
vertical-only negatives. Do not extend these findings to unobserved categories.

| Role / target category | Frames | Possible-depth evidence | Contained-depth evidence | Contained and corridor-compatible |
| --- | ---: | ---: | ---: | ---: |
| Selection positive | 178 | 178 | 141 | 134 |
| Selection distance negative | 309 | 104 | 0 | 0 |
| Selection depth-valid lateral negative | 89 | 89 | 72 | 59 |
| Evaluation positive | 168 | 168 | 132 | 126 |
| Evaluation distance negative | 324 | 101 | 0 | 0 |
| Evaluation depth-valid lateral negative | 84 | 84 | 65 | 60 |

Across both roles there are 428 OUTSIDE_ONLY, 314 BOUNDARY_ONLY and 410
CONTAINED_PRESENT frames, with zero fully missing frames. Valid-zone counts
are selection 18-64 / evaluation 19-64 (both medians 60). Frozen baseline UNKNOWN
is still true on 516/576 and 524/576 frames: UNKNOWN is not absence of all
measurements. There are 303 native-backed positive frames; contained-depth
evidence is present on 253 (selection132/158, evaluation121/145). Native
attribution is evaluator-only and does not assign the descriptor's witness zone
to the target.

## High-score false positives and suppressed true positives

Use the original fixed high cutoff 7.6612162590026855 only to identify the
previous supplement's added alerts (A_current=false). No score is reselected.
Suppressed positives are original-high Boundary rescues excluded by the frozen
[current-only](CURRENT_ONLY_RESULTS_20260921.md) cutoffs.

| Role / subset | Total | Contained-depth present | No contained-depth |
| --- | ---: | ---: | ---: |
| Selection added high-score distance negatives | 7 | 0 | 7 |
| Evaluation added high-score distance negatives | 8 | 0 | 8 |
| Selection added high-score lateral negatives | 4 | 2 | 2 |
| Evaluation added high-score lateral negatives | 9 | 5 | 4 |
| Selection balanced-suppressed Boundary positives | 46 | 44 | 2 |
| Evaluation balanced-suppressed Boundary positives | 63 | 54 | 9 |
| Selection native-backed subset of those positives | 36 | 35 | 1 |
| Evaluation native-backed subset of those positives | 46 | 43 | 3 |

Contained evidence retains native-backed suppressed positives in 7/8 layouts
on each role. It separates most of this population from all high-score distance
negatives, but is not lossless. For uniform the native retention is35/36 and
40/43; for original35/36 and37/40. The stricter corridor-compatible contained
flag retains balanced33/36 and41/46, with no distance-negative benefit beyond
the simpler flag; it still exists on2/4 and3/9 high-score lateral negatives.
The permissive possible-depth flag retains every suppressed positive but also
all7/7 and8/8 high-score distance negatives. Every prespecified lossless
diagnostic gate is false. These cross-tabs are evidence coverage, not a newly
evaluated alert policy or promoted performance number.

## Boundary costs and the two motivating frames

Of the37/36 positive frames without a contained interval, each role has16 first
positive samples,16 last positive samples, and5/4 interior samples. Thus every
one of the32 positive clips lacks this witness at both endpoints. This is not
merely a negligible collection of isolated interior frames.

The existing interval upper bound is1.06*r+.13. Requiring the entire interval
below3 m implies r <=2.707547 m; this is algebra of the frozen interval, not a
new selected cutoff. Its conservative evidence requirement naturally differs
from the strict target-volume boundary at3 m. No sensor uncertainty was reduced
and no truth labels were relaxed to hide this discrepancy.

The native-backed balanced-suppressed cases excluded by contained evidence are
selection f0326 (interior) and evaluation f0246 (interior), f0255 (last), f0749
(first). Their maximum depth fractions are .994782, .923516, .856014 and .760130.
These values are reported for diagnosis, not used for a threshold retry.

For the motivating negative f0676, actual nearest target Z is3.103923 m, yet the
minimum public return is2.940234 m with interval[2.633820,3.246648]. That minimum
comes from z41, whose full horizontal support is OUTSIDE; its joint score is0.
Other zones provide possible corridor support. A minimum point distance below
3 m therefore neither resolves the uncertainty nor establishes target ownership.

For the positive f0465, actual nearest target Z is2.334700 m; the minimum public
return is2.254175 m with interval[1.988924,2.519425]. Its z26 is horizontally
CROSSING and possible, not definite, corridor support. This is a genuine public
distance distinction between the motivating frames, while lateral attribution
remains unresolved. Neither minimum is asserted to be the target's exact range.

## Decision and limits

Retain COMPONENT_OR_CHALLENGER / COMPONENT for this observed axis separation
and explicit boundary-cost diagnosis. There is no qualified lossless distance
gate and no new candidate. The prior current-only and last-layer negatives remain
unchanged. This audit supports neither “there is no distance information” nor
“a ToF distance check alone fixes the spatial model.” Overlap of these summaries
does not prove an information ceiling for the full64 zones, and the results do
not establish whether the encoder discarded a recoverable feature.

Four focused validity/interval/missingness tests pass. Frozen A geometric scores,
validity, definite counts, UNKNOWN and current flags reproduce exactly. Independent
[saved-record verification](verify_axis_evidence.py) passes:76 hash entries,
7 seals, all1152x64 descriptors, actual geometry labels, all role/group/subset
counts and IDs, unchanged alerts/scores, and the complete boundary-cost detail.
The inherited analytic support integrator is reused and disclosed; remaining
descriptor geometry, counts and diagnostic gates are independently reconstructed.
Receipts are `independent-audit.json` and `independent-audit.log`.
The work used CPU for small saved-array
reductions (TASK_NOT_GPU_SUITABLE); there was no GPU/model load or paid allocation.
All scores, alerts, source files, reserved-test status and demonstration remain
unchanged. Evidence resides under `artifacts.local/work/ba-axis-evidence-20260921/`,
including public/analysis seals, per-zone records, per-layout subsets, known-frame
details and `boundary-cost-detail.json`. Supported registration fails at the
existing `experiments/index.jsonl:303 input_fingerprint` mismatch; inheritance
returns `unknown terminal id: ba-axis-evidence-20260921`. Both receipts are kept;
global metadata is pending, with no bypass. Stop this audit after scoped delivery.
