# Active observation: a small closed-world identification gain, no recall gain

2026-09-21. One [frozen parametric pilot](ACTIVE_VIEW_PROTOCOL_20260921.md)
completed. Terminal `SCOPED_IN_PRIOR_COMPONENT`: adaptive view choice resolves
114/273 initially ambiguous questions versus104/273 for fixed translation and
0/273 for repetition. The ten net additional answers are **eleven correct
negative identifications minus one positive identification**, not ten rescued
obstacles. All32 out-of-prior cases remain UNKNOWN. Retain a narrow observation-
selection mechanism component; do not promote an alert policy or launch a retry.

## Same question and observation budget

The question is fixed in the original camera frame: X[-0.30,0.30],
Z[0.30,3.00], with a stipulated always-overlapping vertical profile. Translation
does not recenter the corridor. The source is a two-dimensional analytic slice
of one rectangular obstacle and a far wall. Eight sparse center rays provide
quantized **radial** ranges; Cartesian extent supplies query truth. This is not
an 8x8 hardware model, RGB experiment, UE capture or Calibration comparison.

Every arm has two observations. Fixed and adaptive move0.12m one way; passive
does not move. All four candidate adaptive actions have the same displacement
magnitude. Costs exclude actual motion execution, tracking and settling. Source,
initial observations, generic hypothesis forecasts and all action choices are
sealed before true-scene next observations; final public decisions are sealed
before evaluator truth. Selection uses only initial ranges and the explicit
prior bank, not scene identity or a measured future view.

## Actual decisions

The primary opportunity subset has273 mixed-label initial observations:
113 true intersections and160 negatives. The complete in-prior bank contains621
cases,344 positive and277 negative;348 were already decidable initially.

| Initially ambiguous273 | Correct decisive | Wrong | UNKNOWN | Correct positive | Correct negative | Positive UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| Repeat,0m | 0 | 0 | 273 | 0 | 0 | 113 |
| Fixed +X,0.12m | 104 | 0 | 169 | 47 | 57 | 66 |
| Adaptive,0.12m | 114 | 0 | 159 | 46 | 68 | 67 |

Adaptive versus fixed changes eighteen cases: eleven UNKNOWN become correct
negative identifications, three UNKNOWN become positive, and four previously
positive answers become UNKNOWN. There are no wrong decisive answers in this
closed-world source. Unknown positives remain misses of identification; they
are not counted as correct abstentions or silently removed from the denominator.

| Whole in-prior621 | Correct decisive | Wrong | UNKNOWN | Positive answers /344 | Negative answers /277 |
|---|---:|---:|---:|---:|---:|
| Repeat | 348 | 0 | 273 | 231 | 117 |
| Fixed | 452 | 0 | 169 | 278 | 174 |
| Adaptive | 462 | 0 | 159 | 277 | 185 |

The frozen identification gate passes, but obstacle-positive coverage does not
improve over fixed scanning. A negative answer means all remaining finite-bank
hypotheses are nonintersecting, **not certified free space**. In-prior zero wrong
is largely structural: the noiseless true scene is included in the same exact
forward-model bank. It is not generalization or safety evidence.

## Information and observed-return accounting

For the273 ambiguous cases, summed candidate counts are34,917 initially,
12,899 after fixed, and12,851 after adaptive. These are sums over correlated
hypotheses/cases, not independent sample sizes or calibrated entropy estimates.
Adaptive removes only48 more candidates than fixed on this subset.

Both moving arms first observe target returns in the **same74 previously unseen
in-prior cases**; neither beats the other on target-ever-seen acquisition.
Across the primary subset, target-ever-seen cases are88/273 for repeat and
162/273 for either moving arm. Next-view target-hit ray incidences are114,
173 and176 respectively. Newly target-hit ray indices relative to the first
view are0,115 and106; the same angular index across translated views is not
the same physical ray. These are evaluator-only attribution counts, not public
object associations, geometry points or rescued events.

On273 ambiguous cases adaptive chooses +X220 times, -X35, +Z18, -Z0; the listed
tie-break order favors +X. The whole-bank counts are486/103/24/8. The policy
often behaves like the fixed comparator; no action was selected using outcomes.

## Mismatch is a severe limitation

| Fixed out-of-prior stratum | Cases | Initial no-bank-match | Final correct/wrong/UNKNOWN, all arms | Final no-match, repeat/fixed/adaptive |
|---|---:|---:|---|---|
| Off-grid target geometry | 24 | 14 | 0/0/24 | 14/19/19 |
| Wall Z4.20 to4.80m | 8 | 8 | 0/0/8 | 8/8/8 |

The ten initially ambiguous off-grid cases still provide zero decisive answers.
Fixed/adaptive obtain five newly seen target cases in that stratum and two in
the changed-wall stratum, yet cannot interpret them through the prior. Missing
bank support stays UNKNOWN. Zero wrong with zero correct in32 cases is an
availability failure, not robust accuracy. No mismatch tolerance or bank
expansion was fitted after observing this result.

## Verification, evidence and reproduction

Six focused tests pass: unavailable future/truth APIs during selection,
deterministic public-only selection, full-extent/fixed-query geometry, radial
versus optical depth, empty/mixed-bank UNKNOWN, and missing/tampered choice seals.
An independent saved-output calculation (same agent, different equations) passes:
41 hash links;3,105 bank forecasts;653 initial observations;653 action choices;
1,959 selected observations;1,959 public posteriors;653 truth labels;18 aggregate
groups. It reconstructs rays by face intersections instead of the implementation's
slab clipping and reproduces the eighteen paired changes. Stage order verifies
all choices precede actual second measurements and public predictions precede
truth evaluation. A second agent's read-only
[cross-review](../../../../artifacts.local/work/ba-shape-hypotheses-20260921/run-v1/active-view-independent-review.json)
also passes: five upstream seal types,653 choice-cost reconstructions,1,959 saved
posteriors/decisions,653 fixed-query truths and complete cohort metrics. It found
no material issue and independently retained the negative-identification gain
and32-case mismatch abstention limits. That review did not replicate ray rendering
or run another scored experiment; the earlier independent-equation ray audit was
performed by this experiment's own agent.

Evidence root: `artifacts.local/work/ba-active-view-20260921/`. `run-v1/` contains
source, bank, forecasts, initial/selected observations, choices, predictions,
evaluation, summary, stage order, local disposition and hash seals.
`independent-saved-audit.json` records the separate recount. `test-receipts/`
retains the intentionally tampered tiny choice file from the rejection test.
`frozen-code/` preserves exact run-time code bytes for later reproduction.
Supported registration was attempted and reports the pre-existing ledger303
`input_fingerprint` mismatch; supported inheritance reports unknown terminal
`ba-active-view-20260921`. Argument/exit-code JSON and original output receipts
remain beside the run. The sealed local disposition's pending-registration
field records the pre-attempt state; these later receipts record the actual
attempts. Global metadata admission remains incomplete, with no ledger bypass.

```powershell
python -B -m unittest discover -s scripts/research/nearfield_opportunities -p test_active_view.py -v
python -B scripts/research/nearfield_opportunities/active_view.py --output artifacts.local/work/ba-active-view-20260921/run-v1 --protocol research/active/dtr-r0/nearfield/ACTIVE_VIEW_PROTOCOL_20260921.md
```

The main command was executed exactly once. It refuses an existing output path;
a later authorized reproduction must use a fresh path and is not fresh evidence.
Python entry is `E:/codex-tools/bin/python.cmd`. Scalar analytic work ran on CPU
(`TASK_NOT_GPU_SUITABLE`); the source-to-evaluator stage log spans about0.30s,
excluding final serialization and process startup. This is host bookkeeping,
not acquisition, movement or device latency. No persistent process, GPU worker,
paid allocation or temporary scene remains. Durable evidence is retained.

Known exact poses, a finite single-obstacle prior, noiseless quantized ray model,
equal prior enumeration and two static measurements are the explanatory limits.
This result supports testing observation choice as its own mechanism; it does
not reopen MZ155/MZ182 or justify a general temporal/active-sensing success claim.
The one pilot ends here, with no tuning, capture, new model or automatic successor.
