# Active-view opportunity audit: positive-priority changes the tradeoff

2026-09-21, authorized read-only analysis of the prior sealed public bank,
forecasts, choices and saved decisions. No new sensor observation, renderer,
policy execution or scientific source run occurred in this audit. The following
counterfactuals are **posthoc finite-prior bounds**, not measured new-policy
performance or independent confirmation.

There is a concrete candidate: minimize positive hypotheses left unresolved,
then total unresolved hypotheses, with fixed action-order ties. On the saved
prior it can identify56 positive/62 negative cases versus fixed47/57 and old
adaptive46/68. This is a favorable change against fixed scanning, but a positive/
negative tradeoff against the old adaptive policy. The old policy is itself on
the Pareto frontier; its lost positive cannot be restored while retaining all
68 negative identifications under the same two-view, four-action contract.

## Complete initial ambiguity classes

The273 initially ambiguous in-prior cases fall in13 identical-observation classes,
with113 positive and160 negative hypotheses. An admissible policy must choose
the same action for every member of an initial class. All actions translate0.12m
and obtain one additional observation; the original reference query stays fixed.

Each action cell below is `positive decisive / negative decisive / UNKNOWN`.
The candidate column is only an algebraic proposed choice over saved forecasts.

| Class | Initial positive/negative | +X fixed | -X | +Z | -Z | Old action | Proposed positive-priority |
|---|---:|---|---|---|---|---|---|
| 000 |5/4|0/0/9|2/0/7|0/3/6|3/1/5|+Z|-Z|
| 001 |4/3|2/0/5|0/1/6|0/1/6|2/0/5|-X|+X|
| 002 |2/2|1/1/2|0/0/4|0/0/4|0/0/4|+X|+X|
| 003 |1/7|1/7/0|0/6/2|0/1/7|1/7/0|+X|+X|
| 004 |4/2|4/2/0|3/0/3|0/1/5|1/0/5|+X|+X|
| 005 |7/3|3/0/7|3/0/7|0/1/9|0/0/10|-X|+X|
| 006 |4/2|3/0/3|4/2/0|0/1/5|1/0/5|-X|-X|
| 007 |7/3|3/0/7|3/0/7|0/1/9|0/0/10|+X|+X|
| 008 |4/3|0/1/6|2/0/5|0/1/6|2/0/5|+X|-X|
| 009 |2/2|0/0/4|1/1/2|0/0/4|0/0/4|-X|-X|
| 010 |1/7|0/6/2|1/7/0|0/1/7|1/7/0|-X|-X|
| 011 |5/4|2/0/7|0/0/9|0/3/6|3/1/5|+Z|-Z|
| 012 |67/118|28/40/117|28/40/117|6/6/173|10/16/159|+X|+X|

Class identities are lexicographic initial range-bin signatures; complete public
signatures and bank IDs are retained in the artifact. The maximum positive count
and minimum UNKNOWN are jointly attainable within each class, hence jointly
across all13:56 positive,62 negative,155 UNKNOWN. These are conditional on the
noiseless finite prior and are not upper bounds for all available sensor data.

| Saved-forecast accounting | Positive | Negative | UNKNOWN | Decisive total |
|---|---:|---:|---:|---:|
| Actual fixed, reproduced |47|57|169|104|
| Actual old adaptive, reproduced |46|68|159|114|
| Posthoc class-consistent maximum-positive/minimum-UNKNOWN bound |56|62|155|118|
| Truth-conditioned per-case oracle, **not a policy** |76|106|91|182|

The last row chooses different actions for indistinguishable cases using their
hidden identity; it is unimplementable from initial observations. The56/62 bound
instead shares one action per initial class, but remains a posthoc prior audit.
It does not prove an implemented policy produces those observations correctly.

## Where the tradeoff comes from

Classes001 and011 each lose two positive identifications under old adaptive
versus fixed. Gains of one each in006,009,010 leave the reported net loss of one.
In001, moving -X resolves one negative but leaves all four positives unknown;
+X resolves two positives but leaves all three negatives unknown. In011, +Z
resolves three negatives and no positives, whereas -Z resolves three positives
and one negative. No action dominates old adaptive in both positive and negative
identifications in any of the13 classes.

The exact aggregate Pareto frontier in `(positive,negative)` counts is:
`(40,70),(43,69),(46,68),(48,67),(50,66),(51,65),(53,64),(54,63),(56,62)`.
Thus more than46 positive identifications necessarily sacrifices some of the68
negative identifications. This is not a zero-cost repair of old adaptive.

Against fixed, the proposed positive-first choice retains all47 positive IDs,
adds9, loses one negative identification and gains six others: aggregate+9/+5.
So aggregate dominance over fixed is not per-case retention of all negative
answers. Restricting choices to no fewer positive and total decisive counts than
fixed within each initial class yields the same candidate on this prior.

## A different objective, not a renamed pair split

For action `a`, partition current candidate hypotheses by predicted next
observation into groups `g`, with positive/negative counts `p_g,n_g`.
The old objective minimizes `sum(p_g*n_g)`, then squared group sizes.
Positive unresolved mass is instead `sum(p_g for groups with p_g>0 and n_g>0)`;
secondary cost is total membership of those mixed groups. Under uniform bank
weights these are proportional to expected unresolved-positive and unresolved-
total probability. Normalization within one initial class does not change order.
The weights are an explicit enumeration prior, not calibrated scene frequencies.

These objectives differ: one heavily mixed large opposite-class group can
dominate pair counts even if it contains few positive cases. Class001 illustrates
the actual reversal: pair cost4 chooses -X with zero positive answers; cost6
chooses +X with two positive answers. Changing only the pair-score threshold or
a monotone calibration cannot reproduce the new action order.

The simple positive-first/total-UNKNOWN/fixed-order criterion differs from old
adaptive in five classes (four task-metric changes and one tie change). It reaches
the56/62 bound in these stored forecasts. **This is an executable candidate
specification, not evidence of a completed new policy execution.**

Implementation detail of this saved audit: `public_positive_objective` and its
guarded form originally use `(positive UNKNOWN, total UNKNOWN, opposite-pair
cost, action index)`, retaining pair cost as a tertiary tie-break. That is not
the proposed selector's three-element rule. The subsequent
`independent-set-check.json` explicitly recomputes choices with only
`(positive UNKNOWN, total UNKNOWN, action index)` and verifies identical chosen
actions in all13 classes on this prior. Consequently the table and56/62/155
totals apply to the simpler candidate here. This equivalence is dataset-specific;
the old audit code/output were preserved, not silently rewritten or rescored.

## What is worth testing and what remains blocked

This supports one small, explicitly consumed-Development implementation check
of the positive-priority selector, not another action or weight sweep. Freeze
one rule and seal all choices before producing their actual second observations;
report positive answers, wrong answers, abstention, ID retention and the negative
identification cost against both existing policies. Do not select actions by
actual test truth or copy the oracle table as a hidden class-label lookup.

The maximum gain is modest: even the class-consistent upper bound leaves57/113
positive and98/160 negative ambiguous cases unresolved. The all-background
initial class012 alone retains117/155 total UNKNOWN and39/57 positive UNKNOWN.
Its observation gives no public basis to select different directions for its185
members. A different objective cannot repair that information limitation.

The prior experiment's32 out-of-prior cases all abstained. This audit generated
no counterfactual views for them and establishes no repair. The candidate's
weakest assumption is that finite-bank positive mass represents the true scene;
different hypothesis counts, omitted shapes, noise or pose error can invalidate
its choice. Zero wrong in a noiseless in-prior enumeration is structurally easy.
After one implementation check, repeating the same bank cannot establish
generalization; changed-observation/prior-coverage evidence would be a separate
question. No new execution is authorized by this report alone.

## Verification and artifacts

The audit rederives all13-by4 forecast partitions and reproduces old fixed and
adaptive outcomes for every ambiguous case. It checks sealed inputs and leaves
all11 consumed files byte-identical. A second arithmetic check enumerates every
reachable aggregate count pair without Pareto pruning and reproduces the frontier;
it also verifies retention of all fixed positive IDs and the simple tie rule.
This is same-agent verification, not an independent-agent review.

Script: [active_view_opportunity_audit.py](../../../../scripts/research/nearfield_opportunities/active_view_opportunity_audit.py).
Artifacts: `artifacts.local/work/ba-active-view-opportunity-20260921/audit-v1/`
contains the complete per-class audit, input hashes, completion receipt and
`independent-set-check.json`. It reads old forecasts, not a renderer or predictor.
CPU small-set arithmetic is `TASK_NOT_GPU_SUITABLE`; no persistent process remains.

```powershell
python -B scripts/research/nearfield_opportunities/active_view_opportunity_audit.py --source artifacts.local/work/ba-active-view-20260921/run-v1 --output artifacts.local/work/ba-active-view-opportunity-20260921/audit-v1
```

The script refuses an existing output directory. Any reproduction needs a fresh
audit path and remains saved-data analysis, not a scientific repeat or fresh test.
No old source/protocol/predictions, current decisions or global ledger was edited.
