# Delayed onset: observation-to-zone-to-alert audit

2026-09-20 EXPLORE; one consumed diagnostic, no alert candidate or new experiment
on fresh source. Preserve Calibration, fixed high threshold and one-frame hold.
Question: what observable64-zone support remains at f0005/f0293, where did its
readout weaken, and are proposed structural explanations exclusive to positives?

Source: ba-core-workpoint-transfer-20260920 complete432; Core78P/210N, Boundary
layout78P/66N. Inspect onset triples f0004/5/6 and f0292/3/4. Do not omit negative
phases or count the high threshold's8FP as the entire negative population.

## Fixed observable audit

Recompute unchanged Calibration from saved64 boxes/ranges, checking every
observation hash and original predictions, anchors, zone scores and decisions.
Keep absent ranges absent. Emit64 records per frame: range, full interval, possible/
definite, depth fraction, angular-given-depth, raw joint and effective possible
joint. No interval narrowing or new physical confidence claims.

Extract the following descriptive features on all432 before label joins:
valid/possible/definite count, number of positive-score zones, maximum depth
fraction among possible zones, top1/2/3 score sums, total joint mass, top1/total
concentration, four-neighbor positive-zone components and their largest mass/size,
and the exact positive-zone mask. Connectivity never crosses a row boundary.
These sums are descriptive and not probabilities, alternate scores or alert rules.
No fitting, feature selection, weights, connected-support algorithm or threshold
search follows. Anonymous observable feature extraction excludes truth, class,
layout and native geometry. Seal features before the root's label join.

For both target frames report each scalar's empirical range/median in allCoreP,
other76P, HEAD42P, strong HEAD37P, BODY36P, Core210N, INSIDE66N and OUTSIDE144N,
plus Boundary144. Report the counts of negatives at/above and at/below each target
value, not a selected classifier's accuracy. Count negatives with the exact same
positive-zone mask. Also count negatives simultaneously no smaller on the fixed
support-magnitude tuple (top1,total mass,largest connected mass,positive-zone count,
maximum possible-zone depth fraction). This tests that specific monotone support
intuition only; it is not an exhaustive64-vector separability or information ceiling.
Do not treat lack of exact noisy-vector equality as useful discriminative evidence.

## Evaluator-only explanation and limits

An independent bounded source audit examines ONLY the six selected native frames:
hashes, true target/corridor samples, proxy candidate bins, observed winner lineage,
zone contributors and public score factors. Reuse the original simulator exactly
for identity verification; dropped/losing returns stay evaluator-only, never become
public support. Distinguish geometric possibility from target-owned observed returns
and from contributors actually inside the native corridor. Native masks are for
attribution only. This is not an oracle rescue, model input or causal intervention.

Output an evidence table from raw supports through score factors to max/threshold,
with exact adjacent-frame transitions and counterexamples from all negative periods.
Single-frame factorization is arithmetic diagnosis, not proof of a physical cause.
If a distinction appears, describe it as a future hypothesis without implementing it.
If these features overlap, limit the conclusion to these features and sampled source;
do not announce8x8 single-return's information limit or choose extra sensing by default.

Stop after one saved-data audit, independent check, report/current/disposition and
scoped delivery. No capture, training, timing/window changes, threshold sweep or
automatic successor. Scalar/array source attribution uses CPU (TASK_NOT_GPU_SUITABLE).
Keep outputs/seals/receipts in artifacts.local/work/ba-onset-zone-audit-20260920/.
