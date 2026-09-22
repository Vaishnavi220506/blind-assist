# Selected checkpoints have weak train-set localization and lateral query use

2026-09-22, completed inference-only diagnostic. Both selected models have the
same fixed query ranking in every one of1152train/dev images. The selected joint
checkpoint has weak localization even on its training observations; this is not explained
solely by a held-layout generalization gap or by the0.5mask cutoff. It retains
conditional distance information, but query-specific lateral discrimination is
poor. Preserve the original exact-recipe NEGATIVE_CONTROL; retain this diagnostic
as a COMPONENT of the evidence, not as a useful model component.

## Scope and fixed state

[Diagnostic brief](QUERY_OCCUPANCY_DIAGNOSTIC_20260922.md) and
[original negative result](QUERY_OCCUPANCY_RESULTS_20260922.md).
Replayed864train and288dev frames with the original classifier epoch16 and
occupancy epoch8, both in eval/inference mode. There is no optimizer, backward,
BatchNorm update, new checkpoint or threshold selection. These are selected
states, not the final24epoch states, which were not retained. A train failure
here does not establish that a changed or longer optimization cannot learn.

Original checkpoint and observation/train/dev label hashes were verified and
unchanged after execution. Original dev TP/FP/FN reproduced exactly. Prediction
arrays were saved before metric analysis. Mixed RGB/ToF observation files were
hashed and memory-mapped, but only train/dev rows were forwarded; no held
evaluation labels were opened and no held rows were inferred. This is consumed,
same-generator simulation Development, not new validation evidence.

## Fitting and query distinction

| Selected model/split | TP / FP / FN at original cutoff | Recall | FP rate | Frame AUROC | Frame AP |
| --- | --- | --- | --- | --- | --- |
| Classifier train |155 /65 /229|40.36%|13.54%|0.834|0.739|
| Classifier dev |21 /8 /107|16.41%|5.00%|0.793|0.657|
| Occupancy train |185 /82 /199|48.18%|17.08%|0.810|0.694|
| Occupancy dev |20 /8 /108|15.63%|5.00%|0.805|0.686|

Train has384positive/480negative frames; dev128/160. Constant-score frame AP
is0.444in both. Low fixed-cutoff recall alone is not evidence of absent signal:
frame AUROC is about0.8. The selected occupancy state has a small train/dev
ranking gap, while both splits have poor useful fixed-cutoff performance.

For the same image, compare valid occupied query scores with valid empty query
scores. Occupancy lateral ordering is correct960/1680=57.14%on train and
192/528=36.36%on dev; across BODY/HEAD it is1968/2616=75.23%and624/888=70.27%.
The classifier has exactly the same direction counts. These are correlated
query pairs, not independent trials. A follow-up read of saved predictions,
independently reproduced, finds exactly one strict ordering in all864train and
288dev images in both arms: query indices `[3,4,5,0,1,2]`, or
**HEAD-left > HEAD-centre > HEAD-right > BODY-left > BODY-centre > BODY-right**.
All15query-pair directions are fixed, without ties or reversals. Thus the pair
win/loss counts are explained by this fixed query preference and label mix;
they do not establish image-conditioned relative query ordering. In particular,
the primary centre max always chooses HEAD-centre in both arms, even for BODY
hazards. Absolute scores still vary by image and may encode obstacle/range cues.
Cross-height pairs here include different lateral positions, so their aggregate
is not an isolated height-only intervention.

The separate matched-layout comparison changes the obstacle's lateral position
while retaining base layout and posed frame. Occupancy ranks INSIDE over OUTSIDE
in125/192train pairs but only33/64dev pairs (51.56%); BOUNDARY over OUTSIDE is
123/192and35/64(54.69%). A different global threshold cannot repair pairwise
score ordering. No new cutoff was proposed or selected.

## Spatial heatmaps: weak ranking, not simply a missing cutoff

Pixel metrics preserve native foreground/background area inside mixed45x80
cells, omit invalid area and keep tied scores atomic. The numbers below are
macro averages over1400train/440dev queries with visible foreground.

| Occupancy localization metric | Train | Dev |
| --- | --- | --- |
| Foreground/background AUROC |0.5437|0.5764|
| Foreground AP |2.522%|2.787%|
| Constant-score AP / foreground prevalence |1.602%|1.595%|
| Mean foreground probability |0.0681|0.0700|
| Mean background probability |0.0658|0.0659|
| Original-cutoff IoU |0|0|
| Soft IoU |0.01015|0.01061|
| Peak-cell occupied fraction |1.727%|2.273%|

There is some subthreshold ranking signal, but its absolute localization quality
is weak. Centre BODY/HEAD queries alone are weaker: train/dev macro AUROC
0.5118/0.5521, AP1.107%/1.398%against constant-score0.745%/0.725%.
Correct-query maps improve AP over the same image's empty-query maps
by only0.0081percentage points on train and0.0122on dev on average. Query changes
alter map probabilities by about0.001mean absolute difference. This supports
weak query-specific spatial response, not proof that the queries have zero effect.

All masks remain below0.5(maxima0.4428/0.4178). However,1239/1400train and390/440
dev positive masks contain a cell with more than50%true foreground, so pooling
soft labels alone does not explain the absent fixed-cutoff foreground. There
are no geometric-positive queries without visible support in these two splits.

Foreground occupies only0.436%/0.413%of all valid query-pixel area, including
empty queries. The selected occupancy eval-mode CE/BCE/Dice components are
0.5585/0.0806/0.9905on train and0.6495/0.0803/0.9905on dev. Small pixel BCE
therefore coexists with near-one Dice loss. These losses are reconstructed from
probabilities clipped at1e-12; they are not the historical online training loss
or guaranteed bit-identical logits loss at extreme saturation. Observed mask and
distance probabilities in this replay do not enter that clipping region. Imbalanced
foreground supervision is a plausible contributor, not an isolated causal result.

## Distance signal survives conditionally

Seven-class positive-bin accuracy is59/1400=4.21%on train and2/440=0.45%on dev;
1341/1400and438/440positive queries choose no-intersection as their argmax.
If the no-intersection class is excluded, the correct distance bin wins within
the six occupied bins in85.64%train and70.45%dev queries. This is an evaluator-
conditioned diagnostic: it neither solves occupancy presence nor establishes
reliable measured range. It does show why the earlier conditional distance MAE
could look reasonable while endpoint occupancy and masks failed.

## Interpretation and delivery

The evidence prioritizes query-conditioned spatial learning as the unresolved
mechanism. The invariant score ordering is a direct measured defect of these
states, stronger evidence than a small average query perturbation alone. The
current network turns the query into global channel gain/offset
and uses pooled spatial features for distance; this can allow query preferences
and coarse distance cues without useful query-local masks. The current run does
not separate this architectural hypothesis from sparse supervision, checkpoint
selection or optimization. It also does not establish that more data, longer
training or a changed spatial model would fail.

Any later experiment should first demonstrate training-set query-conditioned
localization before treating a classification comparator as a strong learned
baseline. This task does not run that successor or reopen the consumed recipe.
Original thresholds, A/UNKNOWN, held-evaluation disposition and App remain unchanged.

Inference backend is actual CUDA, selected by equivalent CPU/CUDA forward
benchmarks; occupancy batch median103.83/7.21ms. Diagnostic execution and CPU
rank aggregation took13.06s before runtime bookkeeping. Eight focused synthetic
tests pass, covering weighted ties, invalid/empty area, query direction, matched
OUTSIDE pairs, float64 fixed cutoffs and low-probability heatmaps. The descriptive
figure was visually inspected. Independent audit PASS: four prediction hashes,
two original checkpoints, dev replay and all query pair margins/directions
reproduce. Weighted sklearn AP/AUC independently verifies every12th visible-
positive query (116train/37dev); all saved per-query rows reaggregate to the
reported macros. This is sampled independent pixel-ranking recomputation, not
a claim that every pixel metric was independently recalculated. The historical
capture-source EOF formatting difference is verified against the prior recovery
receipt and archived execution bytes; actual diagnostic dependencies match.
The process exited and no continuing worker remains.

Evidence root: `artifacts.local/evidence/ba-query-occupancy-diagnostic-20260922`.
Keep input-seal.json, prediction-seal.json, four prediction NPZs, result.json,
details.json, backend receipts, focused-tests.txt and diagnostic.png/svg. The
governed run specification and console log are adjacent files with `-run.json`
and `-console.txt` suffixes. independent-audit.json records the separate audit.
No original evidence tree was rewritten.

Terminal `terminal-query-occupancy-diagnostic-20260922` is explicitly assigned
COMPONENT_OR_CHALLENGER/COMPONENT for diagnostic evidence only. Archived run
`ba-query-occupancy-diagnostic-20260922` anchors source revision
`3be9045cd76a04d8e4e7bc16ddbad3649251f732`; all427earlier ledger rows remain
byte-identical, with428rows after registration. Delivery receipts are local.
