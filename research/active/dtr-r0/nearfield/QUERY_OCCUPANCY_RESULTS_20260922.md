# Single-frame query occupancy: no usable joint alert or localization gain

2026-09-22, completed user-authorized EXPLORE. The exact paired prototype is a
**NEGATIVE_CONTROL**, not a retained alert challenger or localization component.
The occupancy arm detects more events than its matched classifier but adds too
many false alerts, and its selected visible-mask output is empty at the fixed
0.5 cutoff. The unchanged ToF A reference is substantially better than both.
This closes this recipe, not the broader possibility of spatial learning.

## Comparison and source

[Protocol](QUERY_OCCUPANCY_PROTOCOL_20260922.md): six BODY/HEAD by lateral queries,
RGB plus regional 8x8 ToF observations, a common ImageNet MobileNetV3-small prefix,
regional fusion and query decoder. The paired change is binary supervision
versus joint first-intersection classification and visible-region supervision.
The latter predicts six distance bins plus a no-intersection class; its alert
score is the sum of occupied-bin probabilities. There is no separate alert head.
Both arms have158137parameters, the same initialization and image order, and
24epochs. This does not isolate first-hit versus mask supervision separately.

One successful capture produced1728frames from48new procedural base layouts,
with three lateral clips and12posed samples per layout. Whole layouts are
disjoint:24train/8dev/16evaluation, or864/288/576frames. Target fronts span
about0.47-4.03m. All frames remain, and no query was invalidated by the declared
unexplained-surface check. Labels use full rendered bounds of all declared
controlled cubes; visible labels independently use native optical depth.
All880positive held queries have visible occupied pixels. This is new-layout,
same-generator **controlled-object simulation Development**, not source-disjoint
natural evidence or real walking. Six fixed queries are evaluated; arbitrary
query generalization is untested.

The dev selection chooses classifier epoch16/cutoff0.5822458267211914 and
occupancy epoch8/cutoff0.40865737199783325. Their dev TP/FP/FN are21/8/107 and
20/8/108, each at5%FPR. Training finished its original allocation; no second
optimization attempt, held threshold selection or temporal fusion was used.
Checkpoints, source/input hashes and all576held public-input predictions were
sealed before the held-label evaluation join.

## Held alerts at the fixed dev-selected cutoffs

There are256positive and320negative frames,32events, and48clips. Truth coverage
is100%. FP rate uses all320known negative frames. Sensor UNKNOWN remains
independent of truth and alert:487/576frames are UNKNOWN, including all320negative
frames. Silent UNKNOWN is not a true-negative/free-space certificate.

| Arm | TP / FP / FN | Recall | Precision | FP rate | Events | False segments | Sampled false duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A current reference |193 /6 /63|75.39%|96.98%|1.88%|32/32|6|1.2s|
| A one-frame hold reference |225 /26 /31|87.89%|89.64%|8.13%|32/32|20|5.2s|
| Matched classifier |62 /34 /194|24.22%|64.58%|10.63%|16/32|17|6.8s|
| First-hit + visible mask |88 /44 /168|34.38%|66.67%|13.75%|29/32|29|8.8s|

Occupancy adds26TP and10FP, with12additional false segments. The10.16point
recall gain meets the magnitude target but exceeds both allowed costs (+2FP,
+1segment); it also fails the FP-reduction alternative. All44occupancy FP are
in OUTSIDE clips. Dev FPR5% did not transfer to either learned held point.
Whole-layout bootstrap95%percentile interval for recall difference is
[-1.95,+23.05]points (1000resamples), not evidence of a stable improvement.
The descriptive full curves cross; no replacement threshold was selected.

| Family (64positive/80negative frames each) | Classifier TP / FP / FN | Occupancy TP / FP / FN | Events classifier -> occupancy |
| --- | --- | --- | --- |
| BODY protruding plane |17 /12 /47|29 /14 /35|4/8 ->8/8|
| BODY suspended solid |18 /9 /46|13 /8 /51|4/8 ->7/8|
| HEAD hanging plane |10 /5 /54|23 /11 /41|2/8 ->6/8|
| HEAD horizontal |17 /8 /47|23 /11 /41|6/8 ->8/8|

BODY recall27.34% ->32.81%; HEAD21.09% ->35.94%. No classifier-detected event
is lost and13are gained, but BODY suspended-solid frame coverage falls by5/64.
In INSIDE clips TP33 ->43/128; BOUNDARY29 ->45/128. OUTSIDE FP34 ->44/192.
Detected-subset onset median/max is0.2/0.8s for classifier (16events),0.2/1.0s
for occupancy (29), and0/0.4s for A (32). Different detected subsets cannot
establish an onset improvement. All event identities and timing changes remain
in metrics.json; among16shared detected events,8are earlier and8later. Learned
exit-tail sampled duration is0s; A current/hold total
0.2/3.2s across32positive clips, with no censored exit. Posed0.2s sampling is
not measured human warning latency.

## Spatial outputs did not establish the proposed capability

Mean IoU across880visible-positive queries is **0.000 at cutoff0.5**, with maximum
predicted pixel probability0.4334. Zero false
masks on empty queries therefore does not indicate useful localization.
Seven-class argmax accuracy is74.48% over3456queries, but positive-query bin
accuracy is only13/880=1.48%. The always-no-intersection control would score
2576/3456=74.54%overall. Overall accuracy is consequently misleading here.

Conditional distance MAE is0.283m, computed after renormalizing only the six
occupied bins on geometric positive queries, even when no-intersection is the
winning class. It does not establish reliable obstacle detection or measured
metric range. The empty mask readout and weak positive-bin classification
prevent a localization-component retention claim. They do not prove that all
subthreshold features contain no information, or that another spatial model
cannot learn. No posthoc mask cutoff or loss change is made.

## Runtime, validation and reproducibility

Actual RTX5060 Laptop CUDA execution: classifier44.07s and occupancy47.16s for
24epochs each, peak allocated630560768/675774464bytes. Equivalent training-step
CPU/CUDA medians were393.50/27.78ms for classifier; both arms used the measured
CUDA backend. Cached-RGB/ToF to CPU-output single-frame p50/p95 is5.61/8.52ms
and5.68/8.46ms respectively (20samples each). This excludes camera/PNG decoding,
is a small host measurement, and is not endpoint or sustained latency evidence.

All37focused geometry, source/launcher, model, learning and independent-audit
tests pass. They cover query boundaries, invalid depth/UNKNOWN, matched initialization, source
seals, atomic threshold ties, low-probability numerical precision and sampled
event metrics. Independent saved-output audit PASS: all576frames/3456queries,
declared AABB geometry, groups, fixed cutoffs, four-arm frame/event/segment
counts and all family/layer/relation strata reproduce. Native rendering itself
is not independently reconstructed; native validity/area checks use the saved
visibility receipt. A separate independent component/decision audit reproduces
IoU, distance/class counts, event differences and both failed retention criteria.
Full predictions and errors are retained.
Git's whitespace check subsequently found one surplus EOF blank line in each of
two capture scripts. All17sealed execution source files were archived byte for
byte before this formatting-only correction; original seals/results stay intact.
`recovery/post-run-source-formatting.json` records old/new hashes and exact
line-ending/EOF equivalence. The auditor's `--source-dir` can verify the archived
execution bytes, and the delivery audit passes using that snapshot.

Two launch integration failures happened before UE capture: output/input
catalog overlap, then the runtime's precreated empty output directory. Stages
were separated into catalog siblings; the launcher now accepts only an empty
directory owned by its running journal. Original failed journals and a hash
repair receipt remain. Scientific cases/specification were unchanged. The
single successful capture, materialization, fit, prediction and evaluation all
passed their governed run contracts. UE's observed process tree was released;
training/evaluation exited. No continuing worker or paid allocation remains.

Evidence uses the canonical artifact junction. Root is
`artifacts.local/evidence/ba-query-occupancy-20260922`; the immutable stage roots
append `-capture`, `-prepared`, `-fit`, `-predictions`, and `-evaluated`.
Root contains the plan, stage run specs/console logs, recovery receipts and
independent-audit.json, independent-audit-delivery.json and
independent-components-decision-audit.json. Evaluated contains metrics.json, frame-results.json,
comparison.png/svg and fixed-rule examples.png. Large data and checkpoints
remain local evidence, not Git payloads.

Each stage runs through `tools/ba.ps1 run research-ue -RunSpec <saved-spec>`;
`run_query_occupancy.py` exposes the stage implementation. Existing result
directories cannot be reused for another fit. A new authorized experiment
requires distinct provenance rather than overwriting these outputs.

Retain A and its UNKNOWN semantics. Preserve this exact supervision/model/
selection recipe as a NEGATIVE_CONTROL and the new source as consumed
Development evidence. A broader spatial-learning hypothesis remains open,
but adding video before useful single-frame spatial readout is demonstrated
has no support from this result. No App change or successor run is included.
