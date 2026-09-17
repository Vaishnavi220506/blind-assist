# New-domain surface-support oracle: no frozen-A gain, mixed bottlenecks

**Frozen A remains95TP/27FP/13FN,F1 82.61%,recall87.96%,precision77.87% on
216clear frames(75%coverage).** Both native sampled-support and complete-face
support substitutions leave every alert unchanged. No capture, training, model
or threshold change occurred. This is a privileged geometry diagnostic on the
existing consumed288-frame changed-domain cohort, not a deployable model.

The informative result is the error decomposition: **8/13clear misses already
have returned points inside the corridor;5rod misses lack corresponding usable
returns. Completing the linked surfaces adds zero new clear-positive reachability.**
It does add support for four strictly positive boundary frames. This cohort
does not reproduce the old evidence that unsampled full extent is the main
missing quantity for clear BODY/HEAD/rod alerts.

## Exact intervention and the historical comparison

The cited old24/13/0to24/2/0 result was MZ138 on MZ136dev48 with MZ129's
analytic ToF/Radar reducer. Current A is a2485-feature HGB, not that reducer.
We therefore preserved A's pickle/threshold and all non-support features, and
changed only the six native-coordinate support endpoints of existing usable
ToF slots. The [protocol](SURFACE_ORACLE_PROTOCOL_20260917.md) was frozen before
oracle scoring. All original ranges,signal,sigma,status,masks,slot count,IMU,
Radar,region-center features,RGB hypotheses and pooling remained bitwise fixed.

Returned lineage identifies an actor and uniquely verified cuboid face. Complete
face support extends to that native face's bounds, including unsampled regions,
without clipping to RGB/zone. No unreturned face/object/peak was inserted. Native
XYZ also supplies privileged position/depth, not merely a better lateral edge.
Sampled and complete interventions share that authority and the same slots.

A's six endpoints require a per-slot AABB envelope. Multi-face union topology
cannot be fully represented there:90returns' hulls intersect while their exact
face union does not. Exact unions are audited separately and never disguised
as A's feature interface. No RGB feature or head was silently replaced with an
analytic corridor classifier.
Independent direct reconstruction confirms that a standalone hull-intersection
readout would introduce11clear FP(16strict FP), whereas the exact face union
has0. Those are representation-control counts, not A's unchanged alert counts;
the existing single-box interface cannot express the complete surface union.

| Frozen classifier input | Clear TP/FP/FN | F1 | Changed scores/all288 | Changed alerts |
| --- | --- | ---: | ---: | ---: |
| Original A |95/27/13|82.61%|0|0|
| Native sampled-point support envelope |95/27/13|82.61%|42|0|
| Complete linked native-face support envelope |95/27/13|82.61%|11|0|

216frames have changed support coordinates;72have zero usable returns and stay
unchanged. Maximum probability change is0.004188(sampled) and0.000853(full).
On strict288 all arms remain120/45/24,F1 77.67%; on the separate boundary72
all remain25/18/11,F1 63.29%. No existing TP was lost, FP removed or FN rescued.
Core episodes remain18/18, first-sample alerts16/18, mean observed delay0.111s,
maximum1.25s and clear-negative alarm burden27/108=25%. Existing timing/censoring
limits from the tolerance report remain unchanged.

## Why a correct support field hardly changes frozen A

The trained model has900splits using73features. Only one split reads any of
the768support endpoints: tree104,node10,`zone12.slot0.support_y_lo`, threshold
0.028522665612399578m. The root first requires
`zone28.slot0.center_z > 1.687285304069519`. The support split's two leaf
contributions differ by0.0303817raw-logit units, at most approximately0.00760
probability. The native surface coordinates do not feed A's separately pooled
RGB plane hypotheses or its regional center features.

Consequently this intervention is not a mathematical performance ceiling for
a differently trained or analytic support-aware model. Its negative result
closes the specific idea of fixing A by replacing this largely unused feature
field, not the general value of spatial support or learned fusion.

## The13 clear misses, inspected through actual returned lineage

| Missed family | A FN | Already returned corridor points | Full face adds support | No ray hits target | Target hit but no usable return |
| --- | ---: | ---: | ---: | ---: | ---: |
| HEAD |5|5|0|0|0|
| Rod |8|3|0|3|2|
| BODY |0|0|0|0|0|

The5HEAD frames belong to one episode's first five observations. They already
have53usable returns per frame and linked corridor points. Three rod misses
also already have corridor points. Their missed alerts therefore cannot be
explained by missing unsampled complete extent alone; A's decision/association
representation is implicated, without assigning a unique causal feature.

For three other rod misses, private native rays never hit a corridor object;
the frame still contains48/50/50usable returns from other surfaces. For two
rod misses, private rays hit the object but no usable return survives in the
entire frame despite a received ToF packet. Those unreturned rays are inspected
only after prediction sealing for diagnosis, never inserted into oracle inputs.
Existing-return support completion cannot create missing measurements.

## The27 clear false alerts and the cost of naive geometric veto

All linked exact full faces in these27negative frames are outside the corridor.
However17frames have no usable return at all:9missing packets and8received
packets without a detected return. Only10false-alert frames have usable supports
whose oracle geometry is explicitly outside. Their family split across all27
FP is BODY6,HEAD8,rod13; the zero-return subset is BODY3,HEAD3,rod11.

No-return is UNKNOWN, not clear-space evidence. A currently detects19clear true
frames without returned full-face corridor support:14rod,3HEAD,2BODY;18of those
frames have zero usable returns. Replacing the whole decision with
"any returned full face intersects" would gain8A misses and remove27FP, but
lose19existing A TP. Its diagnostic count would be84/0/24,F1 87.50%,recall77.78%,
not a no-loss solution. The sampled-point rule gives the exact same84/0/24 on
clear frames; complete extent supplies no additional clear-task reachability.
These counts are an information-readout control, not frozen A performance,
an implemented candidate, or a reason to suppress independent evidence.

Across all108clear positives, returned full faces support84:BODY34/36,
HEAD33/36,rod17/36. The others remain unobservable to this particular returned
surface readout. On strict144positives, sampled support reaches98 and full
faces102: all four completion gains occur in the boundary pressure family.

## Decision and integrity

The data do **not** justify committing the next research phase specifically to
RGB-guided completion of unsampled full extent. There are two demonstrated
priorities to distinguish in future work: use already-returned spatial evidence
correctly(8clear misses plus the10FP with observable outside supports), and
handle missing/unformed rod returns while retaining independent cues. This is
a diagnosis, not authorization for a new model, CNH, DA-V2 or capture campaign.
The earlier intrusion model remains deferred and untrained.

All6907public/usable returns have resolvable faces; no fallback was needed.
There are463multi-face and143multi-actor returns. All49729linked native points,
including5207corridor contributors, are retained within completed faces at the
inherited1e-5m audit tolerance. Baseline scores reproduce bitwise; every
non-support feature is bitwise unchanged. No native geometry is exported into
the public observation predictor. Native Radar lineage remains unavailable;
Radar features are unchanged rather than being reinterpreted as an oracle.

Artifacts: `artifacts.local/work/corridor-surface-oracle-20260917/` holds frozen
inputs,oracle feature vectors/scores,per-return faces/points,all288cases,the40clear
errors,native retention,summary,CPU backend receipt and completion. The five
focused tests cover missing/ambiguous lineage,unusable slots,unsampled faces and
multi-face hull gaps. Exact independent audit and standalone figure accompany
the report. No persistent worker or GPU allocation was started.
The independent audit directly reconstructs all faces/points from raw lineage
without importing the oracle or tolerance helper. It reproduces all saved
classifier scores, all fixed-feature checks, every clear/boundary/strict count
and the13miss diagnoses. The result figure was visually inspected.

The supported experiment-registration command again encountered the existing
`experiments/index.jsonl:303 input_fingerprint` error; terminal assignment then
reported unknown ID. Registration/inheritance remain pending, with both command
receipts retained. The scoped support-endpoint intervention is documented as a
NEGATIVE_CONTROL, without changing or bypassing the ledger.
