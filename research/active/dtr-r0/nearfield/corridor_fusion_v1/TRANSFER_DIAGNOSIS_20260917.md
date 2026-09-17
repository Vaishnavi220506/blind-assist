# Frozen A: descriptive transfer diagnosis

2026-09-17. Both inspected cohorts are now consumed controlled Development.
This diagnosis makes no fit, calibration or threshold change. It reproduces A's
complete saved score vectors exactly from authenticated features, including
newly extracted public features on the confirmation recovery capture.

A changes from137TP/28FP/7FN on MZ170288 to120/45/24 on targeted confirmation288:
precision83.03% to72.73%, recall95.14% to83.33%, F1 88.67% to77.67%. Both contain
144positive/144negative frames and165 A alerts. Thus an unchanged alert count
hides17additional false alerts and17additional misses; this is a discrimination
shift, not merely a change in overall alert volume.

| Family,72 frames each | Old TP/FP/FN | New TP/FP/FN |
| --- | ---: | ---: |
| Near rod / far wall |36/11/0|28/13/8|
| Boundary pressure |30/6/6|25/18/11|
| Substantial BODY |36/1/0|36/6/0|
| Suspended HEAD |35/10/1|31/8/5|

New physical wall-distance strata each contain48frames,24positive. At3.3m,
A is10TP/3FP/14FN;3.6m20/3/4;3.8m24/14/0;4.0m24/7/0;4.2m24/13/0;4.5m18/5/6.
The nearer-opening strata contain most misses, while3.8/4.2m contain more false
alerts. Wall distance, topology, target width and reflectance co-vary here;
these associations do not establish wall distance or any particular feature
as the cause. Old scenes do not have the same declared wall-distance/topology
schema, so those fields are explicitly missing/legacy rather than inferred.

Observable measurement composition also changes. Median usable ToF returns
rise10to18 and the upper quartile28to47.25, yet frames without any usable
return rise33to72. Dual-return-zone upper quartile rises0to1, maximum7to15.
Among frames with usable returns, median signal falls.04334to.03261; declared
sigma remains exactly.04m throughout. Sigma variation therefore cannot explain
this shift within the simulator. Nearest-geometry candidates are absent on77old
and92new frames; their placeholder zeros are counted as missing for range,
width and overlap distributions, not interpreted as measured zero geometry.

The report includes physical target width, HEAD/BODY, return-count/dual-return
and observable lateral-overlap strata with full denominators. These are
descriptive summaries of different scene groups. A new background-counterfactual
comparison is needed to separate background dependence from target differences;
this diagnosis does not prove a causal mechanism or rescue A by retuning it.

Artifacts under `artifacts.local/work/corridor-intrusion-20260917/diagnosis/`:

- `transfer-shift.png` and `transfer-shift.svg`: standalone four-panel score,
  range/return distribution and family-F1 comparison; PNG visually inspected.
- `frame-diagnosis.csv`:576 rows, physical/source fields explicitly analysis-only.
- `strata.json`, `distributions.json`: counts, metrics, availability/missingness
  and quantiles. Source labels are not new model inputs.
- `new-base-features.npz`: `base` float matrix288x2485 and `ids` in original
  capture order; `new-base-feature-seal.json` records SHA256 and exact A parity.
  Public sensor/geometry extraction only, reusable for authorized training.
- `input-seal.json`, `completion.json`: authenticated old E1 feature cache,
  old S1 predictions, new prediction/input seal and source/model inputs.

Actual new source is
`artifacts.local/work/corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1`;
predictions remain in `corridor-depth-confirmation-20260917/evaluation-v1`.
Recovery changed capture mechanics, not which outcomes enter this diagnosis.
Raw evaluator/native values are used only for scoring and physical strata.
