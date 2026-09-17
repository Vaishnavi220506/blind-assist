# Existing-data tolerance re-evaluation: boundary pressure and transfer coexist

The requested re-evaluation is complete using576 existing frames and stored
decisions. No model was trained, no prediction recomputed, no threshold changed,
and no new capture was used. Both cohorts are consumed controlled Development.
Independent direct closed-AABB checks reproduce every frame's0/3/5/10cm state.
See the [fixed analysis definition](TOLERANCE_PROTOCOL_20260917.md).

At5cm both cohorts retain216/288frames(75%):108 core positives and108 clear
negatives. The72 boundary frames are exactly the inherited shallow-boundary
family in these cohorts. This is now verified from ALL native object bounds,
not assumed by excluding a family. All BODY, HEAD and rod/far-wall frames remain.
The lateral position criterion preserves thin central rods; x/z ranges remain
unchanged. No sensor-return availability or visibility determines truth.

| Native lateral tolerance | Old clear/288 | Old A TP/FP/FN; F1 | Changed-domain clear/288 | Changed A=S1 TP/FP/FN; F1 |
| --- | ---: | --- | ---: | --- |
| Strict0cm |288(100%)|137/28/7;88.67%|288(100%)|120/45/24;77.67%|
|3cm |228(79.17%)|112/22/2;90.32%|224(77.78%)|98/30/14;81.67%|
|5cm |216(75%)|107/22/1;90.30%|216(75%)|95/27/13;82.61%|
|10cm |216(75%)|107/22/1;90.30%|213(73.96%)|92/27/13;82.14%|

The10cm changed-domain definition additionally removes three correctly detected
HEAD-positive frames. It removes none of the40 clear errors remaining at5cm.
There is no monotonic improvement with a larger tolerance and no selected
"winning" tolerance.5cm is an illustrative user-specified prototype analysis,
not an adopted safety standard or a measured improvement of the model.

## What remains wrong after the boundary band is separated

At5cm,29/69 changed-domain strict errors(42.03%) lie in the boundary band.
The remaining40 are27FP and13FN. All27FP lie more than10cm outside the nominal
side boundary; all13FN have native lateral positional intrusion at least10cm.
The13FN comprise8rod and5HEAD frames. This position slack is not overlap width
or Euclidean clearance, and does not assert10cm depth on the unchanged x/z axes.
Background/source properties can co-vary; this is association, not attribution.

| Changed-domain family | Clear frames at5cm | A=S1 TP/FP/FN | F1 |
| --- | ---: | --- | ---: |
| BODY |72|36/6/0|92.31%|
| HEAD |72|31/8/5|82.67%|
| Rod/far wall |72|28/13/8|72.73%|
| Shallow boundary |0;72 separately scored|25/18/11 strict|63.29% strict|

The5cm clear-task recall is99.07% old versus87.96% changed; precision is82.95%
versus77.87%. Separating boundary pressure is useful, but the old/new F1 gap
still measures7.69percentage points. The [earlier descriptive transfer diagnosis](TRANSFER_DIAGNOSIS_20260917.md)
also shows both original cohorts have165 A alerts: discrimination changed even
though alert volume did not.

## All existing methods share the same definition

| Method/domain,5cm | TP/FP/FN | F1 | Recall | Core episodes detected |
| --- | --- | ---: | ---: | ---: |
| A old |107/22/1|90.30%|99.07%|18/18|
| B scalar-depth old |103/21/5|88.79%|95.37%|18/18|
| C global-depth old |94/1/14|92.61%|87.04%|16/18|
| S1 old |107/18/1|91.85%|99.07%|18/18|
| A=S1 changed |95/27/13|82.61%|87.96%|18/18|

C's high clear-subset F1 still misses two complete core episodes; it is not a
reason to reopen the closed online DA-V2 recipe. Global B/C predictions do not
exist on the changed cohort and were not inferred from sparse C invocations.
New A and S1 are bitwise-identical decisions and have identical results at all
tolerances. Old S1's four-FP Development gain is preserved, not promoted by a
different evaluation subset.

Boundary frames keep strict scoring: old A/S1 are30/6/6,F1 83.33%,36/72alerts;
changed A/S1 are25/18/11,F1 63.29%,43/72alerts. They are not counted as automatically
correct or relabelled according to prediction. Full288-frame strict results
remain unchanged in every table's machine-readable counterpart.

## Existing short trajectories: what they can and cannot establish

At5cm all18core episodes in each cohort already contain the obstacle at the
first observed frame: all are left-censored. New A/S1 alert at the first sample
in16/18; mean first observed core-alert delay is0.111s, maximum1.25s. Old A/S1
alert immediately in18/18. These are observed onset delays, not evidence of
pre-entry warning or a validated timely-alert window.

Clear-negative burden complements event detection: old A22/108frames(20.37%),
old S118/108(16.67%), changed A/S127/108(25%). At4Hz the accounting durations are
5.5s/4.5s/6.75s out of27s sampled clear-negative bins. The changed cohort has
16 nuisance segments and alerts in12/18 wholly negative episodes. Complete
episode-local transitions are46;15occur wholly inside the boundary band.
Transitions are a descriptive switching count, not all necessarily erroneous.

There are zero observed core-to-clear-negative release opportunities at5cm,
so release delay and sustained clearance are NOT_EVALUABLE. The six-frame
trajectories span1.25s; .25s per sample including the final bin is an accounting
convention. No interpolation, hysteresis, invented event window or temporal
filter was applied. The evaluator records first off where an opportunity exists;
that is not a claim of stable sustained release.

## Disposition and retained work

Main reporting should separate clear corridor decisions with coverage, strict
boundary pressure, and observed alert behaviour. This analysis does not alone
freeze a production tolerance or establish usefulness for real users. Prioritize
the clear HEAD/rod transfer errors; hardest boundary localization need not own
the whole research direction. No successor experiment is automatically started.

The earlier counterfactual-training implementation is explicitly deferred and
untrained. Its worker capture was cancelled when the user redirected the task;
all owned processes, scheduled task and cache port22248 were verified released.
210RGB files were emitted, but raw/evaluator buffers had not been flushed: these
are partial diagnostic files, not210admitted samples. Cancellation is an execution
redirect, not a failed scientific hypothesis. No report-source capture started.
The task's reproducible worker DDC(454,893,017logical bytes) and four local
untrained-pilot feature arrays(12,333,919bytes) were removed after ownership and
release checks. Original data, partial RGB, diagnostic timing/seals, result
tables, logs and cancellation receipts remain. Worker partial diagnostic payload
owner is this deferred intrusion task; location is recorded in local
`corridor-intrusion-train-20260917/source/cancellation.json`. It has no active
process or allocation and is retained to explain the interrupted acquisition.

Existing-data audit found MZ158288 complete and outside the proposed new model's
TRAIN list: it can support a disclosed held-out-group Development comparison,
not fresh confirmation because its historical outcomes were consumed. MZ146288
is also complete but previously used for S1 fitting/calibration. MZ176 has source
design/bundle only and no returned capture here. The15000body-query collection
retains a different RGB/native-depth/count-label contract; no compatible public
dual-return/Radar/IMU stream was established by this audit. Its RGB/geometry can
still be useful with an explicit adapter, not silently treated as current inputs.
Future work should inventory these roles before acquiring another source.

Artifacts: `artifacts.local/work/corridor-tolerance-20260917/` contains
`summary.json`,576-row `frames.csv`, native distance bins, all family/temporal
records, PNG/SVG overview, input/protocol/completion seals and
`independent-audit.json`. Five focused geometry/temporal tests pass. Standalone
figure visually inspected. Old predictions, labels, model weights and original
reports remain untouched; no Android/default-App change.

Supported registration was attempted. The pre-existing
`experiments/index.jsonl:303 input_fingerprint` mismatch prevents registration;
inheritance assignment consequently reports an unknown terminal. Evaluation
decomposition is documented as retained; ledger/inheritance metadata remains
pending and no ledger was manually changed. Receipts are `registration.log`
and `inheritance.log` beside the result.
