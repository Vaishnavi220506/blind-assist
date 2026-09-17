# S1: conditional frozen-C veto retains a small matched Development gain

2026-09-17, user-authorized specialist-role experiment after E1/E1-I.
[Protocol](SPECIALIST_PROTOCOL_20260917.md),
[adopted balanced research baseline A](BALANCED_BASELINE_20260917.md).

**S1_CONDITIONAL_SPECIALIST_DEVELOPMENT_GAIN.** On consumed MZ170288, one cheap
observable gate plus frozen C changes A137TP/28FP/7FN to137/24/7: precision
83.03%to85.09%, recall95.14%unchanged, F1 88.67%to89.84%. All30events and their
first-alert times survive exactly; native-supported missed-warning cases are
unchanged. Only10/288frames invoke C. Retain this exact conditional challenger
for an unchanged, separately scoped confirmation, without tuning on this report.
It does not reach90%F1 or prove natural-distribution/device effectiveness.

This reverses neither earlier negative result: global scalar-depth B and global
token C still fail their original tasks. The user changed C's role to a veto
expert; A remains the balanced research baseline, and the App is unchanged.

## Matched policies

| Policy | TP/FP/FN | Precision | Recall | F1 | Events detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 137/28/7 | 83.03% | 95.14% | 88.67% | 30/30 |
| Global C | 103/3/41 | 97.17% | 71.53% | 82.40% | 21/30 |
| **A + conditional C veto** | **137/24/7** | **85.09%** | **95.14%** | **89.84%** | **30/30** |
| Gate directly vetoes, no C | 131/24/13 | 84.52% | 90.97% | 87.63% | 30/30 |
| Privileged rod-family switch | 137/17/7 | 88.96% | 95.14% | 91.95% | 30/30 |

The last row uses evaluator family labels after prediction sealing. It is the
user-proposed conditional diagnostic ceiling, not an implemented policy or
global optimality bound. The deployed gate never receives family labels. The
four corrected FP are4/11of A's rod FP and4/28overall (14.29%overall reduction).
The remaining7rod FP are untouched. All288frames remain counted, with144positive
and144negative; conditional nonalerts=127, always UNKNOWN rather than clear.

The gate alone would suppress6true frames among its10invocations. C accepts
all6and rejects the other4, which are false A alerts. Thus C contributes to
this measured policy rather than merely sitting after an effective cheap veto.
Gate-only results are an ablation, not a selected fallback or successor.

## What the gate actually learned

One depth3/min_leaf6 tree, random_state178017, trained only on A-alert rows of
the MZ146 gate-training subset. Target1 is an A false alert that C rejects;
all other examples are0. True alerts that C rejects receive8xweight. No family
classifier, large network, A/C refit, PCA/layer sweep, new loss or new sensor.

The gate inputs are inexpensive public geometry summaries, raw regional ToF
distance/noise/signal/dual-return statistics, input masks and A score. C score,
depth, tokens, evaluator geometry, scene ID and family do not enter invocation.
After the selected threshold.9, the invoking leaf simplifies to:

`A alert AND nearest.range_m.q100 > approximately 3.93 m`

Here `nearest.range_m.q100` is an inherited observable RGB/ToF plane-hypothesis
feature, not a measured object's true depth. The full tree is saved in `gate.txt`.
All10report invocations happen to be rod/far-wall frames, without a family input.
This is evidence of a narrow learned context proxy in the same generator; it
does not establish a general thin-structure detector or physical ambiguity model.
Other candidate gate features were available but are not responsible for the
selected invocation leaf. Do not claim they were all useful.

## Fitting and selection isolation

A/C/PCA/checkpoint and both original thresholds remain frozen. Gate meta-data
is existing consumed MZ146288, whose observations were not used to fit A/C.
Whole family/scene groups0–3 form TRAIN192;4–5 form DEV96. Original A/C TRAIN192
is not used for fitting the gate because their fitted outputs provide an
unrealistically easy error surface. No original test48 or new capture is read.

Among gate TRAIN,109rows have A alerts;18are useful potential C vetoes and25are
harmful potential C vetoes. DEV A has46TP/19FP. One tree is fitted; only its
invocation threshold is selected on DEV. Minimize FP subject to at most1lost
A TP and no previously detected event lost or delayed more than.25s; ties
prefer fewer lost TP, fewer invocations and higher threshold. No-call fallback
is available. Selected.9invokes6DEV rows, removes3FP and loses1TP; the other
eligible option is no call. This DEV cost remains disclosed despite the cleaner
report behavior. The selection rule is new and prespecified, not a universal gate.

MZ170288 is used only after the new tree/threshold are sealed. It is historically
consumed and inspired the hypothesis, so this is Development transfer, not a
fresh test. An unchanged prospective same-source/independent-source evaluation
would be needed for stronger claims. No result-based gate/threshold revision
or fresh confirmation was performed in this task.

## Strata, events and native evidence

| Family | A TP/FP/FN | Conditional TP/FP/FN |
| --- | ---: | ---: |
| Near rod / far wall | 36/11/0 | 36/7/0 |
| Shallow boundary pressure | 30/6/6 | 30/6/6 |
| Substantial BODY | 36/1/0 | 36/1/0 |
| Suspended HEAD | 35/10/1 | 35/10/1 |

Ordinary216frames change107/22/1to107/18/1; pressure72frames remain30/6/6.
All positive event records and six observed exit/release records are exactly
equal to A. Negative segments remain18, while4Hz false bins decrease7sto6s
(negative-time occupancy19.44%to16.67%). Paired binary decisions both correct
increase109/144to113/144. These binary policies have no comparable continuous
PR-AUC or score-ranking claim; JSON's pair score is binary decision ordering.

The native audit preserves exactly the same5nonalert frames/57returned ToF
corridor samples as A. C-global has24frames/286samples; gate-only has7/76.
The conditional branch introduces no new native-supported missed warning on
this report. Native per-return Radar lineage remains NOT_EVALUABLE. No raw
sensor return is deleted, but final warning suppression is explicitly counted.

Corrected FP IDs (prefix `mz170_near_rod_farwall_`):
`scene1_out_00`, `scene1_out_01`, `scene1_out_03`, `scene5_out_00`.

## Actual conditional runtime

Every report frame was replayed online with caches bypassed for the measured
path: RGB decode → base features → A → cheap gate if A alerts → C only on
invocation. All288online flags and invocation decisions match sealed outputs;
all10C scores match cached frozen scores within1e-12. No all-frame depth
execution is hidden in the reported invocation rate.

| Complete warm path,288matched frames | Mean | p50 | p95 | Observed maximum |
| --- | ---: | ---: | ---: | ---: |
| A | 67.16ms | 66.57ms | 75.09ms | 83.32ms |
| Conditional | 70.02ms | 67.20ms | 76.96ms | 151.69ms |
| Conditional, invoked10 | 143.01ms | 143.55ms | 149.33ms | 151.69ms |
| Conditional, not invoked278 | 67.39ms | 66.80ms | 75.63ms | 83.33ms |

Invocation10/288=3.47%of all frames, or10/165=6.06%of A alerts. Mean additional
cost2.86ms/frame (~4.26%). The observed maximum is not a worst-case guarantee.
Keep C resident; measured depth-model construction/load-to-CUDA takes3.02s,
excluding other application initialization. Peak allocated CUDA427,203,584bytes
(~407MiB); torch/cuda/device match E1-I. CPU OpenCV/sklearn gate is reused;
encoder inference uses CUDA, with its identical-workload backend receipt.
Physical capture/transport, sensor age and physical-device performance are not
included. The A timing sample is now288frames, so it is not directly comparable
to earlier24frame p50 values. The model is an encoder specialist; it does not
need to render a dense depth map during conditional inference.

## Validation, delivery and decision

Two policy tests pass: C can only veto, nonfinite specialist output keeps A,
and expensive/evaluator metadata does not enter the cheap feature function.
Independent read-only review checks gate inputs, frozen expert identities,
whole-group split, threshold selection, report sealing and actual conditional
execution. Saved-output audit verifies hashes, policy, counts, group isolation,
oracle arithmetic,288latency rows/10calls and all288decoded video frames; preview
visually inspected. Initial parse-time tuple syntax error was fixed before any
scientific process started. Successful run completes in76.84s; no retraining,
threshold rescue or report retry. Task-owned inference exits and releases GPU.

Artifacts: `artifacts.local/work/corridor-depth-specialist-20260917/` retains
gate model/text/features, immutable expert bindings, meta caches, development
curve, selection/prediction seals, per-frame scores, native audit, complete
latency samples, all-frame comparison video, source snapshot and completion.
Original E1/E1-I models/payloads remain unchanged and reusable.

Retain A as the user-selected **balanced research baseline**. Retain this S1
recipe as **COMPONENT_OR_CHALLENGER / CHALLENGER**, for separately authorized
unchanged confirmation. No App promotion or broad specialist generalization.
Registration/inheritance remain pending the existing ledger303 fingerprint
error; supported commands fail without manual bypass.

The proposed CNH fallback is not triggered by this scoped gain. A source check
finds existing hypothetical81-bin ray/reflectance histograms only under
evaluator-only `zonal_tof_native`; current public observations expose at most
two returns, no CNH. Future richer-distribution work must introduce a separately
declared public simulation measurement interface or actual hardware data. It
must not silently consume evaluator histograms or call them calibrated CNH.
