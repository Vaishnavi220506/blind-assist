# Three-arm representation ablation: multi-hypothesis superiority not established

**The fixed comparison does not establish multiple RGB associations as the source
of A*'s good performance.** Raw sensor geometry plus unassociated RGB summaries
and a single observable association both reach clear99/5/9 (F193.40%), versus
multi/A*96/3/12 (F192.75%). Multi has fewer false alerts, with lower recall.
There is no joint frame/event dominance; retain the unchanged A* effect version
and the simpler controls as diagnostic evidence, without promoting a new winner.

This is one completed [predeclared EXPLORE ablation](REPRESENTATION_PROTOCOL_20260918.md)
on consumed controlled data. All21 fits completed once. There was no new capture,
report-driven model/threshold change, new loss, MLP, fusion or automatic successor.

## Matched experiment and implementation

All arms use the identical1344 training rows, strict labels, six saved whole-scene
folds, seed185017 and HGB structure:150 iterations,7 leaves, learning rate.05,
L2=1, minimum leaf8, no weighting or early stopping. Each arm independently
refits six folds and a final full-data model. Its single threshold maximizes
pooled1152 OOF clear F1, breaking ties by fewer FP then higher threshold. These
OOF scores are selection statistics, not independent validation.

| Arm | Columns | Representation change | Frozen threshold |
| --- | ---: | --- | ---: |
| Raw | 2224 | All2217 sensor columns plus7 nonassociation summaries | 0.4874581810097215 |
| Single | 2485 | One maximum-pixel-IoU RGB descriptor per usable nonmerged ToF return | 0.4656711585158168 |
| Multi / A* | 2485 | All overlapping descriptors, unchanged original grouping and quantiles | 0.5568065433174727 |

Raw removes259 hypothesis-dependent columns AND both ToF-conditioned seed/count
summaries. It still has native center/support geometry from calibration and IMU;
it is not raw ADC or a test of whether geometry in general is useful. Single
uses the same proposal frontend as multi, breaks IoU ties by lexicographic box,
and never selects using truth, range or corridor overlap. It keeps the same
pooling and column interface, so single versus multi is the narrow candidate-
multiplicity contrast. Raw versus multi tests the wider explicit-association
block; it cannot assign an effect to any one internal component.

The original sealed extractor was left unchanged. Its task-owned copy reproduces
all1344 cached training rows and288 original report extractions exactly. Every
arm preserves the2217 native columns, including128 ToF and4 Radar slots, merged,
missingness and RGB-external information. This preserves input evidence, not a
guarantee that independently trained classifiers retain all true alerts.

Multi's retrained final pickle is **byte-identical** to retained A*:
`dffd5c5a2d6d59c5a0546513148b21736d5aa8601e5a034bcdab9337295cd59c`.
Its1152 OOF scores, selected threshold and288 report scores also match exactly.
Raw model SHA256 is `85028453baa1a57616c6bf38edac354c5287484af8b12a4daefc914ae4e1e9e6`;
single is `b65ce7b7168a7c1125693b25c63709f366db97023982ebae0f9712f510561e7f`.

The288 report frames were excluded from fitting and threshold selection, but
their earlier outcomes had already been examined. They are consumed Development
report data, not fresh confirmation. Original MZ136 dev48/test48 remain excluded.
All three public predictions were sealed before report native labels were joined.

## Frozen-threshold report results

Clear retains216/288 frames (75%), with108 positive and108 negative. The72
boundary-pressure frames are reported separately, not discarded from strict
evaluation. Counts below are TP/FP/FN.

| Arm | Clear counts | Precision | Recall | F1 | Core events | Strict events |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 99/5/9 | 95.19% | 91.67% | 93.40% | 18/18 | 25/30 |
| Single | 99/5/9 | 95.19% | 91.67% | 93.40% | 18/18 | 26/30 |
| Multi / A* | 96/3/12 | 96.97% | 88.89% | 92.75% | 18/18 | 24/30 |

| Stratum | Raw | Single | Multi / A* |
| --- | --- | --- | --- |
| Strict all288 | 111/9/33; F184.09% | 110/9/34; F183.65% | 107/6/37; F183.27% |
| Boundary72 | 12/4/24; F146.15% | 11/4/25; F143.14% | 11/3/25; F144.00% |
| BODY clear72 | 35/1/1 | 35/1/1 | 35/1/1 |
| HEAD clear72 | 35/0/1 | 35/0/1 | 35/0/1 |
| Rod clear72 | 29/4/7 | 29/4/7 | 26/2/10 |

All shallow-family report frames are boundary-pressure frames, so its strict
counts equal the boundary row and it has no clear denominator. Strict P/R are
raw92.50/77.08%, single92.44/76.39%, multi94.69/74.31%.

Both controls rescue the same3 clear rod frames without losing any A* clear TP.
Raw adds3 clear FP and removes1; single adds2 and removes0. Clear false episodes
are4/4/3 and false segments4/5/3 for raw/single/multi. Each control improves
clear correctness in2 of18 clear configurations, ties14 and worsens2. Thus the
three-frame gain is concentrated, not a broadly replicated superiority claim.

The three rescued clear frames are rod scene1 `in_02`, scene1 `in_03`, and scene5
`in_03`. None has native corridor-return support: the first two have49/50 usable
returns elsewhere and the third has0. This cannot be described as recovering
the correct association of an observed corridor return. It also does not identify
which retained RGB or sensor feature caused the rescue.

Paired strict score ordering across144 authenticated lateral pairs is124/144
(86.11%) raw,123/144 (85.42%) single and124/144 multi. Multi does not improve this
paired measure over raw. Ordered pair frames are correlated within24 configurations;
144 is not the number of independent scenes.

## Native support, events and timing costs

All arms correctly alert on all76 clear frames with native corridor-return
support. Across the92 strict native-supported positives, raw/multi each detect82
and single79. Equal raw/multi counts hide an exchange: raw loses5 existing A*
native-supported TP and rescues5; single loses4 and rescues1. All these losses
are boundary frames. Input-slot preservation therefore must not be reported as
perfect native-alert retention.

On69 zero-return frames, counts are raw22/4/14, single22/4/14, multi21/4/15.
The clear subset has41 frames:17/2/4,17/2/4,16/2/5. Missing returns remain explicit
input states; no-alert or lack of a return is not a free-space declaration.

Core events and all their onsets are unchanged across the three arms, including
the existing A* rod scene5 delay. All18 core events begin with the obstacle
already present; these results do not show advance warning.

Strict event totals conceal identity changes. Raw gains3 A*-missed boundary
events but loses2 detected events (shallow scene1 exit and scene2 enter), leaving
25/30. Single gains3 but loses scene2 enter, leaving26/30. Raw improves the
retained scene5 enter delay from.5s to0; neither control delays another retained
event. New detections include delayed responses: raw scene1 enter and scene5
exit at.25s; single scene0 enter and scene5 exit at.5s. Full per-event identities,
release intervals, false segments and sample-time accounting are in summary.json.

## Equal-budget descriptive operating points

These curves use report labels posthoc, exactly as predeclared for diagnosis.
They are not new calibration, unbiased validation or replacements for the frozen
threshold table. Every arm gets the same diagnostic threshold scan.

| Diagnostic | Raw | Single | Multi |
| --- | --- | --- | --- |
| At most3 clear FP: maximum clear TP | 99TP/3FP; core18, strict23 | 96TP/3FP; core18, strict24 | 98TP/3FP; core18, strict25 |
| At least18 core AND24 strict events: minimum clear FP | 99TP/4FP; strict24 | 96TP/3FP; strict24 | 98TP/3FP; strict25 |

At equal FP, raw has one more clear TP than multi but two fewer strict events.
At the event-count constraint, multi uses one fewer clear FP than raw. Single
matches the3FP count but detects fewer clear frames than multi. Matched counts
do not guarantee the same events or onsets; there is no joint dominance here.
The multi diagnostic98TP must not be confused with frozen A*96TP.

## OOF selection and interpretation

| Arm | OOF clear864 counts; F1 | OOF strict1152 counts |
| --- | --- | --- |
| Raw | 405/23/27; 94.19% | 453/56/123 |
| Single | 406/17/26; 94.97% | 490/44/86 |
| Multi | 398/12/34; 94.54% | 483/34/93 |

This calibration evidence again shows a precision/recall tradeoff rather than
unconditional multi superiority. The matched ablation does not prove that data
quantity alone explains A*: data quantity was fixed, not varied. It shows that
the explicit multiple-association block is not necessary for similarly strong
clear performance under this recipe and dataset. Other sensor geometry, RGB
summaries, fitting and calibration remain possible sources of effect.

Decision: do not present "retaining association alternatives improves obstacle
judgment" as an established principal contribution. Retain the multi mechanism
as an implemented representation with an unproven superiority hypothesis. Keep
unchanged A* as the existing low-FP effect version. Retain raw and single as
`COMPONENT_OR_CHALLENGER / COMPONENT` diagnostic controls with their stated costs;
no runtime replacement is warranted by these mixed results. This one run does
not reject every possible multi-hypothesis method or justify a new training run.

## Verification, reproduction and retained evidence

Five focused tests pass for IoU/ties, empty candidates, merged/missing returns,
sensor parity and strict raw column membership. Public allowlist mutation is
checked once per source. Across1632 frames, multi features and native sensor
columns match exactly. The independent recount/replay audit passes3 final model
and18 OOF checkpoint replays, exact rational threshold reselection, all frame
counts and event onsets. Native labels/strata and sampled witness flags were
rechecked from authenticated evaluator records. The auditor reuses sklearn and
the native label contract; it is not an independent physical truth source.

A read-only design and interpretation review confirmed the narrow contrast and
the hidden boundary-event/native-support exchanges. No statistical significance,
different-generator, natural-scene, device or user benefit claim is made.

CPU execution uses Python3.11.9, sklearn1.9.0 and NumPy2.4.4. Backend receipt gives
GPU_BACKEND_UNAVAILABLE for this frozen sklearn/NumPy/OpenCV method, despite a
CUDA device being present. Six folds plus final fit take20.89/25.17/25.69s for
raw/single/multi. Paired frontend calls take195.95s on1344 training frames and
43.50s on288 report frames, excluding the extra original-extractor report parity
call. These are offline experiment costs, not comparative online latency.

```powershell
$py = 'E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe'
$route = 'research/active/dtr-r0/nearfield/corridor_fusion_v1'
& $py "$route/test_representation.py"
& $py "$route/audit_representation.py"
# The completed run is protected against overwrite. Original stage sequence:
# run_representation.py prepare; fit; predict; evaluate
```

Evidence stays under `artifacts.local/work/corridor-representation-20260918/`:
freeze and feature/source hashes, named2224/2485 matrices,21 model checkpoints,
OOF scores/calibration curves, public prediction seal, all288 cases, native
accounting, diagnostic frontiers, audit and logs. No persistent process, worker,
port or paid allocation was created; completed Python processes exited. No
task-owned disposable capture/cache tree was created; durable evidence is kept.

The supported registration call still fails on the pre-existing
`experiments/index.jsonl:303` input-fingerprint mismatch. Structured inheritance
also fails with `unknown terminal id`; intended control role is stated
above. Failure receipts are retained without rewriting or bypassing the ledger.
