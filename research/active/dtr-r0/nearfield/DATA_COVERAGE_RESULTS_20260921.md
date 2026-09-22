# Broader data with the fixed spatial B: partial gains, joint budget failure

The single completed fit improves Core errors and HEAD-horizontal coverage on
new layouts, but exchanges substantial BODY coverage for HEAD coverage. It does
not improve aggregate held Boundary recall over the matched-calibration old B,
does not transfer the development false-alert budget, and regresses on the old
consumed cohort. Close this exact data condition as `NEGATIVE_CONTROL` without
another fit, threshold change or automatic successor. This is a completed
algorithm result, not an unevaluable oracle or proof that better data cannot help.

## What ran

[Frozen protocol](DATA_COVERAGE_PROTOCOL_20260921.md), output
`artifacts.local/work/ba-data-coverage-20260921/`. One capture of 48 new geometry
groups, each with INSIDE/BOUNDARY/OUTSIDE complete 24-frame clips; whole groups
split 24 train / 8 dev / 16 evaluation, or 1728 / 576 / 1152 frames. All 3456
frames pass source admission with no exclusions or replacement. Capture takes
2031.59 seconds inside the capture script; launcher also includes editor startup.
Same map, poses/timing ranges, renderer warmup, camera and hypothetical single
ToF return law. Geometry signatures are disjoint from the six prior cohorts.

Change only the bundled data condition: broader axis-aligned cuboid dimensions
and heights, with appearances crossed across distinct geometries. Training row
count, frozen MobileNet features, 81921-parameter spatial B head, seed, unweighted
BCE, optimizer, batch64 and 1200 updates remain the original recipe. One N fit
takes1.84s on the measured CUDA backend; this excludes rendering, RGB feature
preparation and probes. No depth reconstruction is trained.

This is new-layout **same-generator simulation Development**, not a protected
final test, natural object diversity or hardware evidence. It changes metric
shapes and appearance assignment together, not semantic topology or identical
geometry rendered in every appearance. Individual causes are not isolated.
Evaluation geometry is used by the evaluator and sensor simulator, never as a
model input. The original protected test remains unactivated.

A and the saved B/R weights and cutoffs are frozen references. `B_control` uses
the old B weights with a separately named experimental calibration on the new
dev split, under exactly the same selection rule as N. It never replaces the
saved B working point. Selected logits: B_control6.888704776763916;
N7.03014612197876. Legacy B7.6612162590026855 and R5.128307342529297 stay fixed.
Both selections pass every dev cost cap; Boundary held TP is57/81 for B_control
and43/81 for N. No evaluation labels select thresholds. All1152 public-input
evaluation predictions are sealed before the label join.

## Primary new-layout result

Counts below are TP / FP / FN. Current output is A OR its supplement; held output
applies the unchanged hold to current output. These are saved selected working points under a shared budget rule,
not equal-realized-FP dominance or lossless preservation of B's individual flags.

| Arm | Core current | Core hold | Boundary current | Boundary hold | Boundary events, hold |
| --- | --- | --- | --- | --- | --- |
| A |154 /13 /22|159 /25 /17|8 /3 /168|11 /5 /165|3/16|
| B frozen |168 /42 /8|171 /63 /5|92 /10 /84|100 /19 /76|12/16|
| R frozen |154 /19 /22|159 /33 /17|46 /3 /130|57 /7 /119|9/16|
| B_control |169 /44 /7|172 /66 /4|98 /10 /78|106 /19 /70|12/16|
| N, broader data |174 /18 /2|175 /34 /1|94 /11 /82|106 /19 /70|13/16|

N and B_control both have60.23% held Boundary recall,84.80% precision and9.13%
FPR. N reduces Core held FP66 to34 and increases Core TP172 to175. Its Core
held precision/recall/FPR are83.73%/99.43%/5.74%; B_control72.27%/97.73%/11.15%.
These measured Core improvements must not be erased by the failed joint gate.

N gains held Boundary TP over A in12/16 groups. It detects13/16 Boundary events;
three remain wholly missed. The maximum first-in-event delay among detected
Boundary events is.4s, versus.6s for B_control. These maxima refer to different
detected event sets and do not establish that every B event is retained or faster.
Core events are16/16 for N/B versus15/16 for A/R. N actually supplements A's
missed Core event; retaining A's existing flags/onsets itself remains a structural
property of A OR and unchanged hold, not independent model benefit.

UNKNOWN stays exactly the original A evidence status in every arm: Core681/768,
Boundary384/384 current frames. It can coexist with an alert; no supplement
converts this evidence status to known-clear. UNKNOWN is not used as negative
truth. Full counts, precision/recall/FPR and unknown accounting are in metrics.json.

## Why the joint decision fails

All costs are increments over A, not total errors. The inherited callable rule
constrains false-alert segments in BOTH current and held outputs. The frozen
text abbreviated this as held segments; a pre-fit protocol-clarification receipt
records the full unchanged inherited rule. No criterion changed after outcomes.

| N cost | Added FP / cap | Added false-alert segments / cap | Pass |
| --- | --- | --- | --- |
| Core current |5 /5|5 /4|No|
| Core hold |9 /11|7 /4|No|
| Boundary current |8 /2|7 /2|No|
| Boundary hold |14 /4|10 /2|No|

Frame-level Core costs improve substantially, but extra alert segments still
exceed their budget. Boundary held FP remains19, equal to B_control, versus A5.
Thus this is not just a nearly missed recall floor: N passes the50% recall and
8/16 group-gain criteria, but fails costs and the prespecified10-percentage-point
held recall gain over B_control (actual gain0). The admissible dev working points
did not transfer their low-error costs to new evaluation layouts.

R on this new cohort has useful lower-cost rescue57/7/119 and9/16 events, but
adds6 Core current FP against cap5 and has32.39% Boundary held recall. Its old
consumed-cohort45.09% result and original disposition remain separate evidence.

## Aggregate equality hides a coverage exchange

All rows below are held Boundary TP / FP / FN; events are detected/4. Denominators
differ by family within this new cohort. Do not compare these46 horizontal
positive frames directly to the old cohort's43 as a matched frame improvement.

| Family | B_control counts; events | N counts; events |
| --- | --- | --- |
| HEAD horizontal |12 /1 /34;2/4|43 /7 /3;4/4|
| HEAD hanging plane |20 /5 /20;2/4|30 /7 /10;3/4|
| BODY protruding plane |38 /9 /6;4/4|9 /3 /35;2/4|
| BODY suspended solid |36 /4 /10;4/4|24 /2 /22;4/4|

The new condition gains31+10 HEAD positive frames and loses29+12 BODY positive
frames relative to B_control. Horizontal recall rises26.09% to93.48%, accompanied
by FP1 to7. BODY plane recall falls86.36% to20.45%, with two formerly detected
events now wholly missed. Therefore equal total TP is not retention of the same
rescues, and the horizontal gain does not establish balanced coverage.

N adds106 current true-positive frames over A across Core and Boundary;35 lack
a returned target-and-corridor sample under the saved lineage count. This is an
evaluator description of model rescue, not a whole-object geometry guarantee or
proof that missing sampled support makes a frame impossible to classify.

## Secondary consumed-cohort regression

Only after primary evaluation sealing, run the unchanged selected cutoffs on
the old1152 frames. This cohort already shaped the research hypothesis and is
**regression evidence only**, not another independent confirmation. Original
RGB feature cache hashes and saved A/B/R predictions are verified and retained.

| Arm | Core hold | Boundary hold | Boundary events |
| --- | --- | --- | --- |
| A |173 /33 /0|14 /5 /159|4/16|
| B frozen |173 /58 /0|140 /19 /33|15/16|
| R frozen |173 /38 /0|78 /6 /95|11/16|
| B_control |173 /60 /0|145 /20 /28|15/16|
| N |173 /50 /0|87 /20 /86|12/16|

All Core events remain16/16. N loses58 held Boundary TP relative to B_control
at the same20FP, or53TP relative to legacy B while adding1FP. Its50.29% Boundary
held recall does not rescue the failed new-layout costs or make this a broad
upgrade. Do not silently combine these173 positives with the new176 positives.

## Runnable delivery and checks

Code: data_coverage_spec/capture/data/learning/regression.py, launcher and
run_data_coverage.py; the saved new model is `fit/head_last.pt`. The reusable
spatial_bce_model observation API remains the inference implementation. Root
stages are freeze, freeze_execution, materialize, prepare, fit, select, predict,
evaluate; capture uses launch_data_coverage.py with the frozen spec/protocol.
The completed output is immutable; these entry points do not authorize reruns.

Twelve focused source/selection/retention tests pass. Independent audit recomputes
A from public ToF, all five current/hold outputs, UNKNOWN, confusion counts,
costs/segments, all-arm complete event counts and first delays, final gates and
the unchanged training recipe. Fresh public RGB+ToF replay of N on64 fixed
frames across all16 evaluation groups has exactly matching scores and supplement
decisions (maximum logit/probability difference0). No training occurs in replay.

N/B feature extraction and head inference select actual CUDA after workload
probes; R's small head selects CPU_FASTER_MEASURED. N's batched1152-row head
inference is.0507s, excluding RGB preprocessing;64-frame public replay including
fresh encoding/probes takes1.85s. These are host execution receipts, not Android,
sensor-to-alert latency or incumbent-speedup evidence.

Protocol/spec/capture/inherited dependencies were sealed before capture. The
orchestrator/materializer/learning wrapper and regression code were sealed before
materialization and fitting, as allowed by the protocol; audit code was separately
pinned before fitting. Preserve all capture, observation, fit, selection,
prediction, evaluation, regression and independent-audit receipts.
The source-snapshot directory and its seal retain the exact30 executed Python
source files' bytes, including line endings that Git may normalize. Two frozen
EOF blank lines are preserved; the default diff check reports only those two
cosmetic warnings and passes with blank-at-EOF excluded. Documentation index
validation passes for12 hot files and579 local links.

Local structured inheritance assigns this exact condition NEGATIVE_CONTROL.
Supported registration remains blocked by the existing index.jsonl:303 input
fingerprint mismatch; supported inheritance reports unknown terminal. Command
receipts and local-inheritance.json are retained without modifying the global
ledger. One initial registration invocation rejected an uppercase protocol ID;
the corrected invocation reached the existing ledger blocker, with both receipts
preserved. This metadata gap does not turn the experiment into registered evidence.

All task-owned capture descendants and execution sessions have exited; no paid
worker or resident service remains. Keep dataset, checkpoint, caches and receipts
under the canonical artifacts.local junction for reproduction. A/demo and prior
B/R/U/G and POINT/REGION dispositions are unchanged. No further training,
threshold/loss adjustment, promoted default or automatic successor is started.
