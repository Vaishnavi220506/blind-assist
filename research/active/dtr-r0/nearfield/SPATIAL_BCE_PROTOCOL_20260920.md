# One spatial RGB-ToF BCE pilot

EXPLORE controlled Development. User authorized one A/B experiment on 2026-09-20
after a read-only audit of MZ136, CCRL and background invariance. No ranking loss,
new hardware, second return, depth reconstruction, automatic successor or demo
replacement. Historical experiments and their dispositions remain unchanged.

## Question and specific change

Can a frozen pretrained local spatial RGB representation plus the original
single-return 8x8 ToF observations improve camera-forward Boundary coverage at
the frozen Core policy's false-alert burden on different procedural layouts?
MZ136 already tested task BCE versus pair hinge; its perfect development pair
ordering did not yield useful common-threshold discrimination. This tests a
different representation/input/label contract, not the novelty of pairing.

A uses the unchanged nominal45 single-return simulator, complete support score,
definite-support bypass and strong threshold 0.4071309640537889. Both A and B
are evaluated current-frame first, then with the SAME nonrecursive one-frame
hold at 0.2 s sampling. B is an independent classifier, without baseline OR,
veto, ranking loss or metric-range output. Raw returns and current UNKNOWN stay
unchanged for every arm, including when a learned alert is positive.

## Source and partitions

40 independent base geometry groups, four existing cuboid families, ten groups
per family. Each family's six train/two development/two test groups give
24/8/8 groups. Every group has INSIDE/BOUNDARY/OUTSIDE lateral interventions and
24 posed approach/dwell/retreat frames per clip: 120 clips, 2880 total frames.
All interventions and times stay in their base partition. No adjacent-frame
split, scene-ID feature, family feature or layout-label feature is allowed.
The fixed specification checks relative geometry disjointness across groups
and against the four earlier Core/full-event sources before capture.

Within lateral pairs target dimensions/material, camera trajectory and backdrop
are fixed. Depth changes along each trajectory create near/far observations;
these are longitudinal repetitions, not additional independent groups. Simulator
seeds are shared by group/time across lateral variants; conditional noise draws
can diverge when geometry changes. Original dropout/noise/return selection stays
unchanged; no histograms or native depth reach B. Native depth is used ONLY for
simulation, source admission and evaluator support audits.

Labels use full rendered object extent intersecting the existing camera-frame
volume X=[-.3,.3],Y=[-.2,.9],Z=[.3,3] m and existing 2 cm boundary band. Complete
Core clips include pre-entry/post-exit negatives. Missing returns never remove
a known-positive frame. Source admission requires target visibility, matching
rendered geometry and no competing corridor surface; a failure closes as source
NOT_EVALUABLE without silently replacing/excluding cases.

All source observations and private admission records are sealed before model
work. Train labels alone enter fit; development labels alone select threshold;
test prediction and label join is only activated if development admission passes.
Acquisition truth checks on all partitions are disclosed source admission, not
protected-blind evaluation. This is new same-simulator Development, not natural,
hardware or protected final confirmation. Background variation is a limited
procedural robustness check, not a causal background-invariance claim.

## One fixed representation and fit

Local MobileNetV3-small ImageNet1K V1 checkpoint (hash frozen) supplies features
through index3, frozen/eval, at aspect-preserving RGB 256x144. Each calibrated
ToF zone samples an ordered 3x3 feature grid (24x9 channels), without collapsing
that grid by mean/max. Add range/8, validity and x/y angle channels on the 8x8
lattice. Head: 1x1 Conv220->32, 3x3 Conv32->32, flatten2048->32->1, ReLU.
Only the head trains: seed20260920, BCEWithLogits, AdamW lr.001/wd.0001,
1200 updates of64 sampled TRAIN frames with replacement, last checkpoint only.
No augmentation sweep, class reweighting, early stopping, alternative encoder,
loss, fit, seed or threshold search after development/test outcomes.

Frozen encoder still executes online. Record its parameters and preprocessing,
encoding/fusion costs separately; CUDA offline measurements do not establish
edge-device feasibility. No target device performance claim in this experiment.

## Development selection and one test

Evaluate every distinct DEVELOPMENT logit threshold with atomic ties (>=), plus
one above the maximum. Use a single global threshold, never per-family/scene.
A threshold is admissible only when B retains EVERY A Core true-positive frame
for both current and held readouts, and Core and Boundary separately have no
increase in FP frames or FP segments, for BOTH current and held readouts.
This retention also preserves every detected Core event's in-event first time.
Choose maximal Boundary current TP, then Core current TP, then higher threshold.
If no threshold is admissible, close as DEV_NO_ADMISSIBLE_OPERATING_POINT;
preserve train/development exact-threshold diagnostics and do not activate test.
This strict budget is intentional for this bounded question, not a universal
requirement on future algorithms. A silent classifier cannot pass Core retention.

If admissible, freeze that threshold, infer TEST once and seal logits/predictions
before evaluator join. Success requires the same test constraints plus at least
10 percentage points Boundary CURRENT recall gain. Report current and held
TP/FP/FN, precision, recall, all-negative FPR, UNKNOWN, event coverage, first
time, internal silence, pre-entry/exit FP, segments and sampled duration, and
paired recovered/lost IDs. No mask output means IoU is not applicable.
Report failures by BODY/HEAD, group, lateral relation and valid target-return
support; a model alert is not an independently measured metric obstacle.

B versus A tests the complete representation/data/training package; it cannot
isolate RGB necessity, pretraining contribution or pair-data contribution without
further controls, which are outside this run. No ranking arm is authorized here.

## Execution, stop and delivery

One 2880-frame capture, 3600 s capture budget, one fit and one development
selection; at most one admitted test. Store all payloads under
artifacts.local/work/ba-spatial-bce-20260920. GPU-first frozen feature extraction
and fit; scalar source/metrics CPU TASK_NOT_GPU_SUITABLE. Preserve source hashes,
checkpoint, logs, labels, predictions, seals and negative dispositions. Mechanical
defects may be corrected with explicit pre/post hashes and unchanged scientific
recipe; consumed outcomes cannot select a repair or new recipe. Release task UE
process tree and task actors through the existing finally/cleanup mechanism.

Attempt supported experiment/inheritance CLI registration; preserve any metadata
failure receipt without editing/bypassing unrelated ledger errors. Complete the
scoped result, current decision entry, checks and non-force default-branch
delivery. Stop without a second experiment whether the outcome is gain, negative
or not evaluable. Existing Calibration/strong+hold demo and Android stay intact.
