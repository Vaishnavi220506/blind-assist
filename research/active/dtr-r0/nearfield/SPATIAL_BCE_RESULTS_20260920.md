# Spatial RGB-ToF BCE: strong Boundary signal, failed Core replacement

Frozen 2026-09-20; completed 2026-09-21. One new 40-group/2880-frame capture and
one ordinary BCE fit are complete. **The fixed model improves Boundary current
classification on eight development groups, but no common threshold meets the
prescribed Core retention and false-alert budgets.** Close this fixed recipe as
`NEGATIVE_CONTROL` for full Core/Boundary alert replacement. Retain the learned
scores and Boundary finding as scoped diagnostic evidence, not a promoted policy.

No sorting loss, extra input, second fit, threshold relaxation, test inference,
demo change or automatic successor occurred. Existing Calibration and Core
strong+hold remain unchanged. This is not evidence that RGB has no useful signal.

## What was implemented

[Protocol](SPATIAL_BCE_PROTOCOL_20260920.md), [source](spatial_bce_spec.py),
[model](spatial_bce_model.py), [runner](run_spatial_bce.py),
[independent audit](audit_spatial_bce.py).

Four cuboid families, ten new base groups each, split within each family6/2/2:
24 training groups/1728 frames,8 development/576,8 test/576. Every group retains
all lateral variants and all24 approach/dwell/retreat frames in one partition.
Each group has INSIDE, BOUNDARY and OUTSIDE clips; entire Core clips include
pre-entry and post-exit negatives. Boundary invasion is6-17mm, labelled from
complete rendered bounds, not object centers. Near/far samples are longitudinal
variants, not additional independent groups. All2880 pass visibility, rendered
geometry and no-competing-corridor-surface admission; no frames were removed.

The unchanged nominal45 single-return simulator constructs the same64 distances
for A and B. A uses the frozen strong threshold .4071309640537889 and independent
definite-support bypass. B uses MobileNetV3-small pretrained layers through index3
(10,488 frozen parameters), RGB256x144, nine ordered feature samples per ToF zone,
distance/validity/angle channels and an81,921-parameter convolutional head.
One seed20260920,1200 AdamW updates,batch64,BCE,last checkpoint only. No global RGB
pooling, evaluator masks, scene IDs, geometry, baseline OR or metric-depth output.

Both arms are evaluated current-frame first and then with identical nonrecursive
one-frame hold. Measurement UNKNOWN remains identical, even when B predicts an
alert. Observation construction uses native depth; learned inference does not.

## Prespecified zero-logit diagnostic on new development groups

These are the protocol's **zero-logit diagnostics**, not an admitted operating
point and not held-out test performance. Development Core384=86P/298N;
Boundary192=86P/106N. Counts are TP/FP/FN.

| Scope / readout | A | B | A precision | B precision | A FPR | B FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Core current |85/4/1|83/15/3|95.51%|84.69%|1.34%|5.03%|
| Boundary current |5/3/81|78/3/8|62.50%|96.30%|2.83%|2.83%|
| Core + hold |85/12/1|84/27/2|87.63%|75.68%|4.03%|9.06%|
| Boundary + hold |7/5/79|80/9/6|58.33%|89.89%|4.72%|8.49%|

Boundary current recall rises5.81% to90.70% at three false-positive frames in
both arms; event detection rises2/8 to8/8. This is a substantial scoped signal
in the existing RGB/single-return input package. It is already present BEFORE
hold, so it cannot be attributed just to reminder persistence.

But Core current adds11FP and increases FN1 to3. All8 Core events remain
detected, yet two BODY events start0.2s later; one old HEAD onset improves by0.2s.
B loses3 specific A Core true frames and recovers1 different Core frame. Thus
equal event count and modest aggregate recall change conceal actual regressions.
All11 extra current CoreFP on HEAD occur in two OUTSIDE layouts (8 hanging-plane,
3 horizontal-bar FP). BODY contributes four CoreFP across three layouts.

| Development burden | A current | B current | A hold | B hold |
| --- | ---: | ---: | ---: | ---: |
| Core FP segments |4|7|10|13|
| Core FP sampled seconds |0.8|3.0|2.4|5.4|
| Boundary FP segments |3|3|3|8|
| Boundary FP sampled seconds |0.6|0.6|1.0|1.8|

Boundary B's maximum first in-event delay is0.4s. Core and Boundary each retain
their full negative exposure. UNKNOWN is328/384 Core and192/192 Boundary for
every arm; silence does not become known clearance. FPR uses all labelled
negative frames, not only known-clear TN. IoU is inapplicable to scalar alerts.

Across all strata, BODY current is A49/6/39 versus B81/6/7; HEAD is A41/1/43
versus B80/12/4. These pooled layer numbers do not replace the Core/Boundary table.

## Why this is not just an unfortunate chosen threshold or hold rule

All576 distinct development scores plus one above maximum were evaluated with
atomic ties, without test labels. **0/577 thresholds** satisfy all fixed rules.
There is no admitted threshold, and test prediction is not activated.

The following are descriptions of the already computed curve, not replacement
thresholds or additional candidate runs:

- Retaining every A Core true frame requires logit threshold at most
  -17.159542083740234. Even its least-FP endpoint gives Core86/70/0 versus A85/4/1;
  Boundary85/15/1 versus A5/3/81. Hold CoreFP84 versus12, BoundaryFP23 versus5.
- All145 thresholds satisfying every FP-frame/segment budget lose Core true
  frames. The maximum Core current TP among them is77 versus85 (held82 versus85).
- Even ignoring the held-output constraints, **zero** thresholds meet current
  Core retention and current Core/Boundary FP/segment budgets. Hold is therefore
  not the sole cause of rejection.

On training frames, B at zero logit gives524/0/0 overall (1728/1728 labels correct),
including262/0/0 each on Core and Boundary. Training fit is adequate on these
samples; the failure concerns different-layout Core errors and their absolute
scores, not an inability to optimize this particular training set. This does not
identify a single causal explanation or establish that more data would fix it.

Among76 newly recovered Boundary positives at zero logit,50 have native target
corridor contributors in the observed winning return and26 do not;3 old Boundary
TPs are lost. The one recovered Core frame lacks such returned support. These are
model classifications, never invented metric foreground measurements. Native
support is evaluator-only context, not an additional model input or success gate.

![Frozen fit and development threshold diagnostics](../../../../artifacts.local/work/ba-spatial-bce-20260920/development-evidence.png)

## Evidence boundary, checks and execution

The split has only8 independent development groups, sharing Willow, primitive
families, renderer and the hypothetical sensor law with training. Geometry
non-overlap is not natural-scene or hardware independence. The complete B package
changes representation and training/data relative to A; no no-RGB control isolates
RGB's contribution, and no ranking control or ranking benefit is claimed.

The8 test groups were captured and source-admitted, and their observation-only
frozen features were cached. **No test logits, prediction start, model metrics or
test-based selection exist.** Source admission inspected geometry on all splits,
so this is explicitly Development and not a protected-blind evidence claim.

13 focused unit checks passed: group separation and paired geometry, calibrated
sampling order, validity masks, frozen encoder, gradients/checkpoint roundtrip,
atomic thresholds, silence rejection, UNKNOWN, and nonrecursive causal hold.
Independent audit PASS verifies every RGB/native hash, rendered-bound labels,
split/train indices and labels, same-range A replay, the full577-threshold curve,
frame/segment/event counts and absence of test inference. It does not run the
primary selection/evaluation functions or retrain the model. The plot was rendered
and visually checked. Current-only outcome diagnosis used the saved curve.

UE capture1793.532s; all task actors and the owned process tree released. CPU
scalar/source construction122.000s (TASK_NOT_GPU_SUITABLE). Measured same-batch
encoder CPU/GPU medians48.49/2.57ms and head training-step2.47/1.55ms selected CUDA
on RTX5060 Laptop. Feature extraction19.52s; one head fit including save/receipt
about1.97s. These are offline batch/workload timings, NOT phone/edge end-to-end
latency or a device feasibility claim.

Evidence is retained under `artifacts.local/work/ba-spatial-bce-20260920/`:
source RGB/native geometry, single-return observations, private lineage, split
labels, frozen spec/protocol, feature cache, checkpoint, training/backend receipts,
train/dev logits, full threshold curve, diagnostics, audit, plots and logs.
Approximately4.6GiB is retained for reproducibility, including the unconsumed
test observations; it is not an active/resume job. No capture/training process
remains. Small task-owned Python bytecode is removed separately at delivery.

Supported global registration still fails on pre-existing ledger303 fingerprint;
terminal inheritance assignment is attempted separately and its receipt retained.
The [local disposition](SPATIAL_BCE_DISPOSITION_20260920.json) records the closed
scope and metadata gap without manually changing the ledger. The frozen recipe
ends here: no new fit, loss, threshold relaxation, cohort, runtime or demo update.
