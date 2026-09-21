# Regional ToF depth learning: runnable, no useful Boundary rescue

2026-09-21. **NOT_SUPPORTED / both exact recipes NEGATIVE_CONTROL.** One
matched algorithm experiment completed: the REGION supplement adds no true
alerts and one Core false frame on the consumed1152-frame transfer. POINT adds
five current TP, but violates the Core current false-frame/segment budget and
retains too little Boundary coverage. Keep A and every prior B/R/U/G disposition.
No further fit, threshold, loss, representation or backbone retry follows.

[Frozen protocol](REGIONAL_DEPTH_PROTOCOL_20260921.md),
[model](regional_depth_model.py), [runner](run_regional_depth.py),
[observation-only inference](regional_depth_inference.py),
[data and supervision separation](regional_depth_data.py).
Evidence: `artifacts.local/work/ba-regional-depth-20260921/`.

## What was implemented and compared

Both106,537-parameter networks predict81 depth-class probabilities at each of
64x64 local image samples. They use raw RGB within the actual ToF footprint,
public64 axial-Z ranges/validity and camera rays. Native visible depth supplies
TRAIN-only supervision; missing depth is masked, finite>=8m has an overflow
class. Predicted geometry is an inference, not a measured surface certificate.

The sole matched arm difference is the measurement-consistency operator:
REGION pools predicted depth mass over64 samples per zone; POINT uses one
central sample. Both apply inverse-square energy weighting and a soft dominant
bin constrained by the original public range interval. This differentiable
surrogate is not the exact sensor likelihood or an observed depth histogram.
Training also uses identical dense-depth CE and frame BCE. The same top16
predicted corridor probabilities supply the current score; no scalar alert head.

One1200-step AdamW fit per arm used the same initialization and sampled batches,
batch16, seed20260921, lr.001,wd.0001 and the final checkpoint. Training covers
24layouts/1728frames, selection8/576, transfer16/1152. Groups are disjoint, but
transfer previously informed the hypothesis: these are reused controlled
Development results, not independent confirmation. No capture or protected test.

## Alert effects at dev-selected cost budgets

Each output is A OR the new current score, followed by unchanged .2s one-frame
hold. All existing A alerts and UNKNOWN values are preserved. The cutoff is
selected solely on dev; POINT1.5939961672 and REGION2.9149248600. Identical budget
rules do not mean identical realized FP counts or numerical score thresholds.

Counts below are TP / FP / FN on complete clips, including entry/exit negatives.

| Readout | Core current | Core hold | Boundary current | Boundary hold |
| --- | ---: | ---: | ---: | ---: |
| A |171 /17 /2|173 /33 /0|10 /1 /163|14 /5 /159|
| Prior B |173 /36 /0|173 /58 /0|128 /10 /45|140 /19 /33|
| Prior R |171 /20 /2|173 /38 /0|62 /1 /111|78 /6 /95|
| Prior U |173 /24 /0|173 /44 /0|58 /4 /115|66 /11 /107|
| Prior G |173 /25 /0|173 /45 /0|59 /3 /114|68 /9 /105|
| POINT |172 /23 /1|173 /43 /0|14 /1 /159|22 /5 /151|
| REGION |171 /18 /2|173 /34 /0|10 /1 /163|14 /5 /159|

| Held metric | A | POINT | REGION |
| --- | ---: | ---: | ---: |
| Core events |16/16|16/16|16/16|
| Core precision / recall / FPR |83.98% /100% /5.55%|80.09% /100% /7.23%|83.57% /100% /5.71%|
| Core false segments / sampled seconds |24 /6.6|28 /8.6|24 /6.8|
| Boundary events |4/16|6/16|4/16|
| Boundary precision / recall / FPR |73.68% /8.09% /2.37%|81.48% /12.72% /2.37%|73.68% /8.09% /2.37%|
| Boundary false segments / sampled seconds |5 /1.0|5 /1.0|5 /1.0|
| Boundary gain layouts over A |--|2/16|0/16|
| HEAD-horizontal TP / positive frames; events |0/43;0/4|2/43;1/4|0/43;0/4|

Core first-alert times are unchanged in both new arms. POINT's two newly detected
Boundary events first alert at .6s (BODY protruding plane) and1.2s (HEAD horizontal)
after entry. Its other ten Boundary events are wholly missed; REGION misses12.
The two POINT gain layouts are body_protruding_plane_g00 and head_horizontal_g03.
Of its five new current TP, one lacks a returned native target-corridor
contributor; this remains learned inference, not recovered measured support.
REGION has no new current TP in either stratum.

REGION passes all added-cost caps but misses the50% Boundary recall floor and
8/16-layout breadth criterion. POINT adds6 Core current FP (cap5) and5 false
segments (cap4), while Boundary recall reaches only12.72%. Thus the result is
not a narrow budget miss alone. REGION has8 fewer held Boundary TP than POINT
and improves no layout; the regional contribution criterion also fails.

## Local geometry and its limits

Transfer native payloads were loaded only after predictions were sealed. These
metrics describe sampled visible rays, not whole objects or all positive events.
The shared population has4,532,571 known rays,186,021 missing rays and10,954
positive corridor rays. Missing rays are excluded rather than labeled negative.

| Local metric | POINT | REGION |
| --- | ---: | ---: |
| TP / FP / FN at probability.5 |7254 /7791 /3700|6696 /7090 /4258|
| Precision / recall |48.22% /66.22%|48.57% /61.13%|
| IoU |38.70%|37.11%|
| Depth-class accuracy |54.64%|58.46%|
| Expected-depth MAE, finite true depth<8m |.17284m|.17536m|

MAE uses3,779,238 rays and is scene-wide, not a Boundary-specific error. REGION's
higher bin accuracy does not yield better intrusion IoU or alert rescue. These
results reject this exact supervised recipe/measurement surrogate/readout at the
frozen budget. They neither prove regional measurement modeling intrinsically
inferior nor reopen the [closed whole-object oracle](IDEAL_ASSOCIATION_RESULTS_20260921.md).

## Verification, runtime and reproduction

Ten focused data/model checks pass, covering axial rays, missing/far labels,
off-center regional returns, excluded ineligible near bins and finite gradients.
Pre-freeze checks caught and fixed a PyTorch layer-name collision, the inclusion
of ineligible depth bin0, and overly broad source-seal traversal. Source validation
now pins seals and checks only explicitly permitted members; protected test
labels and unused all-split RGB feature caches are recorded as unaccessed.

Actual GPU: RTX5060 Laptop, Torch2.11.0+cu130. Same-workload full-loss median
training probe was213.55ms CPU versus15.09ms CUDA per batch16. The two scientific
fits took17.24s and17.43s. These short host timings exclude preparation and are
not phone or sensor latency. Batched transfer inference including output maps
took.516s/.415s for1152 frames, excluding RGB decode.

Observation-only replay passes on64 fixed frames per arm across all16 layouts,
including RGB decode and resident-model execution. Every supplement decision
matches the sealed batch result. Maximum probability difference is7.73e-7 and
maximum local-map difference3.43e-6. An initial logit-only2e-5 tolerance failed at
6.49e-5 because near-saturated logits amplify small batch-size FP32 differences;
the retained receipt reports this without changing model, threshold or prediction.
Probability agreement and exact frozen decisions govern the completed check.
POINT/REGION mean host time9.24/8.25ms, p9514.08/14.79ms, excludes sensor transport,
A/hold execution and Android. No incumbent-speedup claim.

Run stages with the existing Torch research environment:

```text
python research/active/dtr-r0/nearfield/run_regional_depth.py <stage>
```

Stages are freeze, prepare, fit, select, predict, evaluate, in order. The existing
run is immutable; these commands are not authorization to repeat its fits.
For saved-model use, call `regional_depth_inference.load(checkpoint, device)`
then `predict(head, rgb_uint8, public_ranges, public_boxes)`. Checkpoints are
`POINT-head_last.pt` and `REGION-head_last.pt` in the evidence directory. The CLI
accepts --checkpoint, --rgb, --ranges (64-vector npy), --boxes (64x4 npy),
--device and --output. It emits predicted depth/probability and raw supplement
score; A and the unchanged hold remain separate, as in the experiment runner.

Local structured inheritance marks both exact recipes NEGATIVE_CONTROL. Keep
their runnable implementation/checkpoints as evidence, not a promoted method.
Independent audit PASS:83 file entries across frozen dependencies and six stage
seals, matched initialization/batches/checkpoints, exact prior flags/UNKNOWN,
independent counts/costs/segments/event delays and1152 map shapes. Geometry
arithmetic and native-reference hashes were checked; the independent reviewer
did not regenerate labels or reread native payloads. The main evaluation verified
all1152 native payload hashes. Audit script and receipt are retained.

Supported registration still fails at existing `experiments/index.jsonl:303`
input-fingerprint mismatch; supported inheritance reports unknown terminal.
Both command receipts and the local disposition are retained without editing
the global ledger. All task-owned execution processes have exited; checkpoints,
prepared inputs and diagnostic evidence remain for reproduction. No paid worker,
resident service, App/default change or automatic successor remains.
