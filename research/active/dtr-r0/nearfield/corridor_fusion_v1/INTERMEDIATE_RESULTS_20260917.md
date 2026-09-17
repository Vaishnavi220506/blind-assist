# E1-I: positioned encoder tokens do not establish an incremental alert benefit

2026-09-17, user-authorized single diagnostic after the
[negative depth-statistics result](E1_RESULTS_20260917.md).
EXPLORE, consumed controlled Development; no new capture or original test48.

**E1_INTERMEDIATE_NO_INCREMENT.** Frozen early/late DA-V2 patch tokens with
TRAIN-only PCA and a matched HGB change the A baseline137TP/28FP/7FN to103/3/41.
F1 falls88.67%to82.40%, recall95.14%to71.53%, and9of30events are entirely missed.
The representation recovers ranking quality relative to scalar depth statistics,
but does not beat A or the four-expert ranking curve. Pause this online DA-V2
feature route after the two tested adapters; do not start E2 from these results.

## Matched report on the same288frames

| Arm | TP/FP/FN | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MZ129 | 136/74/8 | 64.76% | 94.44% | 76.84% | N/E |
| MZ145 original causal | 143/69/1 | 67.45% | 99.31% | 80.34% | .9615 |
| Original four-expert causal mean | 140/36/4 | 79.55% | 97.22% | 87.50% | .9795 |
| A, base features / new frame working point | 137/28/7 | 83.03% | 95.14% | 88.67% | .9615 |
| B, added scalar-depth statistics | 135/32/9 | 80.84% | 93.75% | 86.82% | .9266 |
| C, added positioned encoder features | 103/3/41 | 97.17% | 71.53% | 82.40% | .9570 |

A/B and original references are inherited from authenticated E1 outputs, with
no repeated search or modification. A scores are the old static MZ145 scores;
its working-point change is not a new model capability. C receives the same
three candidate configurations and dev-only selection opportunity. All frames
remain in the denominator; C has182UNKNOWN/nonalerts, not182known-safe frames.

## Representation and selection

Same Small checkpoint/source and actual518x924 input as E1. Read normalized
384-channel patch tokens from blocks2and11, zero-based, on37x66grids. Sample
2x2positions in each of8BODY/HEAD range-plane windows and a4x4whole-image layout,
48positions per layer. Keep positional order instead of globally pooling it.
Exactly36/48positions are in the image on each cohort (75%); the remaining
query positions are outside the view and receive zero values and explicit masks.
All original2485sensor/geometry features remain unchanged.

Fit8PCA channels per layer on valid TRAIN tokens only, without labels. Add
2x48x8+48mask=816features to A. PCA and this fixed sampling layout are new
adapters and can still discard useful information; this is not an exhaustive
test of all intermediate features or evidence of physical edge recovery.
The decoder is not run. The encoder has22,056,576parameters and stays frozen.

TRAIN192/dev48/report288 IDs and scene groups exactly match E1. Report frames
are historically consumed; no fresh independence is restored. The original
test48 remains excluded. Only TRAIN contributes PCA statistics and head fits.
Select C on dev F1, then AP, then configuration order. All3models fit TRAIN at
100% accuracy at their selected dev thresholds, so the observed result is not
an inability to fit these training labels.

Selected C:15leaves/150iterations/L2=3/lr=.05/min_leaf8/seed177017.
Dev24TP/2FP/0FN, F1=.96, threshold=.728388090293355. The secondary
dev-recall>=95% rule picks the same threshold. Its actual report recall71.53%
shows poor transfer of that selected working point; it was not retuned after
report access. The report PR curve is descriptive, not a deployment selection.

## Where gains and losses occur

| Family | A TP/FP/FN | C TP/FP/FN |
| --- | ---: | ---: |
| Near rod / far wall | 36/11/0 | 36/0/0 |
| Shallow boundary pressure | 30/6/6 | 9/2/27 |
| Substantial BODY | 36/1/0 | 36/1/0 |
| Suspended HEAD | 35/10/1 | 22/0/14 |

C's rod stratum is perfect on these72consumed frames, and BODY stays unchanged.
This conditional result remains visible; it does not outweigh the HEAD and
boundary omissions or authorize a posthoc family switch. Ordinary216frames
are94/1/14; boundary72frames9/2/27. Paired score ordering improves136/144to137/144,
but pairs with both decisions correct fall109/144to101/144.

C detects21/30events:19immediately and2with.50s delay. It entirely misses2HEAD
events and7boundary enter/exit events. A detects30/30. False segments fall18to3
and false bins7sto.75s (2.08% of sampled negative time for C). Those reductions
must be read alongside the event losses. All six measured boundary exits are
nonalert at the first negative frame, but several corresponding positive events
never alerted, so zero release delay is not evidence of improved event handling.

C nonalerts contain24native-supported frames/286returned ToF corridor samples,
versus A5frames/57samples. Native Radar lineage remains NOT_EVALUABLE. Input
slots are retained, but their final warning contribution is not preserved.
This is an observed cost, not a universal no-lost-TP gate.

## Runtime and checks

Same local RTX5060LaptopGPU and Torch2.11.0+cu130, float32. An actual encoder
probe measured CPU1253ms versus CUDA50.14ms and selected CUDA. Complete warm
latency on the first full episode per family (24frames, outcomes not used to
select them) is:

| Arm measured this run | p50 | p95 |
| --- | ---: | ---: |
| A | 81.57ms | 103.86ms |
| C | 155.20ms | 282.66ms |

Includes RGB decode, original frontend, token preprocessing/encoder/sampling,
PCA and HGB; C conservatively also includes A's small head call. No capture,
transport or event wait. B's prior-run p50=303.80ms is context, not a simultaneous
benchmark. Peak allocated CUDA memory433,968,128bytes (~414MiB); the full
checkpoint is loaded but its decoder is not executed. No phone/board claim.

The successful bounded run completes in86.56s, including feature preparation
60.51s and3head fits19.17s. An initial pre-model startup failed because the
parent receipt uses a resolved F: junction path while the new code used its E:
alias. Comparison now resolves both paths and retains the same checkpoint hash;
the failure receipt is preserved. No scientific fit/prediction preceded that fix.

Two focused tests pass for patch-center sampling and zero/mask preservation
after PCA. Independent read-only review finds no label/PCA leakage or asymmetric
selection. All24online replay vectors match the sealed features within1e-5 and
scores within1e-12. Saved-output audit confirms group isolation, TRAIN-only PCA,
dimensions, IDs, scores/thresholds/counts/families and hashes. All24video frames
decode and the preview was visually inspected.

## Disposition and artifacts

Intended inheritance: NEGATIVE_CONTROL for this positioned-token/PCA/HGB recipe.
Pause both tested DA-V2 online-feature adapters; retain A's Development working
point and the original four-expert reference, with their different recall/cost
tradeoffs. MZ129/App defaults stay unchanged. No PCA-dimension, layer, seed,
threshold or family-selection sweep follows; no E2–E6 work was started.

Artifacts: `artifacts.local/work/corridor-depth-intermediate-20260917/` contains
the PCA/head models, resumable token caches, feature/selection/prediction seals,
per-frame CSV, full curves, native accounting, latency samples, comparison video,
source snapshot, failure/completion and delivery-audit receipts. These durable
outputs are retained for reproducibility; the local process and GPU allocation
have exited. Registration remains blocked by existing ledger303 fingerprint
error; inheritance consequently remains pending, without ledger bypass.
