# BA-Depth Probe: frozen real RGB-ToF near-surface localization

**DEPTHOR improves mixed-zone IoU but loses near-surface recall.** Retain the
descriptive localization gain; do not replace raw support or promote an alert
method. No training, threshold changes, DELTAR, Radar integration or new capture.

## Frozen scope

160 ZJU-L5 test frames: 20 evenly spaced lexical entries from each of eight
scenes. This is disclosed public Development, overlapping the earlier smoke,
not fresh confirmation or the full benchmark. Predictions were sealed before
the separate evaluator read reference depth. Inputs, checkpoints, protocol and
all 480 prediction files have hashes in
`artifacts.local/work/ba-depth-probe-20260918/`.

A: valid ToF distances expanded over clipped official `fr` rectangles; overlapping
rectangles take nearest distance. Missing returns stay UNKNOWN. This deliberately
simple conservative baseline is not MZ129 or a physical echo-extent guarantee.
B: official pretrained UniDepthV2 ViT-S, RGB-only metric inference with estimated
camera and no reference/ToF scale fit. DEPTHOR's embedded MDE is relative depth,
so it cannot independently serve as a metric threshold baseline. B uses a
different backbone; B/C is not a controlled fusion-only architectural ablation.
C: original DEPTHOR-ZJU-Small, unchanged official sparse projection, previously
tested unfold CSPN execution port. D: RealSense reference identity, a tautological
scoring ceiling rather than independent truth validation.

Primary region: union of projected zones with reference P20<2m and P80-P20>1m,
at least 80% valid reference and 64 valid pixels, intersected with raw-known
pixels. Fixed primary threshold 2m; secondary thresholds 1/1.5/2/3m. P20 selection
can miss very thin foreground occupying <20% of a zone. It does not establish
2--5cm rod or head-obstacle capability. RealSense edges may contain errors.

## Primary result

74/160 frames contain 246 mixed zones; the common union has 366,611 pixels.
All primary raw/DEPTHOR pixels are evaluable; mono abstains on 13 negative pixels.

| Method | Precision | Recall | IoU | Pixel FP | Pixel FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw zone | 64.30% | 97.32% | 63.18% | 126,893 | 6,284 |
| RGB metric mono | 78.25% | 91.04% | 72.65% | 59,443 | 21,044 |
| DEPTHOR | 78.05% | 95.69% | 75.40% | 63,181 | 10,125 |
| Reference identity | 100% | 100% | 100% | 0 | 0 |

DEPTHOR reduces pixel FP by 50.21%, improves IoU by 12.22 percentage points,
but has a net increase of 3,841 missed near pixels and loses 1.64 points recall. IoU improves in
7/8 scenes against raw (dorm worsens), but beats mono in only 4/8 scenes.
The predeclared joint condition (higher IoU, noninferior recall, majority scene
IoU wins) fails on recall. No statistical significance or independent pixel
sample-size claim is made.

## Secondary evidence and coverage

| Common raw-known domain IoU | <1m | <1.5m | <2m | <3m |
| --- | ---: | ---: | ---: | ---: |
| Raw | 83.18% | 90.52% | 97.16% | 99.50% |
| Mono | 52.87% | 71.87% | 89.16% | 99.23% |
| DEPTHOR | 86.31% | 92.16% | 97.22% | 99.47% |

DEPTHOR has lower recall than raw at all four common-domain thresholds.
Mixed-zone gains therefore do not establish uniform scene-level superiority.

Boundary bands use >0.5m jumps between adjacent valid reference pixels, dilated
5/10 pixels; missing reference edges are excluded. On the common domain,
raw/mono/DEPTHOR MAE is 1.139/0.842/0.893m at 5px and
0.924/0.665/0.686m at 10px. Mono coverage is 99.52%, others 100%.
DEPTHOR improves on raw but does not beat mono here; this is not paper EWMAE.

Across the full reference-valid image raw covers only 52.36%, mono 99.34%,
DEPTHOR 100%. At 2m the fraction of all reference near pixels detected is
57.78%/91.17%/97.08%. Raw UNKNOWN is counted separately, never silently scored
as far or removed from the coverage denominator. Different-domain IoUs must
not be compared as equal coverage.

## Corridor and interpretation boundary

**BODY/HEAD corridor TP/FP/FN is NOT_EVALUABLE.** Inspected HDF5 samples contain
only rgb/depth/fr/hist_data/mask and no body pose, camera-body transform or
confirmed intrinsic calibration. No image rectangle is invented as a body
corridor. D against itself would not independently validate corridor truth.
No per-frame alert, native sensor contributor retention, event, phone latency,
hardware safety, outdoor transfer or precise thin-object claim follows.

MZ140 direct-transfer and MZ141/MZ142 fixed fine-tuning failures remain intact.
This real-data localization component result neither reverses them nor authorizes
automatic retraining with a new loss. A future geometry/alert bridge needs
adequate calibrated pose/reference data and separate retention evaluation.

## Execution and retention

CUDA RTX 5060 Laptop; strict original model loading; no reference enters model
inputs or mono scale. Runtime and per-frame timings are in runtime.json; scalar
mask scoring runs on CPU. All 160 frames completed for both learned arms.
Three focused tests cover UNKNOWN, clipped regions, overlapping mixed unions,
invalid reference edges and reference identity. Prediction hashes verified at
evaluation. Subagent audit unavailable (service 503); no independent review claimed.

Retain source, protocol, input subset, compressed predictions, per-frame counts,
runtime, logs and summary under the artifact directory. Existing official archive
and pretrained models are reused in their original directories. No owned GPU
job, paid allocation or background process remains. No disposable data expansion
or cleanup outside task-owned output occurs.

Registration using valid identifiers still fails at existing index.jsonl:303
input_fingerprint mismatch. Intended terminal BA_DEPTH_PROBE_ZJU_20260918,
NEGATIVE_CONTROL for the joint IoU-plus-recall replacement role, retains partial
localization evidence. Structured registration/inheritance is pending; no ledger
rewrite or bypass. Scientific result and metadata gap are separate.

Reproduction: run ba_depth_probe.py prepare/predict/evaluate with a NEW --output
directory using the recorded GPU runtime and existing frozen assets. Never
overwrite the completed directory. `protocol.json` fixes selection and criteria
before inference; `prediction-seal.json` precedes reference evaluation.

Follow-up [support-loss diagnosis](BA_DEPTH_SUPPORT_DIAGNOSTIC_20260918.md):
8,468 newly missed pixels minus 4,627 rescued pixels yields the net 3,841.
All 52 affected near-return zones retain predicted near surface; 48 already
satisfy the proposed Q10 constraint. No automatic existence-only SCDE training.
