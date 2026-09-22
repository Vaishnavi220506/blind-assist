# Corridor-relative representation: real low-cost gain, insufficient coverage

2026-09-21. **NO_GO under the frozen acceptance rule.** Boundary hold improves
from 14/173 to 78/173 positive frames and 4/16 to 11/16 events, while Core adds
5 false frames and Boundary adds 1. Every added-cost cap passes and 9/16 Boundary
layouts gain, but 45.09% recall misses the prespecified 50% floor. All four
HEAD horizontal layouts remain wholly missed. Preserve this exact recipe as
`NEGATIVE_CONTROL`; no threshold/loss/feature retry or automatic successor.
The result rejects promotion of this package, not the existence of useful RGB
information or all corridor-relative representations. A/demo stays unchanged.

[Frozen protocol](CORRIDOR_RELATIVE_PROTOCOL_20260921.md) /
[runner](run_corridor_relative.py) / [observation-only model](corridor_relative_model.py).
Artifacts: `artifacts.local/work/ba-corridor-relative-20260921/`.

## What changed and what was actually evaluated

One shared 60->32->32->1 MLP (3,041 parameters), fixed max over valid zones;
native RGB statistics in three interval-conditioned corridor-relative query bands,
with full original interval endpoints and depth fraction. No absolute zone index
or flattened position head. This is the complete new representation package;
resolution, alignment, hand-crafted statistics and aggregation are not separately
attributed. Old Spatial-BCE already had shared convolutions, and historical
pixel-query already used native RGB plus corridor queries.

Training: original 24 layouts / 1,728 frames. Selection: original 8 dev layouts /
576 frames. Evaluation: all 16 already consumed complement-transfer layouts /
1,152 frames, 48 complete clips. Core has 173 positive / 595 negative frames;
Boundary has 173 positive / 211 negative frames. Original protected test was not
opened. No new capture or physical hardware was used. This is reused controlled
simulation Development, not independent confirmation or safety evidence.

One ordinary-BCE fit, seed20260921, 1,200 updates, batch64, lr.001, wd.0001, final
checkpoint only. Dev selected the single inclusive threshold **5.128307342529297**
under the frozen added-FP and segment budgets. It was applied unchanged to transfer.
The 577 dev score breakpoints are calibration of one fit, not model/recipe retries.
R is A OR the new score; original nonrecursive .2s hold is unchanged. Therefore
R cannot remove A's existing false alerts. Every A alert and UNKNOWN is preserved.

## Paired results

A is the frozen geometric Core. B is the old Spatial-BCE supplement with its
already frozen 7.6612162590026855 cutoff. R is this pilot. B and R have different
predeclared selection contracts: the table is an operating-point comparison,
not a matched-cost proof that either representation dominates the other.

| Region / readout | A TP/FP/FN | B TP/FP/FN | R TP/FP/FN |
|---|---:|---:|---:|
| Core current | 171/17/2 | 173/36/0 | 171/20/2 |
| Core hold | 173/33/0 | 173/58/0 | 173/38/0 |
| Boundary current | 10/1/163 | 128/10/45 | 62/1/111 |
| Boundary hold | 14/5/159 | 140/19/33 | 78/6/95 |

| Region, hold | Arm | Recall | Precision | FPR | Events | False segments | False sampled seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Core | A | 100% | 83.98% | 5.55% | 16/16 | 24 | 6.6 |
| Core | B | 100% | 74.89% | 9.75% | 16/16 | 30 | 11.6 |
| Core | R | 100% | 81.99% | 6.39% | 16/16 | 25 | 7.6 |
| Boundary | A | 8.09% | 73.68% | 2.37% | 4/16 | 5 | 1.0 |
| Boundary | B | 80.92% | 88.05% | 9.00% | 15/16 | 13 | 3.8 |
| Boundary | R | 45.09% | 92.86% | 2.84% | 11/16 | 6 | 1.2 |

R current Boundary precision98.41%, FPR0.47%, recall35.84%; current Core
precision89.53%, FPR3.36%, recall98.84%. UNKNOWN silence is not a true negative;
all positive and negative denominators are retained.

All 52 new current true positives are Boundary; 3 have zero native returned
target-corridor contributors in the sealed source admission audit. Their alerts
are learned classifications, not independently measured obstacle range. RGB query
bands do not assert surface ownership or shrink any original ToF support.
This is an alert classifier, with no output object mask or localization-IoU claim.

## Frozen costs and event limitations

| Readout | Added FP / cap | Added false segments / cap |
|---|---:|---:|
| Core current | 3 / 5 | 2 / 4 |
| Core hold | 5 / 11 | 1 / 4 |
| Boundary current | 0 / 2 | 0 / 2 |
| Boundary hold | 1 / 4 | 1 / 2 |

All five added Core hold FPs are OUTSIDE frames in the HEAD hanging-plane g01
layout. The three current FPs likewise lie there; they become one held false
segment. Core positive coverage remains100%, with zero in-event onset delay and
no added pre-entry/post-exit false frames in its INSIDE events. Total Core
pre-entry/post-exit false frames remain9/24; the added cost is wholly OUTSIDE.

R detects11 Boundary events, misses5 completely. Among detected events, first
in-event delays are 0,0,0,.2,.2,.2,.2,.4,.4,.6,1.8s. Median.2s; maximum1.8s does
not include the five wholly missed events. Initial/internal/terminal silent
positive frames are73/17/5 (95 total). Boundary pre-entry/post-exit false frames
are1/5 versus A1/4; no right-censored releases. Detecting an event does not imply
continuous or timely coverage.

| Boundary family | Layouts | Positive frames | A hold TP | R hold TP | B hold TP |
|---|---:|---:|---:|---:|---:|
| BODY protruding plane | 4 | 45 | 3 | 31 | 33 |
| BODY suspended solid | 4 | 44 | 11 | 18 | 44 |
| HEAD hanging plane | 4 | 41 | 0 | 29 | 26 |
| HEAD horizontal | 4 | 43 | 0 | 0 | 37 |

Gains occur in all four BODY protruding-plane layouts, one suspended-solid layout,
and all four HEAD hanging-plane layouts. Whole-event misses are suspended-solid
g02 and all four HEAD-horizontal layouts. The visible weakness is geometry-family
coverage, not merely one bad cutoff. No causal claim that band pooling erased the
horizontal signal is established without a separately designed contrast.
R does detect HEAD hanging-plane g01, the one Boundary event missed by old B,
with5/9 coverage and.6s first-alert delay. Its own1.8s delay occurs in BODY
protruding-plane g00, with only2/12 positive frames covered.
Full layout/BODY/HEAD/background metrics and exact event records are saved in
`metrics.json`, `group-metrics.json` and `frame-results.json`.

## Execution, validation and disposition

Feature extraction took94.94s for3,456 train/dev/transfer frames, including image
IO and hash checks; it is an offline ragged-statistics CPU workload. Equivalent
batch64 training probes measured CPU1.149ms versus CUDA1.946ms; inference CPU.188ms
versus CUDA.438ms. Both selected CPU with `CPU_FASTER_MEASURED`. The actual 1,200
update fit took1.48s. These are host batch measurements, not end-to-end device or
hardware latency. No accelerator performance claim is made.

Five focused tests pass: nested query-band geometry/partition, masked no-return
behavior, valid scores below-20, numerical permutation invariance, and held-exit /
float64-disabled-cutoff semantics (covered across five test methods). Pre-run
review fixed invalid-zone voting and float64-cutoff comparison before freeze.
No model, protocol or selection change followed outcomes.

[Independent verifier](verify_corridor_relative.py) passes321,533 assertions:
five seals, source/code hashes,3,456 observation-feature rows, every577 dev
candidate, all1,152 transfer predictions, UNKNOWN/hold decisions, subgroup
counts and complete event metrics. Its receipt is `independent-verification.json`.
The verifier independently reconstructs readouts and metrics; it does not rerun
training or independently reproduce every image-pixel statistic. Band projection
and partition have the separate synthetic geometry test described above.

Local structured terminal: `local-inheritance.json`, role `NEGATIVE_CONTROL` for
this exact package and promotion question. Useful recorded low-cost gains remain
available as evidence, but the model does not become the retained Core/challenger.
The supported global registration command remains blocked at existing
`experiments/index.jsonl:303 input_fingerprint` mismatch; inheritance reports
unknown terminal. Both receipts are preserved; the global ledger was not edited.
The protected test, A/demo and every earlier frozen negative remain unchanged.
