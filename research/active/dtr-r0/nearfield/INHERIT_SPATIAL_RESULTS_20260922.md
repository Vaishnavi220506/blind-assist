# Inherited spatial mechanisms: one raw/local HGB result

2026-09-22. EXPLORE, consumed controlled Development. One fit per arm and one
576-frame evaluation under the [frozen protocol](INHERIT_SPATIAL_PROTOCOL_20260922.md).
**LOCAL is a COMPONENT_OR_CHALLENGER; RAW is a NEGATIVE_CONTROL.** Neither
declared A-union candidate meets the alert-system gate. No threshold, feature,
fit, loss, temporal policy or gate was changed after seeing this result.

## Mechanism and fairness

Both classifiers use the same public 320x180 RGB, 64 regional ToF rows, six
fixed camera-frame queries, train/dev/eval split, HGB settings and dev selection
rule. Their first 910 features are identical: ordered regional range/validity,
original interval endpoints, regional RGB summaries, global RGB and query
coordinates. LOCAL fills 51 additional slots with pooled RGB/range summaries
partitioned by the relation between each full regional interval and the query;
RAW leaves those slots zero. Both have 961 columns, but different effective
information. This tests a representation package, not equal informative feature
count, pure geometry causality or proven physical return ownership.

The source has 48 layouts and 1728 frames: 864/24 train, 288/8 dev, 576/16 eval.
All original indices and the six original query boxes are retained. Only train
labels fit the models. Dev selects an inclusive cutoff by atomic score ties,
maximizing added TP with at most +1 FP and +1 false segment relative to frozen A.
RAW chose 0.776778260888226 (dev 105 TP / 4 FP / 23 FN / 4 false segments);
LOCAL chose 0.7959699076137833 (119 / 4 / 9 / 4). Dev A was 98 / 3 / 30 / 3.
Prediction seals preceded this implementation's parsing of evaluation labels;
the evaluation source was already consumed by the earlier query experiment.

Inference accepts public RGB/ToF/query geometry only. Family, layout, clip,
split, time, truth, evaluator geometry, object IDs, native depth and A predictions
are absent from learned features. A is recomputed and parity-checked on all
1728 frames, then OR-combined with the selected model frame score. Regional
intervals are retained whole; a local support band is not a new depth reading.

## Evaluation

There are 256 positive and 320 negative frames, 48 clips and 32 events. FPR uses
all 320 known negative labels. Event duration uses sampled frames at 0.2 s,
not wall latency. Standalone models use the same selected cutoffs and are
descriptive controls; their outputs were not selected as replacement candidates.

| Arm | TP / FP / FN | Recall | Precision | FPR | Events | False segments | False sampled duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| A current | 193 / 6 / 63 | 75.39% | 96.98% | 1.875% | 32/32 | 6 | 1.2 s |
| A one-frame hold | 225 / 26 / 31 | 87.89% | 89.64% | 8.125% | 32/32 | 20 | 5.2 s |
| RAW standalone | 89 / 7 / 167 | 34.77% | 92.71% | 2.188% | 22/32 | 7 | 1.4 s |
| LOCAL standalone | 207 / 4 / 49 | 80.86% | 98.10% | 1.250% | 31/32 | 4 | 0.8 s |
| A OR RAW | 200 / 12 / 56 | 78.13% | 94.34% | 3.750% | 32/32 | 9 | 2.4 s |
| A OR LOCAL | 223 / 8 / 33 | 87.11% | 96.54% | 2.500% | 32/32 | 8 | 1.6 s |

All arms retain the same 487 prediction-UNKNOWN annotations (167 positive,
320 negative); alerts can coexist with ambiguous support. No negative silence
is counted as observed clear space: frame TN=0 throughout. Evaluator truth
coverage is 576/576. The scalar per-query binary confusion is separately saved
and does not confer observed-free-space authority: RAW 366 TP / 50 FP / 514 FN
and LOCAL 715 / 27 / 165 over 3456 queries (880 positive).

LOCAL versus A adds 30 TP, 2 FP and **2 false segments**, with +11.72 percentage
points recall and improvements in 14/16 layouts. The system gate required at
least +10 points, at most +2 FP and **+1 false segment**. It fails the segment
budget. RAW adds only 7 TP, 6 FP and 3 segments, failing that gate as well.

LOCAL versus matched RAW adds net 23 TP (24 gained, one lost), removes 4 FP and
one false segment, and improves 12/16 layouts. Recall increases by 8.98 points;
the predeclared 1000-resample paired whole-layout bootstrap interval is
[5.86, 12.11] points. Both BODY and HEAD recall improve. This meets the frozen
component gate. The bootstrap is conditional on 16 consumed synthetic layouts;
it does not establish fresh-source generalization.

Both unions preserve every A-positive frame and all A events, and never delay
A onsets by construction. LOCAL advances 13 event onsets relative to A. Relative to RAW it
advances 10 but delays one by 0.2 s: `head_hanging_plane_g08_boundary`, losing
its true frame 02. LOCAL standalone misses one whole event, so its apparently
better FP count does not justify a post-outcome change of the selected policy.

| Stratum | A TP / FP / FN | A OR RAW | A OR LOCAL |
|---|---:|---:|---:|
| BODY, 128 positive / 160 negative | 99 / 5 / 29 | 104 / 8 / 24 | 114 / 7 / 14 |
| HEAD, 128 positive / 160 negative | 94 / 1 / 34 | 96 / 4 / 32 | 109 / 1 / 19 |
| BOUNDARY, 128 positive / 64 negative | 71 / 0 / 57 | 78 / 0 / 50 | 100 / 0 / 28 |
| INSIDE, 128 positive / 64 negative | 122 / 1 / 6 | 122 / 1 / 6 | 123 / 1 / 5 |
| OUTSIDE, 0 positive / 192 negative | 0 / 5 / 0 | 0 / 11 / 0 | 0 / 7 / 0 |

Of the 30 additional true frames versus A, 29 are BOUNDARY and one is INSIDE.
The two new false frames are
`query_occupancy_body_suspended_solid_g05_outside_03` and `_09`; they form two
separate false segments. All four family-level counts, all 16 layout groups,
per-query confusion, full event identities and first-alert times are in
`metrics.json`; individual outputs and support states are in `frame-results.json`.

## Cost and verification

CPU is declared `TASK_NOT_GPU_SUITABLE` through the shared research_backend
helper for small tabular HGB and ragged RGB statistics. Actual NumPy 2.4.4 and
sklearn 1.9.0 execution used four CPU threads; the real feature probe took
0.01270 s. The complete scientific command took 43.07 s. Shared feature
extraction took 29.46 s for 1728 frames (p50 16.54 ms, p95 23.53 ms), and A
reference/parity calculation 4.44 s. Fits took RAW 4.73 s and LOCAL 2.62 s.
576-frame inference batches took 59.93/57.69 ms; 20 cached-feature single-frame
samples had p50 1.12/1.21 ms and p95 1.61/1.47 ms. Model sizes are
795771/810516 bytes. These host measurements exclude capture and PNG decode;
they are not endpoint latency or a CPU/GPU comparison.

Four focused unit tests passed before execution, covering original-interval
band boundaries/partition, common/raw feature parity and input nonmutation,
invalid-range handling, and atomic dev ties/A retention. A separate governed
saved-output audit passed 12502 checks across 576 frames and 3456 queries:
source/snapshot/model/feature/prediction hashes, group separation, dev cutoff
selection, all confusion counts, strata, events, bootstrap and terminal gates.
It imports neither fitting nor the runner's metric implementation. It does not
independently reconstruct every RGB statistic or the renderer. No scientific
fit or evaluation was repeated. Both governed commands exited successfully;
no capture, worker or paid allocation was created and the CPU processes ended.

## Evidence and disposition

Governed run specs/logs: `artifacts.local/evidence/ba-inherit-spatial-20260922/`.
Scientific outputs and source snapshots:
`artifacts.local/evidence/ba-inherit-spatial-20260922-run/`.
Independent audit: `artifacts.local/evidence/ba-inherit-spatial-20260922-audit/result.json`.
Runtime journals are under `artifacts.local/evidence/resource-fabric/runs/dtr-inherit-spatial/`
with run IDs `inherit-spatial-20260922-v1` and `inherit-spatial-20260922-audit-v1`.
Existing scoped inputs and automatic output lineage sufficed; no UE policy
scope or source role was changed by this subtask.

| Sealed file | SHA-256 |
|---|---|
| metrics.json | `125c4ffeaa2278d5d044af4e9ce99e14bdb690bd5e5767f9122caafadaba74fb` |
| prediction-seal.json | `0ec1ddaf9bc051ff40787c7c853004b1345ed8c220a9f991290e42749460fef4` |
| frame-results.json | `08fc3fd3e1c655b92e9639459e83cb3802132818c110002e193d7e66862743dd` |
| costs.json | `24a1de2819e53c028f65736b99a768148ce52f285ef7e4738954864d9066f75a` |

Recommended terminal roles are RAW `NEGATIVE_CONTROL` and LOCAL
`COMPONENT_OR_CHALLENGER`, with neither replacing frozen A. The useful inheritance
is deterministic query-conditioned support pooling with a simple learner.
This does not establish physical ownership, native localization, range accuracy,
mask IoU, arbitrary-query behavior, hardware feasibility or safety. Those are
unmeasured here, not failed metrics. Earlier MZ147, R/U/G and paired MobileNet
negative packages remain unchanged. Shared CURRENT/ledger/inheritance publication
belongs to the root agent; this subtask changes only its dedicated files.
