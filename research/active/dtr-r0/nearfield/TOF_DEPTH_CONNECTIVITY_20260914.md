# Depth connectivity: valid background bridges separate on the frozen panel

## Decision and attribution

Retain **zone-level depth connectivity as a Development component**. On the
same 60 consumed artificial packets, all 22 eligible known-depth background
bridges are cut and all 117 observed near-anchor zones are preserved. Absolute
and relative gates tie. Use the absolute gate as the simpler experimental
component; there is no evidence that its range scaling is better. The frozen
equal angular readout, its closed status-weight comparator, and MZ116 remain
unchanged. No training, capture, weight sweep, RGB fusion or successor was run.

The strongest supported claim is conditional: **observed valid near/far depth
discontinuities prevent a far background from joining near angular support**.
This does not establish physical echo recovery, arbitrary surface segmentation,
within-zone object identity or a validated sensor/device implementation.

## Three arms, identical inputs and two explicit output views

[Frozen protocol](tof_depth_connectivity_protocol_20260914.json),
[implementation](tof_depth_connectivity.py), and
[runner](run_tof_depth_connectivity.py).

The original packets, evaluator labels and predictions are hash-bound; inputs
are read directly, not regenerated. All 60 predictions for all arms are written
before opening evaluator footprints or previous outputs. The frozen baseline
reproduces every original equal-weight output exactly.

- Baseline: unchanged four-neighbor angular connectivity and equal summaries.
- Absolute: link only if both valid depths are known and difference <0.30 m.
- Relative: link only if difference <0.15 m + 0.10 times the smaller depth.

Each observed zone stays one node. Its linking depth is the minimum positive
finite `SIM_VALID` slot, matching the old per-zone range-summary definition.
Every original slot remains in the output. A merged-only zone has UNKNOWN
depth and stays an isolated angular node; its nominal 3.6 m is never a far-only
certificate. Two resolved returns do not become two within-zone nodes here.
Local threshold connectivity can still chain gradual depth changes; a component
is not guaranteed to be one physical object or one globally bounded depth layer.

Breaking a bridge also exposes a separate far-wall CENTER region. Therefore
raw all-support direction output and near-obstacle notices are different views:

1. **Raw:** the original all-region scorer, including far background directions.
2. **Matched near view:** the same post-component selector on all three arms.
   Keep a region with any valid depth <=2 m or any merged/unknown support;
   defer only valid-only all-far regions, preserving them in raw output and
   `deferred_far_regions`. This is a notice selection policy, not clearance.

The 2 m boundary and both edge thresholds were fixed before this execution.
They are simple engineering settings, not calibrated sensor confidence limits.
The panel was already consumed and designed around near/far extremes; no
independent confirmation or statistical population claim is made.

## Results

| View and arm | Exact direction /60 | False CENTER frames | Bilateral merge frames | Known-depth bridges cut /22 |
| --- | ---: | ---: | ---: | ---: |
| Original raw baseline | 19 | 36 | 12 | 0 |
| Raw absolute gate | 17 | 38 | 0 | 22 |
| Raw relative gate | 17 | 38 | 0 | 22 |
| Matched near-view baseline | 21 | 34 | 12 | 0 |
| Matched near-view absolute gate | **45** | **10** | **0** | **22** |
| Matched near-view relative gate | **45** | **10** | **0** | **22** |

Do not attribute 19-to-45 correctness solely to connectivity: the common notice
selector alone moves the baseline 19-to-21. With that selector held identical,
depth connectivity improves 21-to-45 (35% to75%). Without the selector, raw
correctness falls to17 and false CENTER rises to38 because far-wall components
remain visible. Thus **edge gating alone does not improve the historical
all-return direction metric**, although it removes topology merging.

Matched near-view breakdown (the two depth gates are identical):

| Family | Baseline correct /10 | Depth correct /10 | Baseline/depth false CENTER | Baseline/depth merges |
| --- | ---: | ---: | ---: | ---: |
| Near obstacle + far background | 0 | 10 | 10 / 0 | 0 / 0 |
| Sparse reliable side + weak merged side | 0 | 0 | 10 / 10 | 10 / 0 |
| Same-zone pole + wall | 0 | 10 | 10 / 0 | 0 / 0 |
| One-side dropout | 5 | 5 | 0 / 0 | 0 / 0 |
| Bilateral near obstacles | 6 | 10 | 4 / 0 | 2 / 0 |
| Artificial RGB conflict control | 10 | 10 | 0 / 0 | 0 / 0 |

## What is and is not recovered

Known-depth bridge opportunities require a baseline-connected pair consisting
of an evaluator near-anchor zone with observed valid depth <=2 m and an
outside-footprint, valid-only far zone >2 m. There are **22 frames /126 pairs**:
10 background frames, six resolved same-zone frames, six bilateral frames.
Success requires every such pair to separate, all observed near anchors to
survive selected output, every far endpoint to survive raw output, and no
remaining bilateral merge. Deleting a background/near node or returning empty
UNKNOWN cannot earn success.

There are 36 frames with angularly connected near/background support. The
remaining **14 lack that known-depth opportunity**: ten weak merged-side frames
and four merged-only pole frames. They are not counted as successful metric
depth recovery. Isolating uncertain nodes also eliminates ten weak-side
bilateral merges, but leaves the same ten nuisance CENTER frames. The other
two bilateral merges are removed across known depth discontinuities.

All three arms retain **117/117 observed near-anchor zones**, including
**83/83 valid-near anchors**. The evaluator specifies122 anchors; five absent
dropout returns stay missing. Each arm preserves124 unresolved selected zone
observations across the panel. Whole-frame UNKNOWN remains0/60 because another
side still supplies support; this does not imply complete observation. Dropout
still changes LEFT / LEFT+RIGHT at all nine adjacent transitions.

Of the ten correct same-zone outputs, six contain resolved valid near/far slots.
The other **four have correct coarse direction but UNKNOWN near distance**;
the hidden pole range is not recovered. The reported45 correct direction sets
include those four. RGB remains an unconsumed synthetic annotation; 10/10 in
that family is an independence control, not fusion arbitration evidence.

The fixed panel contains only1 m and4 m valid values. The two gates accept and
reject exactly the same graph edges on every frame. Their real-range/noise
behavior cannot be ranked using this panel.

## Current simulated ToF interface

The inspected [forward model](mz115_zonal_tof.py) and
[sensor wrapper](mz115_zonal_sensors.py) implement a hypothetical finite-footprint
model, not calibrated VL53L8CX firmware. Public observations retain64 zone IDs
and angular bounds, packet availability, and up to **two strongest returns per
zone** with distance, simulation status, noise sigma and signal-strength proxy.
The model uses a chosen0.04 m noise sigma; this is not measured hardware accuracy.

Histograms, private subray coordinates, actor identities and return-to-hit
lineage stay evaluator/provenance-only. They are not predictor inputs. Thus the
interface can carry two resolved slots, but cannot recover a second physical
surface already collapsed to one merged tuple. This round only consumes
authored status/range tuples, not the forward model's physical behavior.

## Validation, metadata and reproduction

Eight focused tests pass: bridge splitting and retention, merged uncertainty,
slot-order invariance, strict/relative thresholds, packet/invalid UNKNOWN,
adjacency and unchanged equal summaries, common-selector behavior, and rejection
of deletion/empty-UNKNOWN as bridge success. The run also verifies exact frozen
baseline replay, source hashes, conservation of every observed zone and original
target tuple, and graph equality between challengers.

Three-arm prediction plus views took0.0780 s on CPU, excluding serialization
and evaluation (`TASK_NOT_GPU_SUITABLE`). No persistent processes or allocated
workers were started. A read-only first-level audit independently checks the
saved results; its receipt is retained beside the run.

`register-experiment` again fails on the existing
`experiments/index.jsonl:303: input_fingerprint does not match input_refs in the checkout or at recorded code_revision`.
The intended `COMPONENT_OR_CHALLENGER / COMPONENT` inheritance remains pending
because the terminal is unregistered. The ledger is untouched; this is local
Development evidence, not a completed structured registration.

```powershell
python research/active/dtr-r0/nearfield/test_tof_depth_connectivity.py
python research/active/dtr-r0/nearfield/run_tof_depth_connectivity.py --output artifacts.local/work/tof-depth-connectivity-20260914/run-v1
```

Use a new output path on replay. Durable predictions, per-frame evaluation,
summary, hashes, registration/inheritance attempts and independent audit are
under `artifacts.local/work/tof-depth-connectivity-20260914/`. Original input
evidence remains under `artifacts.local/work/tof-directional-stress-20260914/`.
