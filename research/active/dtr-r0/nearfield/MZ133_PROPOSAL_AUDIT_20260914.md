# MZ133 proposal audit: resolution, attribution and active sampling

Audit date: 2026-09-14. Scope: supplied research proposal, MZ128/129/131/132
reports, owning readout/simulator code, saved MZ129 outputs and four cited
external sources. This is a factual/design audit, not an experiment terminal.
No predictor, archived evidence, default or registration is changed here.

The proposal identifies a useful separation between spatial measurement and
decision logic. Its local numeric claims are supported, with the qualifications
below. The main external correction is that **VL53L9CX volume-production status
is not confirmed by the retrieved sensor product page**. Higher-resolution
simulation remains useful without a hardware purchase assumption.

## Local facts and limits

| Claim | Inspected evidence and conclusion |
| --- | --- |
| One possible central-zone support can sustain an alert | `mz128_zone_weighting.py` defines `FOUR=(0,0,1,1,1,1,0,0)` and threshold 1. `readout` deduplicates active zone IDs and adds each weight once. `mz115_spatial_allocation.possible` uses interval overlap with forward 0.2–3.6 m, lateral ±0.3 m and height 0.4–2.05 m. Supported for this binary arm; independent Radar, guards and certain-coarse support can also alert. |
| MZ132 changes support on 20 old FP but removes only two | [MZ132 report](MZ132_RESULTS_20260914.md) states this for the mask arm. It also adds two different BODY FP, so totals remain 139 TP / 93 FP / 5 FN. Do not omit the additions or call this a net two-frame gain. |
| Radar-only improvement has at most four remaining FP opportunities | [MZ128 diagnosis](MZ128_FP_DIAGNOSIS_20260914.md) reports 89/103 FP independently supported by ToF. [MZ129](MZ129_RESULTS_20260914.md) preserves ToF and removes ten FP without additions. Saved-output recount independently confirms 89 ToF-sufficient cases among the remaining 93. This bound holds for fixed ToF and the fixed OR decision architecture/population. It does not bound Radar's recall, timing, association value in a changed joint model, or performance on other scenes. |
| MERGED uses 0.02–4 m and bypasses ordinary RGB assignment | `mz115_spatial_allocation.allocate` sets these bounds; `assign_groups` admits only `SIM_VALID`. Supported. Status is a simulator category, not an ST firmware status equivalence. |
| 75/82 FP-supporting MERGED returns have one source actor | Supported by the [MZ128 diagnosis](MZ128_FP_DIAGNOSIS_20260914.md). `mz115_zonal_tof.measure_zone` marks MERGED for merged maxima **or component range span ≥0.10 m**. One extended/slanted actor can therefore cause it. One actor does not imply one depth or a proven narrow surface interval. |
| MZ131's 158 removed native corridor samples and one lost TP are separate | Supported by [MZ131](MZ131_RESULTS_20260914.md): the lost rod frame has zero newly removed native corridor contributors. The 158 samples are across 54 returns/24 frames elsewhere. This distinction does not excuse either failure or prove that the missing real hazard was observable. |
| The rod example ranges are real saved diagnostic values | [MZ128 diagnosis](MZ128_FP_DIAGNOSIS_20260914.md) records zone 50 of `mz123_near_rod_farwall_pair1_out_09`: measured 3.04 m, native lateral bounds [-0.536,-0.497] m, predicted [-0.600,-0.168] m. Native geometry remains evaluator-only. |

The read-only recount matched all 288 MZ129 prediction IDs to the saved frame
report, selected false alerts using that report's labels, and counted
`score >= 1 OR certain_coarse`. Inputs inspected:
`artifacts.local/work/mz129-extent-correction-20260914/replay-v1r2/predictions.json`
and `artifacts.local/work/mz123-frozen-early-20260913/returned-v1/analysis-v1/frame-report.json`.
The four remaining FP without sufficient ToF are:

- `mz123_suspended_head_pair0_out_01`
- `mz123_suspended_head_pair0_out_09`
- `mz123_shallow_boundary_stress_pair0_enter_00`
- `mz123_shallow_boundary_stress_pair0_exit_08`

All four have `common_radar=true` and empty guards. This is an opportunity
ceiling, not proof that all four can be corrected without losing a true alert.

## Official hardware and paper checks

External reads used Exa search/fetch and primary sources on the audit date.
The supplied UM3109 landing page timed out; its claimed FoV was checked against
the official VL53L8CX datasheet instead. Counts describe sources actually read,
not search-result limits. No full-literature-review or procurement claim follows.

- **VL53L9CX:** the retrieved [ST sensor product page](https://www.st.com/en/imaging-and-photonics-solutions/vl53l9cx.html)
  supports up to 54×42 zones, 55°×42° FoV, depth/IR/confidence outputs and
  I3C/MIPI CSI interfaces. Its sensor heading says Evaluation, under
  characterization, with limited engineering samples; some listed boards and
  software separately say Active. Do not transfer an accessory status to the
  sensor or report confirmed mass availability. Price, stock, actual operating
  power and integration readiness were not established.
- **VL53L8CX:** [ST datasheet DS14161 Rev 12, July 2025](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf),
  pages 1 and 4–5, supports 4×4/8×8 and a nominal 45°×45° detection volume.
  The FoV is measured under specified reflectance, illumination and configuration
  conditions; it is not a hard ideal rectangular response for every scene.
  Thus 29.5 cm at 3 m for 45°/8, 5.3 cm for 55°/54, and 5.2 cm for a 1°
  yaw error are reasonable nominal geometry calculations, not accuracy promises.
- **Forward-model precedent:** the abstract and opening model discussion of
  [Thrun, Learning Occupancy Grid Maps With Forward Sensor Models](https://robots.stanford.edu/papers/thrun.occ-journal.pdf)
  support the proposal's dependency argument: independent inverse-cell updates
  can incorrectly close an open doorway, while a joint forward explanation can
  reconcile the measurements. This does not establish a suitable ToF likelihood,
  corridor-warning calibration or an improvement in this project.
- **DELTAR:** the [authors' arXiv abstract](https://arxiv.org/abs/2209.13362)
  explicitly treats lightweight ToF measurements as regional depth distributions
  and combines them with RGB. The abstract was inspected, not the full method or
  code. It supports the representation motivation; it does not verify that this
  simulator's two returned tuples expose DELTAR-equivalent inputs, nor that the
  proposed local model inherits its results.

## Decision-changing gaps for execution

1. **Resolution contrast must regenerate measurements.** Keep 45°×45° FoV and
   the same ±11.25° central horizontal alert domain (four of eight columns becomes
   sixteen of thirty-two). Preserve full-field association context. Recompute
   finite-footprint rays, mixture reduction and statuses; old 8×8 interpolation
   cannot test the hypothesis. Truth geometry may generate observations in a
   simulator but must not become predictor positions, actor IDs or labels.
2. **Geometry benefit is not free hardware benefit.** The current forward model
   samples nine rays per zone, uses per-zone strength normalization, 0.04 m noise
   and a fixed signal floor. Keeping these at 32×32 increases total ray samples
   sixteenfold while preserving per-zone measurement quality. Label this as a
   geometry-capability tier. A separate disclosed sensitivity can reduce signal
   and increase noise/misses at fixed total optical/integration budget; its
   scaling is an assumed stress model until calibrated, not ST performance.
3. **MERGED is both a measurement-model and a readout bottleneck.** Finer zones
   can change how often the ≥0.10 m spread rule fires, so an improvement may
   combine finer angles with fewer full-range statuses. Report status counts,
   active support counts, ToF-only and full-system outputs. A null result cannot
   by itself identify MERGED, pose or readout as the unique cause; distinguish
   them with support attribution before choosing another experiment.
4. **Joint attribution needs identifiable alternatives.** Retain background,
   off-image and unexplained hypotheses. Predict the actual simulator's smoothed
   histogram/peak/weighted-return process where used, rather than inventing a
   generic average-return likelihood. A good roadside explanation does not
   certify absence of an additional thin corridor hazard. Uncalibrated fit scores
   must not be labeled posterior probabilities. Raw evidence retention and alert
   decisions are separate, but any withdrawn alert needs missed-hazard and timing
   accounting, including UNKNOWN as a non-alert outcome.
5. **Active sampling must add measured discrimination.** Declare known pose
   offsets and the extra observation budget. Fixed-scene counterfactuals avoid
   mixing motion with sampling effects. Intersecting old envelopes alone is not
   source correspondence or evidence that a newly visible surface is the same
   contributor. Compare additional-view and single-view results with their cost.
6. **Development and evaluation remain distinct.** The old 288 frames and MZ132
   ideal visible masks are consumed controlled Development tools. The latter are
   rendered source proxies, not demonstrated RGB frontend performance. Declare
   any new paired scenes and candidate rule before reading new outcomes; separate
   within/outside, removed-hazard/retained-background and retained-hazard/changed-
   background cases. Once inspected for adaptation, those scenes are Development
   too. No resulting simulation score establishes hardware or user safety.

The audit supports the scoped resolution comparison followed by observable joint
attribution and active-disambiguation prototypes within the user's authorization.
It does not support buying hardware from a claimed production status, training a
new RGB network from MZ132, or treating unchanged alert reasons as mandatory
inheritance. Existing failed experiments retain their recorded negative roles.
