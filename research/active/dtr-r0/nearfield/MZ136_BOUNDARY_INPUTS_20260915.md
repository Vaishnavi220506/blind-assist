# MZ136 shallow-boundary input audit

2026-09-15. User-authorized continuation after the
[grouped readout failure](MZ136_GROUPED_READOUT_20260914.md). Existing controlled
simulation inputs only: one RGB +8x8 ToF +Radar +IMU. This work examines the48
TRAIN shallow-boundary frames, not dev/test outcomes. No new capture or learned
alert-head fitting. MZ129 remains the retained alert baseline.

## What information is present

| TRAIN boundary cohort | Target-distance support |
| --- | ---: |
| Public ToF | 45/48 frames |
| Public Radar | 35/48 frames |
| Either sensor | 48/48 frames |

Support identity is determined **only in the diagnostic evaluator join**. The
predictor cannot select returns using that identity. All245 returned ToF
components happen to have pure target lineage in this cohort (241VALID,
4MERGED); that does not establish this property for other scenes. Raw
measurements, merged components and missing packets remain explicit.

For the closest16 frames, native lateral clearance/overlap is4.49–5.83mm.
Under a hypothetical perfect front-edge bearing, the sufficient depth-error
tolerance is34.33–63.76mm. The lateral and depth error must not be compared as
if they were the same coordinate. The approximate image-edge budget is only
0.81–1.43pixels at640x360, median0.85pixels, before depth/association/yaw errors.

With a truth-selected best native-supported range and perfect edge bearing,
46/48frames meet the sufficient depth tolerance, including14/16closest frames.
This is an oracle diagnostic, not an executable selection policy. The two
failures have missing ToF and a target Radar return whose projected depth
error is about47–48mm. A two-sided error-bound failure is not proof that both
directional classifications fail.

Using the public minimum ToF range directly as depth meets this hypothetical
tolerance on44/48frames (available45/48); projecting that same measurement with
the coarse zone-center direction meets39/48. This is a warning against silently
treating a5.625-degree zone as a point direction, not proof that skipping range
geometry is generally better. Noise and geometric biases may cancel.

ToF measures contributor-weighted slant surface distance; Radar reports the
first hit's horizontal distance. Forty-six ToF components include side-surface
points beyond the object's front plane, by up to152mm. Neither sensor directly
reports the object's center/front depth. The silhouette's inner edge may also
lie on a back corner. Even an exact silhouette does not justify assigning it
the front-plane depth. These assumptions must remain visible in any new input.

## Observable edge frontend

`mz136_boundary_geometry.py` exposes proposals and a front-plane **hypothesis**;
it does not issue alarms or clear raw sensor evidence. It groups near SIM_VALID
ToF returns by a0.25m camera-depth span, uses their image footprints to restrict
the search, and extracts narrow positive horizontal-contrast components.

The initial TRAIN probe covered45/48frames with mean inner-edge error5.063px.
Inspection found two implementation weaknesses: clipping the search at a zone
border can truncate an RGB edge despite absent adjacent returns, and a low
vertical-persistence threshold includes long background tails. A disclosed
revision pads the search by8px, requires positive contrast on at least half the
supported rows, and bridges small internal texture gaps. Coverage stays45/48;
mean error is2.870px, median3.010px, p905.116px, with9/45within1px.

Those errors use the full projected native AABB as diagnostic reference, which
may extend outside the image. They are not automatically comparable to an
image-clipped or rectified edge reference. At the approximate front-plane
zero-overlap decision, the45covered frames give20TP/9FP/2FN/14TN; the other3
remain unavailable (including2positives), not negative classifications. This
does not satisfy the subpixel-scale boundary requirement or the alert goal.

The revised predictions were replayed exactly after source receipt and all48
RGB hashes were checked. The frontend uses CPU OpenCV/NumPy; its implemented
cohort/morphology pipeline has no equivalent GPU backend. A measured frame took
0.00190s under OpenCV4.10.0 and NumPy2.4.4. This does not change the frozen
OpenCV5 MZ129 runtime or its cached predictions.

## Fixed rectified-edge refinement

Hypothesis: perspective tilt and internal texture bias the coarse column box;
an IMU/intrinsics homography followed by signed-gradient edge localization can
reduce this error without fitting another alert head. The fixed method takes
the first proposal by observed contrast support, rectifies verticals, and searches
within eight pixels of each seed edge. It uses a trimmed vertical Scharr profile
and quadratic peak localization. Native geometry and labels are evaluator-only.
One fixed refinement run was allowed; no outcome-driven refinement sweep follows.

| Consumed TRAIN48 diagnostic | Result |
| --- | ---: |
| Coarse proposals available | 45/48 |
| Fine edges available | 38/48 |
| Coarse seed MAE on those same38 frames | 2.886px |
| Refined MAE on those same38 frames | 1.227px |
| Refined median / maximum absolute error | 0.771 / 3.465px |
| Refined absolute error strictly below1px | 24/38 |
| Coarse seed retained, fine edge unavailable | 7/48 |
| No observable proposal | 3/48 |

Both MAEs in this table use **the same38 frames and the same reference**:
project the native AABB convex hull, clip it to the observed image viewport,
then rectify it. They do not compare directly against the earlier full-AABB
reference. Improvement is conditional on refinement availability: it neither
establishes subpixel accuracy across the cohort nor resolves the missing10
fine-edge estimates. The seven failed refinements retain their coarse seed and
an explicit state; they do not become free-space evidence. Independent raw
ToF/Radar data remain unchanged.

All48 predictions were sealed before the native evaluator join, then exactly
replayed with source, receipt and RGB hashes verified. Four focused tests cover
vertical rectification/roundtrip, signed subpixel peaks, missing input and coarse
seed retention. All31 `test_mz136*.py` tests pass. The actual CPU refinement
probe took0.00472s for one representative frame; this is not a latency benchmark
distribution. `research_backend.py` records `GPU_BACKEND_UNAVAILABLE` for the
implemented OpenCV/NumPy pipeline. There is no persistent worker or allocation.

## Decision and registration

Retain the frontend as a **geometry-input component for disclosed Development**.
The useful contribution is more precise observable RGB edge input when available,
with explicit coarse/fine availability. MZ129 remains the alert baseline. This
run performs no dev/test scoring and proves no precision/recall or timing gain.
The unresolved question is how to combine the visible edge with public surface
range support while retaining coverage; the oracle-selected range is not a
deployable answer. This run ends here; no successor capture or fitting is included.

Intended terminal: `MZ136_BOUNDARY_EDGE_INPUT_COMPONENT`, intended structured
role `COMPONENT_OR_CHALLENGER` / `COMPONENT`, scoped to consumed TRAIN geometry
inputs. Supported registration is blocked by the existing
`experiments/index.jsonl:303` input-fingerprint mismatch. The subsequent supported
inheritance command reports an unknown terminal. Registration and inheritance
therefore remain **pending metadata**, not completed authority. Both command
outputs are retained in the artifact root; no ledger or inheritance bypass occurs.

## Evidence and limits

Canonical artifact root:
`artifacts.local/work/mz136-corridor-pair-20260914/`.
`boundary-range-audit-v1` holds all48 frame diagnostics,245ToF component records,
Radar records, source hashes and the evaluator-only oracle calculations.
`boundary-edge-probe-v1` and `boundary-edge-probe-v2` retain both proposal
versions, code snapshots, predictions sealed before the truth join, and raw
errors. The v2 receipt includes the source/RGB replay check and CPU placement.
`boundary-rectified-edge-v1` contains the parameter/source freeze, sealed
predictions, per-frame cases, summary, CPU record, replay verification and
`representative-overlay.png` (first four available frames, green native reference
and red observed edges). Reproduce only in a new output directory: the runner
refuses to overwrite the existing result.

The useful finding is an explicit error budget and a reproducible input
frontend. These data do not prove hardware adequacy, sensor insufficiency,
robust segmentation, deployment performance or alert improvement. A native
oracle edge or range choice must not enter observation-only inference.
