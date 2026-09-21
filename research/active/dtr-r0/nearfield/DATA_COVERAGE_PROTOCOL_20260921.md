# Fixed spatial B, broader training coverage

EXPLORE; one procedural simulation Development experiment, authorized 2026-09-21.

Question: can a broader geometry and appearance training condition improve
Boundary supplementation at a fixed false-alert budget, using exactly the
original spatial B representation, head, loss and training allocation?
POINT/REGION remain closed. R's useful but incomplete rescue remains recorded.
This changes a data condition, not the sensor interface or inference mechanism.

## Source and allocation

Generate 48 distinct axis-aligned cuboid geometries, grouped 24 train / 8 dev /
16 evaluation, with four families equally represented. Each geometry has the
same INSIDE/BOUNDARY/OUTSIDE lateral triplet and 24-frame complete event per
relation: 1728 / 576 / 1152 frames. Seed 20260922; groups and relative geometry
must be disjoint from all six preceding source specifications. Retain original
trajectory ranges, .2-second sampling, camera, map, renderer warmup, background
size range, and single-return simulation law including common paired noise.

Broaden target depth/width/height/world-height ranges (metres): horizontal
.07-.36/.40-1.25/.045-.24/1.60-1.96; hanging
.075-.30/.30-1.00/.18-.58/1.66-1.92; body plane
.10-.46/.30-1.10/.28-.90/1.15-1.55; body solid
.20-.70/.28-.95/.22-.75/1.08-1.48. Stratify each dimension within each family
and split. Balance target/background material assignment across relations and
cross it with distinct geometries. This is not full-factorial rerendering of
the same geometry and does not isolate geometry from appearance contributions.
No new topology, rotations, moving camera orientation or real-world claim.

RGB, depth, ToF and evaluator truth come from each same rendered scene. Preserve
whole-cohort admission: every target visible, fewer than four competing corridor
pixels, exact authenticated bounds and unchanged source checks. Any admission
failure stops as NOT_EVALUABLE; no exclusions, replacement cohort or training.
One capture, 3600-second launcher allocation, task-owned process cleanup.

## Fit and comparison

One new N fit, original spatial_bce_model RECIPE verbatim: frozen MobileNet
features, original B head, 1200 updates, batch 64, unweighted BCE, AdamW,
original seed and learning rate, 1728 training rows. No search or second fit.
No native depth, object metadata, labels or geometry enter inference.

A, original B at 7.6612162590026855, and original R at 5.128307342529297
remain frozen references. Additionally name B_control explicitly: the same
old B weights calibrated only on the new dev split under the same selection
rule as N. This comparison does not replace or reinterpret the saved B cutoff.
Calibrate B_control and N separately: enumerate finite dev scores and a disabled
cutoff above maximum, require the existing per-Core/per-Boundary incremental
cost caps (current FP floor(.01*Nnegative), held FP floor(.02*Nnegative),
held false-alert segments floor(.125*Nclips)); maximize Boundary held TP,
then minimize Core held FP, then Boundary held FP, then prefer higher cutoff.
No evaluation-driven threshold changes. Keep A OR branch and unchanged hold.

Primary evaluation uses only the 16 new, split-isolated groups after fit and
selection. Seal all public-input predictions before joining evaluation labels.
Report TP/FP/FN, precision/recall/FPR, UNKNOWN, frame and event costs, complete
events, first alarms, all groups and four families including HEAD-horizontal.
Preserving A flags/timing is structural retention, not independent model gain.
Keep N as a useful candidate only if all cost caps pass, Boundary held recall
is at least 50%, at least 8/16 groups gain over A and A flags/onsets are retained.
Support a material data-condition gain only if this also improves Boundary
held recall over B_control by at least 10 percentage points. Actual FP costs
may differ within the shared budget; this is not equal-realized-FP dominance.

After primary predictions and result are sealed, use the consumed old 1152
frames only for a fixed-cutoff regression report. They shaped this hypothesis;
they are not fresh confirmation. Original protected test stays unactivated.

## Integrity and stop

Freeze this text, spec, source/capture code and inherited model/evaluator files
before capture. While capture runs, implement the orchestration/materializer
and learning wrapper; freeze their execution hashes before materialization or
fit. This staged code seal does not permit changing the scientific allocation,
threshold rule or criteria. Preserve receipts for mechanical failures; do not
overwrite a completed fit, threshold or sealed prediction. A mechanical repair
may resume an unconsumed stage with explicit original/repaired hashes, never
change the experimental recipe. One result closes this condition, with no
automatic successor. Finish audit, report, inheritance and scoped delivery.
