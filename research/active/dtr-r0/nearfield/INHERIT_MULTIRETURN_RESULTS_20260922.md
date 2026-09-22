# Inherited multi-return: useful information and earlier onset, with extra false alerts

2026-09-22. One authorized EXPLORE on the consumed query-occupancy evaluation
partition is complete. The fixed second-return mechanism supplies useful new
native-backed information across14/16base layouts. With the unchanged hold,
TP/FP/FN improve from **225/26/31 to254/29/2**. All32events were already detected;
entry-sample detection rises17/32 to31/32. This is a real sampled onset/coverage
gain with **3extraFP and2extra false segments**, not a lossless alert upgrade.

Retain **COMPONENT_OR_CHALLENGER / COMPONENT** for hypothetical observation
preservation and its measured task tradeoff. The frozen zero-added-Core-cost
alert-challenger criterion fails. Closest-exported and independent two-return
readouts are identical on all576current and held decisions, so the more complex
readout has no demonstrated additional value. Retain A and UNKNOWN; do not
promote a default, tune a threshold, start hardware work or launch a successor.

## Source and frozen mechanism

[Protocol](INHERIT_MULTIRETURN_PROTOCOL_20260922.md),
[runner](inherit_multireturn_20260922.py),
[independent audit](inherit_multireturn_audit.py),
[synthetic tests](inherit_multireturn_tests.py).
Only576evaluation frames from16base layouts/48clips were regenerated and scored.
The remaining1152train/dev frames were not decoded or evaluated. The source
was already consumed Development after the
[query experiment](QUERY_OCCUPANCY_RESULTS_20260922.md). Each clip has12posed
samples, approaching and departing over about0.47--4.03m; there is **no dwell
phase**. This extends the distance/shape coverage of the old432/576pilots but
retains the same renderer and hypothetical sensing assumptions.

The existing `simulate_two` was reused without changing10cm quantization,
600mm native-mean separation, four-sample minimum,1%relative proxy-weight floor,
strongest qualifying secondary peak, dropout, noise or seeds. All576first slots
exactly reproduce saved original ToF values including invalid entries, and the
full strongest decision dictionaries reproduce saved A. Threshold
0.4071309640537889, full geometric support, definite bypass and nonrecursive
one-frame hold remain fixed. Two disjoint return supports are never filled in.

Native depth enters only the explicit hypothetical sensor generator and later
evaluator. Public packets contain ranges/sigma/proxy signal/status and boxes;
public identities contain only index/id/clip/frame/time. Public observations
were sealed before inference, and all predictions were sealed before the truth
join. Neither class, lateral relation, bounds, labels nor native arrays enter
readout. This is **additional hypothetical sensing**, not equal-input algorithm
improvement or calibrated hardware multi-target detectability. CNH remains paused.

## All-frame task effects

There are256positive and320negative frames. All rows remain; truth coverage is
100%. Negative UNKNOWN silence is not a known-safe TN. Closest and two-return
rows below each represent their exactly matching outputs, not pooled samples.

| Readout | TP / FP / FN | Recall | Precision | FPR | Events | False segments / sampled seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Strongest current (A) |193 /6 /63|75.39%|96.98%|1.88%|32/32|6 /1.2s|
| Closest / two-return current |211 /7 /45|82.42%|96.79%|2.19%|32/32|7 /1.4s|
| Strongest + original hold |225 /26 /31|87.89%|89.64%|8.13%|32/32|20 /5.2s|
| Closest / two-return + same hold |254 /29 /2|99.22%|89.75%|9.06%|32/32|22 /5.8s|

No original current or held TP is lost. Current gain is18TP/1FP; held gain is
29TP/3FP. Both gain in14/16base layouts. All18new current TP have native
target-in-corridor contributors in their actual triggering second slots.
On14positive frames, first-slot target-corridor support is absent and the
second slot newly supplies it. This distinguishes recovered sensing information
from a stronger decision on already represented support.

| Complete subgroup | Strongest current | Closest current | Strongest hold | Closest hold |
| --- | --- | --- | --- | --- |
| Core384frames (128positive/256negative) |122 /6 /6|125 /7 /3|126 /26 /2|128 /29 /0|
| Boundary192frames (128positive/64negative) |71 /0 /57|86 /0 /42|99 /0 /29|126 /0 /2|
| OUTSIDE192negative frames, FP only |5|6|10|12|

Core contains complete INSIDE and OUTSIDE clips, including entry/exit negatives.
Core current false segments6→7 and false duration1.2→1.4s; held segments20→22
and duration5.2→5.8s. Boundary gains coverage without addedFP on this cohort.
Those facts cannot be collapsed into a global no-cost claim.

| Family (64positive/80negative each) | Strongest current | Closest current | Strongest hold | Closest hold |
| --- | --- | --- | --- | --- |
| BODY protruding plane |49 /4 /15|53 /4 /11|56 /11 /8|64 /11 /0|
| BODY suspended solid |50 /1 /14|52 /1 /12|60 /6 /4|64 /6 /0|
| HEAD hanging plane |49 /1 /15|53 /1 /11|56 /6 /8|64 /6 /0|
| HEAD horizontal |45 /0 /19|53 /1 /11|53 /3 /11|62 /6 /2|

Every family keeps8/8events. BODY held recall rises116/128 to128/128;
HEAD109/128 to126/128. All remaining held misses are HEAD horizontal.
Layer/relation/family and all16base-layout complete metrics are saved, including
precision/FPR/UNKNOWN and the full clip-level timing records.

## Onset, interruptions, release and actual support

Core entry-sample detection improves15/16 to16/16. In
`query_occupancy_head_horizontal_g05_inside`, first alert moves from0.8s to0.4s,
removing the0.4s sampled entry delay. Its three current recoveries at frame02,
03 and09 have respectively12,15 and15actual triggering-slot target-corridor
contributors. Frame02 had none in the original first slot. Original hold already
covered frame09, leaving two additional Core held TP.

Boundary entry-sample detection improves2/16 to15/16; its detected-onset median
falls0.4s to0s, while the maximum remains0.4s. No event is newly missed or delayed.
All current arms have32/32events, so this is earlier sampled detection and fuller
coverage, not an event-count gain. Posed0.2s timestamps are not measured human
warning latency.

Current internal silent runs increase15→27 overall (Core3→3, Boundary12→24).
Adding an early isolated alert can turn initial silence into an internal gap;
this does not imply lost current TP, but it remains a real output-interruption
count. The fixed hold removes these sampled internal gaps in both arms.
Overall alert episode counts are52→65current and37→38held; no audio was played
and no speech-queue or actual user burden was measured.

Held exit carryover is3.2→3.4sampled seconds, all in Core. Both held arms have
one **right-censored release**: `body_protruding_plane_g05_inside` stays active
through both observed post-exit samples, so0.4s is observed carryover, not the
complete release duration. This differs from the source report's zero censored
*exit* count: exit is observed here but subsequent silence is not. Current arms
have0.2s total carryover and no censored release.

The one extra current FP is `head_horizontal_g10_outside_04`; its triggering
second slot has zero target-in-corridor contributors. Hold extends this OUTSIDE
false alert by one sample and adds a separate INSIDE exit sample, accounting
for all3extra held FP. Preserve this lateral-ownership cost alongside the gain.

## Observation cost, validation and disposition

The proxy emits8442second returns across36864zone-frames, with at least one
second return on566/576frames. Distance-slot capacity rises64→128per frame.
The public compressed observation array is141034bytes, plus118439bytes of
identities and363bytes of seal metadata. These are local serialization costs,
not measured sensor bandwidth, physical availability, power or hardware latency.
Current UNKNOWN falls487→481frames (Core295→289, Boundary192unchanged);
all320negative frames remain UNKNOWN for every arm.

24focused synthetic tests pass: inherited first-slot/lineage parity, secondary
separation and count rules, dropout, independent random streams, separate
supports/UNKNOWN, readout purity, native projection and nonrecursive hold/event
metrics. Independent saved-output audit passes576frames and27complete groups:
six-arm frame/event/segment/duration recount, onset/hold, full rendered-bound
truth, independent native-ray projection and the frozen secondary-peak law.
It imports no inference, generator or original metric implementation.
`independent-audit.json` retains the counted checks and evidence boundary.

All three scientific stages passed governed UE admission, source-family lineage,
asset registration and master/fabric verification. An initial construct attempt
was blocked before any subprocess by an unregistered plan asset. The supported
asset-catalog registration fixed this metadata prerequisite; the original
failure remains, and the identical frozen scientific plan ran as `construct-v2`.
There was one successful576-frame construction, one prediction and one evaluation.

CPU TASK_NOT_GPU_SUITABLE timings including I/O/checks: construction7.376s,
prediction4.060s, evaluation plus independent audit11.065s. The workload is small
per-zone histogram/scalar geometry; no training or image model inference occurred.
No UE, device connection, GPU worker, background service or paid allocation was
started; all task commands exited. Native source/RGB were reused without copying.

Evidence roots are `artifacts.local/evidence/ba-inherit-multireturn-20260922`
and its `-observations`, `-predictions`, `-evaluated` siblings. They preserve
frozen protocol/source hashes, input identities, private contributors, public
packets, prediction/evaluation seals, all576rows, full metrics, audits and logs.
The private contributor index is intentionally retained for provenance.

This one pilot supports the inherited information-preservation mechanism and
its concrete onset/coverage tradeoff on wider-distance Development. It does not
pass the frozen zero-added-Core-cost upgrade criterion and does not establish
a complex two-return algorithm advantage. No mask is predicted, so there is no
IoU/fine-localization claim. Preserve both gains and costs; stop without tuning.
Root integration owns the shared terminal/ledger registration and delivery.
