# Frozen supplementation transfers Boundary recall, with substantial extra FP

2026-09-21. One prospective same-simulator Development verification completed:
16 new base groups,48 complete clips,1152 frames. **The rescue signal transfers,
but the consumed-development low-FP result does not.** Retain COMPONENT evidence,
not a runtime or full-alert upgrade. The original full-replacement negative and
consumed-dev complement diagnosis remain intact.

[Frozen protocol](SPATIAL_COMPLEMENT_TRANSFER_PROTOCOL_20260921.md),
[runner](run_spatial_complement_transfer.py),
[independent audit](audit_spatial_complement_transfer.py),
[local disposition](SPATIAL_COMPLEMENT_TRANSFER_DISPOSITION_20260921.json).

## One unchanged decision on a new source

A is original strong geometry with threshold0.4071309640537889 and definite
bypass. C is A OR(original frozen B logit>=7.6612162590026855), followed by the
same optional nonrecursive0.2s hold. No network, checkpoint, threshold,
preprocessing, return law, hold rule or sample selection changed after outcomes.

New seed20260921 supplies four groups per existing cuboid family in the original
parameter/material ranges. All source geometry is disjoint from the40 original
BCE groups and four earlier source specifications. This is new procedural
Development within one generator/simulator, not independent natural distribution
or hardware evidence. Original576 test frames remain unactivated; they were not
used here. All1152 new source frames passed admission; none were excluded.

Core=768 frames,173 positive/595 negative; Boundary=384,173 positive/211 negative.
Each group supplies complete INSIDE/BOUNDARY/OUTSIDE24-frame clips. These are
approach/dwell/retreat trajectories, not walking past/through an obstacle.

## Current prediction and complete-event costs

Counts are TP/FP/FN; FPR includes all labelled negatives, including UNKNOWN.

| Readout | A Core | C Core | A Boundary | C Boundary |
| --- | --- | --- | --- | --- |
| Current |171/17/2|173/36/0|10/1/163|128/10/45|
| Original hold |173/33/0|173/58/0|14/5/159|140/19/33|

| Stratum/readout | A precision | C precision | A FPR | C FPR | A/C false segments |
| --- | ---: | ---: | ---: | ---: | --- |
| Core current |90.96%|82.78%|2.86%|6.05%|16/23|
| Core hold |83.98%|74.89%|5.55%|9.75%|24/30|
| Boundary current |90.91%|92.75%|0.47%|4.74%|1/8|
| Boundary hold |73.68%|88.05%|2.37%|9.00%|5/13|

Boundary current recall increases5.78% to73.99% (+68.21 percentage points),
with118 additional positive frames; held recall8.09% to80.92% (+126 frames).
Event detection grows4/16 to15/16, and current gains occur in15/16 base groups.
The predeclared rescue-signal check (>=10pp and >=8 groups) passes. Repeated
frames are not118 independent obstacle events, and this is not a significance claim.

Core current recovers two frames in HEAD horizontal g03, including its delayed
onset from0.2s to0s. A's original hold already covered those two frames and had
zero Core FN. All16 Core events and every A current/held flag/onset are retained;
the held Core readout gains no TP while adding25 FP frames.

One Boundary HEAD hanging-plane g01 event remains entirely missed. Five detected
Boundary events still start late: four by0.2s and HEAD horizontal g01 by0.4s.
Preserving A's first reports does not make every new event timely or fully covered.

## The new cost is predominantly spatial, not just holding

Among19 added Core current FP,17 lie in OUTSIDE clips; one is an INSIDE pre-entry
frame and one an INSIDE post-exit frame. The19 are concentrated in only three
base groups: BODY protruding-plane g01 (+8), BODY suspended-solid g02 (+3),
and HEAD hanging-plane g02 (+8). Their held Core additions are+10,+5,+10.
Thus a small number of layouts contribute most of the lost specificity.

Boundary current adds9 FP:5 before entry and4 after exit. There are therefore
28 added current FP across the two strata, already present before holding.
After identical hold the incremental total is39 FP (25 Core+14 Boundary),
equivalent to7.8 sampled false-alert seconds, versus5.6s added current cost.
Both strict current and complete no-added-FP/segment gates fail.

| Base group | Added Boundary current TP | Added Core current FP | Added Boundary current FP |
| --- | ---: | ---: | ---: |
| BODY protruding-plane g00 |2|0|0|
| BODY protruding-plane g01 |12|8|3|
| BODY protruding-plane g02 |5|0|1|
| BODY protruding-plane g03 |7|0|0|
| BODY suspended-solid g00 |8|0|0|
| BODY suspended-solid g01 |10|0|0|
| BODY suspended-solid g02 |10|3|1|
| BODY suspended-solid g03 |8|0|0|
| HEAD hanging-plane g00 |2|0|0|
| HEAD hanging-plane g01 |0|0|0|
| HEAD hanging-plane g02 |12|8|3|
| HEAD hanging-plane g03 |10|0|0|
| HEAD horizontal g00 |10|0|0|
| HEAD horizontal g01 |6|0|0|
| HEAD horizontal g02 |5|0|0|
| HEAD horizontal g03 |11|0|1|

This result supports useful complementary predictions but does not support
stable low-FP lateral discrimination. Reducing hold alone cannot remove17
OUTSIDE current mistakes. No alternate cutoff, screen, hold or training arm was
tried in response. Neither this combined-model contrast nor its successful
rescues isolates RGB-specific contribution, ranking loss or a particular feature.

Of120 new current TP (2 Core+118 Boundary),89 have returned native target/corridor
contributors and31 do not. These remain model classifications, not measured
target ranges. UNKNOWN stays656/768 Core and384/384 Boundary in every arm.
Native support is joined only after sealed predictions, never supplied to C.
No masks/IoU, edge-device latency, user-benefit or safety conclusion is claimed.

## Existing data should be reused before routine expansion

The user's reminder prompted a bounded [existing-data inventory](SPATIAL_DATA_REUSE_INVENTORY_20260921.md).
Five closely matched prior sources total4752 frames, including576 reserved test
frames. Existing CitySample BODY-query5k+10k provide15000 frames with native
RGB/depth/camera/ownership records; their old input/label contract needs explicit
adaptation before current corridor training. Static groups cannot replace full
event sequences. Hypersim/SANPO/ZJU have other useful, limited roles.

Counts were verified from existing specifications/manifests/admission summaries,
not a new full payload audit. The additional500 NFO UE frames overlap BODY-query5k
and are not an independent addition. This inventory establishes a reuse option,
not a new fit, source relabeling or authorization to consume protected test data.

## Validation and closure

Three focused source tests cover deterministic new groups, disjointness and
mutation rejection, and exact count-only capture adaptation. A normalized-input
integration check passes. Pre-run review caught integer lattice boxes being
passed to a normalized-box model API; before capture this was corrected to the
exact original model default. Old/new code and protocol receipts are preserved;
no new model outcome existed and no scientific recipe changed.
Staged whitespace review reports one extra blank EOF line in the already sealed
source specification. Its bytes are preserved; a scoped whitespace check with
only `blank-at-eof` disabled passes. No scientific check is waived.

All source/model/code hashes are sealed; the original head matches its training
receipt. GPU backend probes selected CUDA for both encoder and head. These are
offline batch measurements and do not establish edge-device performance.
Predictions were sealed before evaluator join. Independent audit PASS checks source
hashes, native geometry, observation-only A/C flags, metrics, added IDs, per-group
effects, onset retention and original test non-activation.

Task-owned UE descendants were released with no survivors; model processes ended.
All source, checkpoints by reference, predictions, seals, exact false-frame
identities and diagnostics remain under
`artifacts.local/work/ba-spatial-complement-transfer-20260921/`.
Global registration still fails at existing ledger303; supported inheritance
assignment returns unknown terminal. Local structured disposition and receipts
retain that metadata gap without bypassing it.

End this experiment as COMPONENT evidence: rescue signal transfers, specificity
does not meet the frozen upgrade criteria. Existing baseline/demo remains.
No threshold retry, training, retained-test access or automatic successor.
