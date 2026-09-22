# Single-frame query occupancy Development comparison

User-authorized EXPLORE, 2026-09-22. Test whether joint first-occupancy and
visible-region supervision improves a small RGB + 8x8 ToF network over a matched
query classifier on unseen procedural layouts. No video, App promotion, or
hardware/natural-distribution claim. Previous consumed recipes stay unchanged.

## Representation and source

Both arms use the same pretrained MobileNetV3-small prefix, stride-4 visual
detail, regional ToF tokens, query decoder, initialization and optimization
budget. Both receive six BODY/HEAD x left/centre/right queries. Thus this pilot
isolates the **first-hit plus localization supervision package**, not the
invention of queries or a separate contribution of either auxiliary task.
The classifier predicts one binary probability per query; occupancy predicts
six first-hit bins plus no intersection in the declared range, and a visible
mask. Its alert probability is the occupied-bin sum, with no independent head.

Camera X-right/Y-down/Z-forward, axial metres, calibrated simulation HFOV100.
Query width .6m; x centres -.3, 0, .3m. HEAD Y[-.2,.42], BODY Y[.42,.9].
Depth edges [.3,.75,1.25,1.75,2.25,2.75,3.0]m; closed outer bounds. The geometric
first hit is the nearest intersection with the full rendered bounds of ALL
declared controlled cubes, including the background, not the target centre.
This is controlled-object occupancy, not a complete hidden-scene reconstruction.
Visible masks are computed separately from valid native axial depth; invalid
depth is ignored, and an empty visible mask is not geometric free space.
Unexplained visible surfaces inside a query are reported and that query's
controlled-object label is invalidated rather than silently becoming negative.

UNKNOWN remains the unchanged public ToF evidence axis, independent of learned
alert probability. A probability or bin is a model estimate, never a measured
foreground range or certified clear space. No inherited evidence is overwritten.

Metadata-only reuse search found existing event cohorts cover about2.29-3m;
BODY-query5k/10k anchor bands cover .85-1.35 and2.0-2.6m with different query
semantics and incomplete scene geometry. Preserve these sources. Generate one
1728-frame paired cohort:48 novel groups, four existing cuboid families, whole
groups split24train/8dev/16evaluation; each group has three lateral relations
and12 posed approach/retreat frames spanning about.5-4m. Keep invisible/partly
visible cases and report them. The nominal.2s sampling is not real walking.

## Budget and selection

One capture, 2400s launcher budget with process-tree cleanup. Source failure is
NOT_EVALUABLE; repair mechanical defects with receipts, never silently replace
scientific cases. Training runs on measured actual CUDA if suitable.
Per arm: up to24 epochs, batch16, AdamW, shared seed202609223, lr3e-4,
weight_decay1e-4, cosine schedule; encoder lr is one tenth the decoder lr.
ImageNet normalization,320x180 RGB, no label-dependent crop or augmentation.
Occupancy loss: cross-entropy + mean visible-mask BCE + soft Dice (each weight1).
Classifier loss: mean BCE. Model selection every4 epochs on dev only, by maximum
frame recall at dev FPR<=5%, then fewer false segments, then earlier epoch.
No held evaluation access during fitting or threshold selection. A second
matched optimization attempt (at most another24 epochs per arm) is permitted
only for a documented train/dev fitting defect, before any held predictions.
It is not an automatic sweep. Seal selected checkpoints and public-input
predictions before joining held labels. No tuning after that join.

## Evaluation and decision

Primary alerts are centre BODY or centre HEAD at3m, direct current frame with
no hold. Include unchanged A current and A+one-frame hold as historical
references on the new observations; the strong classifier is the primary peer.
Choose each learned arm's cutoff using dev only. Report realized held costs,
full descriptive PR curves (not posthoc selected replacements), per-family and
INSIDE/BOUNDARY/OUTSIDE TP/FP/FN, precision/FPR, complete events, onset, exit
tail, false segments/duration, UNKNOWN and visible-support coverage.
Report query-bin accuracy/MAE, mask IoU and wrong/lost/gained event identities.
Bootstrap uncertainty by whole layout group, not by query or adjacent frame.

Retain as an alert challenger if the selected held point adds>=10 percentage
points recall with no more than2 extra FP frames and1 false segment, or removes
>=30% FP with recall loss<=3 points; neither BODY nor HEAD aggregate recall may
fall>5 points, and no whole family may lose>1 detected event. This is a bounded
Development usefulness target, not guaranteed outcome or lossless certification.
If only localization improves, retain it only as a component. No joint alert
gain closes this exact recipe and preserves all costs. The two arms cannot
attribute separate causal benefits to first-hit vs mask supervision. Finish
reproducible outputs, focused checks, source/decision registration and scoped
Git delivery; do not add temporal fusion in this task.

Host timings/peak memory are measured for the complete small network. Endpoint
hardware/latency has not been validated; the intended small-model constraint
remains, and a host CUDA result is not an edge-device speed claim.
