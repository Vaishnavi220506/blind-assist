# MZ110: consumed association opportunity diagnosis

EXPLORE, 2026-09-13. One bounded replay of the existing 288 MZ107--109 frames.
Question: do missed positive frames already contain target Radar returns and
RGB regions, and how much recovery can association alone expose? No capture,
training, threshold sweep, hardware model change or production predictor edit.

Authenticate MZ109 cached nominal/interval predictions, raw inputs and capture
receipts. Reconstruct the existing angular, ToF and reciprocal gates from only
observations and cached RGB proposals; require exact nominal decision and
association parity. Seal this trace and a gate-removal counterfactual before
reading evaluator/provenance. Counterfactual removes only ToF agreement and
retains angular gating, reciprocal uniqueness, nominal geometry and fallback.

Evaluate baseline, nominal MZ108, interval MZ109 and the counterfactual separately.
Count frames once, report panels/families and full TP/FP/FN/TN/UNKNOWN. Decompose
each original arm's FN by the earliest missing target-specific stage: retained
hazard Radar, RGB proposal, angular compatibility, ToF agreement, reciprocal
selection, then geometry/readout. Also report overlapping opportunity flags;
stage attribution is not a claim that removing that stage alone fixes a frame.

Evaluator-only target support requires a real-actor Radar provenance ID belonging
to a corridor-positive native object. Project each object's bounds separately;
an existing proposal matches at IoU >= 0.5 (the prior coverage threshold). Reject
proposals matching multiple objects. This is projected-box identity support,
not a pixel mask, visibility proof or deployable association oracle.

Report two existential recovery ceilings on each arm's missed frames: some
identity-supported pair yields positive unchanged nominal extent, with the
existing angular gate and without it. Keep measured range, observed integrated
yaw, pitch, existing RGB box and corridor fixed. Do not substitute true depth,
true pose or true object extent into the readout. These are optimistic candidate
set ceilings, not achievable end-to-end policies; no oracle FP reduction claim.
Also report the smaller ceiling with unchanged interval readout. Truth is used
only in the explicitly evaluator-only diagnostic stage. Ghosts have no target
identity; persistence does not create it. No-support stays UNKNOWN and Radar
alone has no observed height. Current initial body reference is not future intent.

One trace-parity check plus focused tests of gate isolation and ghost rejection.
Stop after this diagnostic and deliver results. A positive ceiling motivates a
separately scoped observable matcher; no ceiling motivates checking missing
range or geometry evidence. Do not start the successor automatically. The source
uses center-ray ToF and hypothetical Radar, so neither result establishes a
physical sensor ceiling, real-route reminder burden or natural event performance.
