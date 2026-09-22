# Retained A operating-scope stability audit

2026-09-22, EXPLORE, consumed simulation Development. The user asks to validate
a clear operating scope and explain misses, false alerts and uncertainty.

The tested baseline is unchanged A: geometric strong threshold
0.4071309640537889, definite-support bypass, nonrecursive one-sample 0.2 s hold,
and clip/timestamp reset. Calibration 0.007085703945147101 is a comparator only.
RGB is part of the available system input but no RGB model runs in this audit.

The declared scope is a fixed, level current-camera query X=[-0.30,0.30],
Y=[-0.20,0.90], Z=[0.30,3.00] metres, hypothetical 45-degree 8x8 single-return
ToF, and complete approach/dwell/depart clips of four opaque axis-aligned cuboid
families: head-horizontal, head-hanging-plane, body-protruding-plane and
body-suspended-solid. Core contains all INSIDE and OUTSIDE clips, including
pre-entry/post-exit negatives. Boundary is a separate challenge, never dropped
from the report. Actual sampled object dimensions, distances and overlaps will
be reported; the query's 0.3–3m interval is not proof of tested coverage everywhere.
No arbitrary shape, material, lighting, rotation, occlusion, missing-return,
walking-speed, body-heading, hardware or natural-environment robustness claim.

Use every saved evaluation identity from exactly two existing cohorts:
ba-spatial-complement-transfer-20260921 and ba-data-coverage-20260921 (1152 each).
Do not use the latter's train/dev labels or activate a protected test. The first
cohort and nominal second-cohort outcomes are already known. This is explicitly
a retrospective scope audit, not fresh confirmation or prospective certification.

One explanatory contrast: do the retained event/cost properties survive both
layout variation and the previously defined projection stress? Replay dx=-2,0,+2
on the 256x192 box grid with every range unchanged, on both cohorts. Both signs
are mandatory; no offset/threshold/path/model selection. The wider cohort's
shifted results fill the one missing cross-layout sensitivity comparison.

Verify parent seals; select identities from saved predictions without outcome
filtering. Recompute from public ranges/boxes, verify nominal score, support,
current/held alerts and UNKNOWN against saved A, then seal all predictions before
the evaluator join. Hash evaluator files opaquely first. Preserve all source files.

Reuse the existing Core usefulness checks, explicitly retrospective here: every
Core event detected, first in-event delay <=0.2s, each coverage >=5/6, maximum
positive silence <=1 sample; >=50% fewer false frames than matching Calibration,
nonincreasing false segments and INSIDE-negative/OUTSIDE false-frame/segment
counts. Report each condition independently and nominal-to-shift deltas. Passing
relative to Calibration does not imply low absolute burden or stable calibration.
No invented practical false-alert tolerance or newly optimized pass criterion.

Report TP/FP/FN, all-negative FPR, precision, event misses/delays/coverage,
false segments and sampled duration, pre-entry/post-exit/OUTSIDE attribution,
family strata, and frame IDs for every miss/false alert. UNKNOWN is lack of
definite current support, can coexist with an alert, and is not a correct negative.
Partition it by no valid return / no possible corridor support / ambiguous support,
and separately show alerting versus silent UNKNOWN. These are readout causes,
not proven physical causes or an object-level observability oracle.

Budget: one public replay of 2304 existing frames x three offsets, two fixed
thresholds, one evaluator join and independent saved-output audit. CPU scalar
geometry/metadata: TASK_NOT_GPU_SUITABLE. No capture, training, web search,
paid allocation, threshold change or App/demo change. Stop after evidence and
delivery; any failed scope condition is recorded, not repaired by excluding cases.
Keep A retained; this audit is a diagnostic COMPONENT, not a new algorithm.
Use supported registration/inheritance commands and preserve any blocker receipts.
