# Added false alerts: distance-boundary leakage and layout-dependent lateral errors

2026-09-21. Recomputed from sealed saved predictions on the consumed 16-layout,
1152-frame transfer cohort. **Existing inputs carry useful distinctions, but the
tested score and public distance witnesses do not separate all added errors from
rescued positives.** This is a diagnostic consolidation, not a new alert method,
fresh validation, RGB-only ablation or full-input information ceiling.

## Question, method and stop

The user requested error-source attribution and existing-input separability.
Hypothesis: added errors mix distance-window failures, lateral qualification and
hold extension; distinguishing these may explain why one scalar score fails.
Baseline A and B/R/U/G predictions remain frozen. B is the original spatial
supplement (called C in the original transfer report); U/G are the matched
ordered-spatial arms without/with extra geometry. All comparisons below use the
same mode of A as reference. No candidate thresholds have been equalized.

One [saved-output audit](audit_added_error_sources.py) joins frame IDs, evaluator
axis categories and previously sealed public descriptors, then counts all added
alerts. No fitting, inference, capture, threshold selection, decoder or default
change. The score maximum below is a descriptive order statistic, not a selected
operating point. Existing witness presence is cross-tabulated, not applied as a
new alert policy. Stop after this attribution and scoped delivery.

Source revision at start was `6647d2f1`, two commits beyond the user's `4bb76b28`;
the intervening commits concern isolated hardware bring-up. This diagnosis uses
the same simulation cohort and does not incorporate hardware measurements.

## Where B's additional errors occur

| Saved readout | Added Core FP | Added Boundary FP | Added Core TP | Added Boundary TP |
| --- | ---: | ---: | ---: | ---: |
| Current | 19 | 9 | 2 | 118 |
| One-frame hold | 25 | 14 | 0 | 126 |

Of the 28 additional current FP, 15 fail the distance window and 13 have valid
depth but remain laterally outside. These are evaluator categories: distance
failure takes precedence, so four OUTSIDE distance negatives also fail lateral
qualification. They are not 15 causally isolated distance-only errors.

Core's 19 current additions comprise 17 in entirely negative OUTSIDE clips,
one INSIDE pre-entry frame and one INSIDE post-exit frame. All 19 are concentrated
in BODY protruding-plane g01 (8), BODY suspended-solid g02 (3) and HEAD
hanging-plane g02 (8). Boundary adds five pre-entry and four post-exit negatives.
Thus neither hold alone nor HEAD horizontal misses explain the added FP burden.

The 39 additional held FP comprise 26 frames also positive in B_current and 13
hold-only frames. There are 23 OUTSIDE frames, six pre-entry and ten post-exit
frames. The net increase from 28 to 39 is 11, not 13: two current additions were
already positive in A_hold and cease to be additions against the held baseline.
Reporting current errors plus a simplistic hold delta would obscure this detail.

## What the existing public input can distinguish

All 28 added current FP and all 120 rescued current TP have possible-depth
evidence, so that permissive witness distinguishes none of this population.
An existing stricter witness asks whether at least one observed range interval
lies wholly in [0.3,3] m; it does not identify the target or its lateral owner.

| Original B current subset | Frames | Has contained-depth witness | Lacks witness |
| --- | ---: | ---: | ---: |
| Added distance FP | 15 | 0 | 15 |
| Added depth-valid lateral FP | 13 | 7 | 6 |
| Rescued true positives | 120 | 104 | 16 |

Absence of this witness therefore covers 21/28 added FP but also 16/120 rescued
TP. Requiring additional corridor compatibility leaves 5/28 FP and 97/120 TP:
two more false frames distinguished at the expense of seven more true frames.
These are coverage counts, not measured filtered-alert performance with hold.
They do not justify interpreting absence as free space or modifying UNKNOWN.

The [whole-cohort axis audit](AXIS_EVIDENCE_RESULTS_20260921.md) also found this
contained witness absent at both endpoints of all 32 positive clips. Its
conservatism is therefore consequential for entry/exit, not harmless cleanup.

## Same-layout response does not yield a common score threshold

Recounted paired evidence gives BOUNDARY logit > matched OUTSIDE logit in all
173/173 positive-depth pairs. The [paired diagnosis](LATERAL_PAIR_DIAGNOSTIC_RESULTS_20260921.md)
found no pair with identical full ToF arrays, so this response cannot be assigned
specifically to RGB. It does establish useful response to the lateral change.

Across the selected 120 rescued TP and 28 added FP, however, a rescued TP has a
higher B score in only 1121/3360 comparisons (33.36%, no ties). Against the 13
depth-valid lateral errors it wins 397/1560 comparisons (25.45%). These are
conditional hard-subset ranking diagnostics, **not whole-cohort model AUC** or
independent statistical estimates; frames share layouts.

The highest added-negative score is 15.769264 (`f0676`, distance-negative).
Only 16/120 rescued positives exceed it, all from two layouts. Excluding every
added current FP by increasing this original scalar cutoff would therefore
discard at least 104/120 existing rescues on this consumed cohort. Even if the
distance errors were resolved separately, only 17/120 rescues score above the
highest lateral FP (15.351711, `f0712`). This is score overlap, not merely a badly
placed single cutoff; monotone score calibration cannot change that ordering.
No threshold was applied or selected and no earlier stopped recipe is reopened.

## The newer representations do not remove the lateral question

| Candidate | Added current FP / TP | Added held FP / TP | Current FP with contained depth | Current rescued TP with contained depth |
| --- | --- | --- | --- | --- |
| B | 28 / 120 | 39 / 126 | 7/28 | 104/120 |
| R | 3 / 52 | 6 / 64 | 3/3 | 50/52 |
| U | 10 / 50 | 17 / 52 | 3/10 | 39/50 |
| G | 10 / 51 | 16 / 54 | 5/10 | 41/51 |

R's three current FP (five held Core FP) all occur in OUTSIDE HEAD hanging-plane
g01. All three satisfy both contained depth and corridor-compatible contained
depth. They are not errors a distance-only qualification can remove. B does not
alarm on these three frames, showing that a representation change also moves
the failure to a different layout. R's one additional Boundary held FP is a
distance-negative exit frame, generated only by hold.

U/G held Core additions are all OUTSIDE and concentrate in HEAD hanging-plane
g02 and BODY protruding-plane g01. U has 11 Core additions (9 lateral,2 distance)
and 6 Boundary distance additions; G has 12 Core additions (11 lateral,1 distance)
and 4 Boundary distance additions. These preserve a lateral-specificity gap.

The seven-pair [existing RGB anchor diagnostic](RGB_MULTIZONE_RESULTS_20260921.md)
was also inspected visually: visible target displacement is present, but that
specific whole-zone anchoring procedure explains zero negative frames. This
rejects that attribution recipe, not the existence of usable RGB information.

## Decision and evidence limits

Distance information can distinguish a substantial portion of added errors,
with nonzero true-positive and event-boundary costs. The unresolved harder part
is lateral qualification across layouts: the same conservative public summaries
can accompany both recovered true obstacles and out-of-corridor false alerts.
Current scalar scores cannot retain all rescues while removing these errors.

Keep this as COMPONENT diagnostic evidence. Do not promote a distance filter,
another threshold, R/U/G, or a new network on this result. No full-input collision
or information-ceiling test was performed: full RGB plus 64 ranges may contain
other recoverable distinctions. Baseline, demo, existing negative controls,
UNKNOWN and protected test remain unchanged. No successor was started.

Reproduction: `python research/active/dtr-r0/nearfield/audit_added_error_sources.py
--output artifacts.local/work/ba-added-error-sources-20260921/result.json` (the
output must not already exist). Evidence includes all subset IDs, layout counts,
witness counts, score comparisons and input/script SHA-256 hashes. Four parent
file seals/protocol bindings and all 1152 ID/truth associations passed. A separate
read-only recount independently agreed on R/U/G origins and witness counts.
CPU scalar reductions only; no persistent worker, GPU or paid resource allocated.

Supported registration remains blocked by the pre-existing
`experiments/index.jsonl:303 input_fingerprint` mismatch; supported inheritance
reports an unknown terminal. Local disposition and both command receipts are
preserved beside the result; the global ledger was not bypassed or repaired.
