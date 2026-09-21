# Fixed-score current-only threshold audit

2026-09-21. EXPLORE, consumed same-simulator Development. The user authorized
only this audit after discussion of the last-layer failure. No model fitting,
feature extraction, conditional hold, new data, protected-test access or runtime
promotion. Stop after the result and scoped delivery, regardless of outcome.

Question: does removing the old one-frame-hold constraint from cutoff selection
recover useful Boundary alerts at the same current-frame false-positive budget?
The prior pilot rejected its combined current-plus-hold recipe; its binding
sample was a positive pre-exit frame. That result remains unchanged.

Use only the saved original/uniform/balanced scores and metadata from
`artifacts.local/work/ba-last-layer-20260921`. Keep SELECTION g00/g01 and
EVALUATION g02/g03 (eight layouts / 576 frames each). All three lateral clips
stay together. Both roles have already been consumed; no fresh-confirmation
claim, partition retry or outcome-selected head. Do not read FIT content.

For each of the three fixed scores, current prediction is A_current OR
score >= cutoff. Select the lowest float64 cutoff adding zero current FP to A
on every SELECTION negative: nextafter(max score among negative AND not A),
+infinity). There is no hold constraint and no manually chosen score floor.
Preserve atomic >= ties. UNKNOWN is retained as sensor status, never relabeled
as negative evidence; evaluator truth supplies negative labels independently.
Freeze all cutoffs and both roles' predictions before joining EVALUATION truth
or native contributors. Source hashes and role/prediction seals must reconcile.

Primary diagnostic support requires, on BOTH roles: every A_current flag
retained; no added Core or Boundary FP; no increase in false segments; Boundary
recall gain >= .10 and gains in >= 4/8 layouts. These are prospective diagnostic
criteria, not the older recipe's rescue-retention gates or deployment criteria.
Report original versus uniform versus balanced without selecting a winner from
EVALUATION. A current Core 82/11/2 is distinct from held Core 84/19/0; report
recovery/remaining loss relative to the latter too. Failure stops this fixed
readout low-FP supplement route; success supplies evidence for considering a
separately authorized decoder comparison, never automatic continuation.

Report counts, precision/FPR, group gains, event entry/coverage/internal silence,
false segments/duration, exit costs, recovered/lost frame identities and native
support. Include negative-depth versus valid-depth lateral FP. No masks: IoU N/A.
Apply the unchanged nonrecursive one-frame hold to each frozen current policy
only as a diagnostic counterfactual, never to choose thresholds or claim a new
candidate. Compare it with A_hold to quantify costs of the old decoder.

Use standard-library scalar replay on CPU (TASK_NOT_GPU_SUITABLE). Preserve
the original models, scores, run and demo. Save output under
`artifacts.local/work/ba-current-only-20260921/`, perform an independent saved
record recount, attempt supported registration/inheritance, and retain receipts
if existing metadata prevents those operations. Do not repair unrelated ledger
or working-tree changes. Release task-owned resources after scoped Git delivery.
