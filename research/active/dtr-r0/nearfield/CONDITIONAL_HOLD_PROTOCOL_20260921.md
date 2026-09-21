# Fixed-entry, original-score conditional continuation

2026-09-21. User authorized this separate decoder question after reviewing the
current-only result. EXPLORE on consumed same-simulator Development; the earlier
readout/current-only/RGB negatives remain intact. No model fitting, new readout,
entry-threshold retry, geometry gate, test access or runtime replacement.

Hypothesis: after a frozen current trigger, a lower original-score threshold can
recover silent positive samples without copying alerts into negative exit frames.
Use the original score only. Keep the existing current-only original entry cutoff
15.769264221191408, and all A_current triggers. Per independent 24-frame clip:

    state = OFF
    trigger_t = A_current_t OR original_score_t >= tau_on
    state_t = trigger_t OR (state_previous AND original_score_t >= tau_keep)
    alert_t = state_t

Reset only at the observable clip start, never at evaluator event boundaries.
Continuation is recursive and may exceed one frame; score equality passes.
Current A triggers are authoritative even while ON; this preserves the frozen
baseline rather than suppressing its contributors. No positive label or native
support controls state. UNKNOWN remains unchanged. All scores must be finite.

Use the existing complete-layout partition: SELECTION g00/g01 and EVALUATION
g02/g03, eight layouts and576 frames each. Both roles are consumed. Selection
chooses tau_keep from unique SELECTION original scores below tau_on, plus
tau_on as the no-continuation control. This exactly enumerates selection-state
partitions for >= without a numerical grid or a second search dimension.
Feasible thresholds add zero negative alerts relative to original_current on
all selection Core and Boundary frames. Among feasible thresholds maximize
Boundary TP, then Core TP, then choose the highest threshold. If no added TP is
possible, the tie-break selects tau_on; do not force a lower threshold. Preserve
the complete selection curve and freeze ONE threshold and both roles' predictions
before reading evaluation label values. Hashing labels for provenance is allowed.
Never sweep evaluation thresholds, select a head, alter the split or retry.

Comparators: A_current, A_hold, original_current and original_hold, with the
existing clip-local NONRECURSIVE one-frame hold. Primary comparison isolates the
decoder against original_current, not merely the combined gain over A_current.
Report TP/FP/FN, precision/FPR, false segments and sampled duration, event entry,
internal/terminal silence, exit carryover/release, native contributors and gains
by layout. Distinguish current Core82/11/2 from old held Core84/19/0.

Also report seed opportunity: first current trigger in each clip, whether it is
negative, positive event frames before any trigger, and the theoretical true
suffix reachable by never releasing. This ignores false costs and is explicitly
an upper bound, not a candidate. Decompose added positives relative to current
into initial/internal/terminal or previously missed events; an earlier negative
seed must not be silently treated as a correct event entry.

Keep a local mechanism result if useful, but a candidate needs on EVALUATION:
all original_current flags preserved; zero added negative frames (hence Core
FP<=11 and Boundary FP=0); Core TP>=82; at least3 added Boundary positive frames
in at least2 distinct Boundary events/layouts compared with original_current.
The last two criteria reject a1-2-frame or single-event coincidence without
reusing the previous4/8-layout broad-repair gate. No automatic deployment or
successor follows either result. One frozen selection/evaluation only.

Outputs: artifacts.local/work/ba-conditional-hold-20260921/. Standard-library
CPU replay is TASK_NOT_GPU_SUITABLE. Preserve source seals and predictions;
record independent recurrence/selection/count verification, supported ledger
registration/inheritance receipts, and scoped normal Git delivery. No paid
resources or persistent worker are needed.
