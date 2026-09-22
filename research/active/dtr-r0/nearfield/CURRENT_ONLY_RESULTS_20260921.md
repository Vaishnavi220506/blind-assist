# Current-only audit: local rescue, no cross-layout repair

2026-09-21. Consumed same-simulator Development. Removing hold from cutoff
selection releases real Boundary alerts, but does not establish a useful
cross-layout repair. Original/uniform/balanced scores rescue 12/8/5 evaluation
frames, all in one layout. The original score outperforms both new readouts at
this operating point. Keep the baseline and demo; stop without conditional hold,
new training, cutoff retries or protected-test access.

## Fixed comparison and provenance

The [protocol](CURRENT_ONLY_PROTOCOL_20260921.md) was written before selection.
[Runner](run_current_only_audit.py) reuses saved scores from the
[last-layer pilot](LAST_LAYER_RESULTS_20260921.md), without inference or fitting.
SELECTION is transfer g00/g01, EVALUATION g02/g03: eight complete layouts and
576 frames each, disjoint by layout. Both were previously consumed. Predictions
and cutoffs were sealed before joining EVALUATION truth/native values; hashing
the label file for provenance is disclosed separately from reading its values.

Each policy is A_current OR score >= its SELECTION-only cutoff. The cutoff is
the next float64 above the maximum score of a negative frame not already alerted
by A. All Core and Boundary negatives participate, including negative-depth
cases. UNKNOWN sensor status remains unchanged; it is not negative evidence.
Old nonrecursive one-frame hold is replayed only as a diagnostic counterfactual.

| Score | Prior current-plus-hold cutoff | Current-only cutoff |
| --- | ---: | ---: |
| Original | Not reselected under the old joint contract | 15.769264221191408 |
| Uniform | 7.2858613574493125 | 5.577118945421804 |
| Balanced | 7.419484618921127 | 5.911917173597916 |

All three current-only cutoffs are now bound by the actual negative
`transfer:f0676`, body_protruding_plane_g01 BOUNDARY at 0.8 s, rather than the
positive pre-exit `f0688`. The original score's previous fixed-high supplement
used 7.6612162590026855; that was a different operating contract and is preserved.

## Current-frame results

Entries are TP/FP/FN. SELECTION Core has 89 positives / 295 negatives, Boundary
89 / 103. EVALUATION Core has 84 / 300, Boundary 84 / 108. Core includes complete
INSIDE and OUTSIDE clips; Boundary is separate. There are no masks, so IoU N/A.

| Policy | Selection Core | Selection Boundary | Evaluation Core | Evaluation Boundary | Evaluation gaining layouts |
| --- | --- | --- | --- | --- | --- |
| A | 89/6/0 | 6/1/83 | 82/11/2 | 4/0/80 | - |
| A OR original | 89/6/0 | 10/1/79 | 82/11/2 | 16/0/68 | 1/8 |
| A OR uniform | 89/6/0 | 10/1/79 | 82/11/2 | 12/0/72 | 1/8 |
| A OR balanced | 89/6/0 | 10/1/79 | 82/11/2 | 9/0/75 | 1/8 |

Evaluation Core precision/FPR is 88.17%/3.67% for every current policy;
Boundary precision/FPR is 100%/0%. Boundary recalls are 4.76%, 19.05%, 14.29%,
10.71%; gains are 14.29, 9.52, 5.95 percentage points. All existing A current
flags are retained, with exactly the same current FP identities: 11 Core
negative-depth FP, zero Boundary or valid-depth lateral FP. Core false segments
remain 10 and sampled false duration 2.2 s; Boundary remains zero.

Selection gains are the same four frames for every score, all in
body_protruding_plane_g01 (2/4 have retained native corridor support). Evaluation
gains are all in head_hanging_plane_g02: original 12 (10 native-backed), uniform
8 (7), balanced 5 (4). Both new-head rescue sets are subsets of the original's.
Native support is evaluator-only attribution, not a new inference input.

Each candidate detects 3/8 Boundary events versus A's 2/8, on both roles. The
new evaluation event has 12/12, 8/12, 5/12 current coverage respectively. All
three enter on its first positive sample; uniform has three internal silent
frames plus one terminal silent frame, balanced six internal plus one terminal.
Existing A event timing/coverage is preserved. Core current still misses
`f0221` and `f0227` in head_horizontal_g03 INSIDE: 0.2 s initial delay and one
internal silent frame. Old A_hold recovers both; no current-only arm does.
Therefore unchanged current Core is not full retention of held Core 84/19/0.

UNKNOWN counts remain selection Core324/Boundary192 and evaluation
Core332/Boundary192 for every arm, whether alerting or silent. Unsupported
silence is not counted as a certified true negative.

## Old-hold counterfactual and interpretation

| Policy with unchanged hold | Evaluation Core | Evaluation Boundary | Boundary false segments / sampled seconds |
| --- | --- | --- | --- |
| A | 84/19/0 | 6/2/78 | 2 / 0.4 |
| A OR original | 84/19/0 | 18/3/66 | 3 / 0.6 |
| A OR uniform | 84/19/0 | 17/2/67 | 2 / 0.4 |
| A OR balanced | 84/19/0 | 14/2/70 | 2 / 0.4 |

On SELECTION, each candidate's Boundary 10/1/79 becomes 13/4/76 with hold,
versus held A 8/3/81: all add the known exit FP `f0689`. On EVALUATION only
original adds an exit FP (`f0473`); both new heads retain A's two Boundary FP.
Their rescues are not universally erased by old hold. All FP in this audit,
including these tails, are negative-depth cases. Full event/exit/per-group
records are retained in the output; durations use sample count times 0.2 s.

The old hold constraint did suppress genuine local current-frame signal.
However, it was not the only limitation: after removing it, the new readouts
still fail to deliver broad low-FP gains and are worse than the original score.
xAUC improvement does not establish improvement at this operational cutoff.
All heads fail the prespecified diagnostic criterion on both roles: gains span
1/8 rather than at least 4/8 layouts; SELECTION recall gains are only 4.49 pp.
No outcome-selected replacement, conditional decoder or two-head training follows.

## Verification and disposition

Three focused threshold tests pass (pre-exit positive, atomic ties, preexisting
A false positive). The [independent saved-record audit](audit_current_only.py)
passes: 41 hash entries, source role/prediction seals and output prediction/
evaluation seals reconcile. It independently reconstructs cutoffs, current and
clip-local nonrecursive hold flags, full frame/event/segment metrics, group gains,
native changes, FP geometry and decision gates. FIT files included by the source
role seal are hashed only, not parsed. This scalar replay used standard-library
Python on CPU (TASK_NOT_GPU_SUITABLE), with no paid allocation, GPU/model load
or worker. Audit receipts are `independent-audit.json` and `independent-audit.log`.

Payloads, protocol snapshot, source hashes, cutoffs, predictions, metrics and
per-frame changes are retained under
`artifacts.local/work/ba-current-only-20260921/`. No source/model files were
changed. Local disposition is NEGATIVE_CONTROL for broad current-only low-FP
repair with these three frozen scores and this fixed partition; the observed
local rescues remain diagnostic evidence. This does not reject every decoder
or establish a raw-sensor information limit. Supported registration again fails
on `experiments/index.jsonl:303 input_fingerprint does not match input_refs`;
inheritance assignment returns `unknown terminal id: ba-current-only-20260921`.
Both receipts are preserved, with global metadata pending and no ledger bypass.
