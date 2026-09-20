# Frozen 32D last-layer pilot: no useful low-FP repair

2026-09-21. EXPLORE on consumed same-simulator Development. **Negative result:**
both 33-parameter readouts become silent supplements at the fixed no-added-FP
operating point. They preserve A, but retain zero of 68 evaluation Boundary
frames rescued by the old spatial supplement. No runtime replacement.

## Scope and reproduction

The [frozen protocol](LAST_LAYER_PROTOCOL_20260921.md) fixes two fits, one split
and one cutoff per head. [Runner](run_last_layer_pilot.py) stages are `prepare`,
`extract`, `fit`, `predict`, `evaluate`. Payloads and receipts are under
`artifacts.local/work/ba-last-layer-20260921/`. Source data are referenced from
`ba-spatial-bce-20260920` and `ba-spatial-complement-transfer-20260921` in the
same artifact work directory; source hashes are frozen in `protocol.json`.

FIT uses eight original DEV layouts / 576 frames; SELECTION uses transfer
g00/g01, eight layouts / 576 frames; EVALUATION uses g02/g03, eight layouts /
576 frames. Every layout keeps its three complete lateral clips together.
All roles were previously consumed, so EVALUATION is not fresh confirmation.
The original reserved test was not activated. Role labels are materialized
from existing consumed labels; evaluation truth is joined after sealing
predictions. Native support never enters fitting or cutoff selection.

The frozen network supplies 32 penultimate features. A FIT-only standardizer
precedes a linear head with 32 weights and one bias. Uniform BCE is the control;
equal layout-by-binary-label mass is the challenger. Both use the same fixed
L2=.001, zero initialization and float64 LBFGS recipe. Balanced weighting is
not hard-negative mining: each layout has similar class counts and it mostly
upweights positives. There were exactly two full fits and no later tuning;
the disclosed disposable eight-iteration fits only benchmark CPU/GPU placement.

A is the existing strong geometry baseline. C is A OR the original spatial
score >=7.6612162590026855. New heads also supplement A by OR. Each cutoff is
the lowest representable value that adds no selection FP to A under both
current-frame and unchanged nonrecursive one-frame hold. It accounts for
triggers that would persist into a negative next frame. Cutoffs are
7.2858613574493125 (uniform) and 7.419484618921127 (balanced).

**The binding constraint is a true-positive pre-exit frame**, not a current
negative: both cutoffs are set by `transfer:f0688`, body_protruding_plane_g01
BOUNDARY at 3.2 s. Its A_current is false and its label is positive; it is
forbidden only because hold would extend into negative `transfer:f0689`,
where A_hold is false. This couples spatial separation to the unchanged
temporal policy. No current-only alternative cutoff was selected or evaluated.

## Fixed evaluation results

Eight layouts, 576 frames. Entries are TP/FP/FN. Core includes complete INSIDE
and OUTSIDE clips (384 frames, 84 positives, 300 negatives); Boundary includes
192 frames, 84 positives and 108 negatives. UNKNOWN is retained, not relabeled
as free space. There is no predicted mask, so IoU is not applicable.

| Policy | Core current | Core hold | Boundary current | Boundary hold | Boundary events current/hold |
| --- | --- | --- | --- | --- | --- |
| A | 82/11/2 | 84/19/0 | 4/0/80 | 6/2/78 | 2/8, 2/8 |
| C | 84/22/0 | 84/34/0 | 72/6/12 | 78/12/6 | 8/8, 8/8 |
| Uniform | 82/11/2 | 84/19/0 | 4/0/80 | 6/2/78 | 2/8, 2/8 |
| Balanced | 82/11/2 | 84/19/0 | 4/0/80 | 6/2/78 | 2/8, 2/8 |

| Policy | Core precision current/hold | Core FPR current/hold | Boundary precision current/hold | Boundary FPR current/hold |
| --- | --- | --- | --- | --- |
| A and both new heads | 88.17% / 81.55% | 3.67% / 6.33% | 100% / 75% | 0% / 1.85% |
| C | 79.25% / 71.19% | 7.33% / 11.33% | 92.31% / 86.67% | 5.56% / 11.11% |

Both new heads reproduce **every A alert flag** on all three roles, for current
and hold. Hence A event timing, internal silence, false-segment identity,
exit costs and native contributors are preserved exactly. Evaluation Core
false segments are 10 current / 13 held; Boundary 0 / 2. C has Core 14 / 17
and Boundary 5 / 8. All policies cover 8/8 Core events. Base UNKNOWN counts
are Core 332 and Boundary 192; an alert supplement does not establish a
new valid sensor status.

Independent geometry recount separates negative-depth cases (axial extent
outside the corridor's depth contract) from lateral OUTSIDE at valid depth.
A and both heads have 11 current / 21 held negative-depth FP across Core and
Boundary, and zero valid-depth lateral FP. C has 19 / 34 negative-depth FP
and 9 / 12 valid-depth lateral FP. The lost 68 current Boundary rescues include
50 with native support; the 72 lost held Boundary rescues include 55 with
native support. These are losses relative to C, not losses of A contributors.

Selection also fails: Boundary remains 6/1/83 (held 8/3/81), versus C 56/4/33
(held 62/7/27). Both heads retain 0/50 selection C rescues and 0/68 evaluation
C rescues; neither gains a frame in any of eight layouts. All-A-retention and
no-added-FP gates pass; recall gain, group coverage, >=90% specific rescue
retention and C-event retention gates fail. FIT results are likewise silent
at these selection cutoffs. Per-frame IDs, event timing and per-group metrics
are retained in the role-specific result, metric and change JSON files.

## Ranking improves slightly; useful separation does not

xAUC below is an equal-weight mean over off-diagonal layout pairs: Boundary
positives in one layout versus OUTSIDE negatives at positive-depth matched
times in another. It is not the matched lateral-pair accuracy and is not
a probability calibration score.

| Role | Original | Uniform | Balanced |
| --- | --- | --- | --- |
| FIT eight layouts | .96200 | .99152 | .99298 |
| SELECTION eight layouts | .88146 | .88382 | .88527 |
| EVALUATION eight layouts | .89750 | .91176 | .91325 |
| All 16 transfer layouts, diagnostic | .89117 | .90101 | .90224 |

All three scores preserve Boundary>OUTSIDE in all 173 matched transfer pairs.
Yet all three have a worst off-diagonal cell of zero across the 16 layouts.
Mean ranking gains therefore leave some cross-layout orderings unresolved.
The actual operating-point failure is additionally bound by pre-exit hold,
so attributing the silent supplement solely to absolute score alignment
would overstate this experiment.
This pilot does not prove that the representation is linearly inseparable:
it tests this particular readout, regularization, weighting, data and
current-plus-hold operating contract. It also does not prove a hardware
resolution limit or justify replacing the encoder.

## Execution and disposition

Original logits reproduced exactly (maximum absolute difference 0). Frozen
feature extraction used CUDA on an RTX 5060 Laptop GPU: equivalent probe
median .227 ms versus CPU .501 ms. Tiny readout fitting used
`CPU_FASTER_MEASURED`: the equivalent probe was CPU 4.34 ms versus CUDA
20.24 ms. Uniform/balanced fits took 31/33 LBFGS iterations, gradient maxima
5.55e-8/3.53e-8 and about 12.6/13.4 ms. These are offline workload timings,
not end-device latency. Five focused unit tests pass. The
[independent saved-record audit](audit_last_layer_pilot.py) passes: 72 hash
entries, all five seals, role separation, 33 parameters, objectives/gradients,
standardization, scores, cutoff blockers, current/hold flags and xAUC reconcile.
It adds geometry FP, contributor and event timing diagnostics without fitting,
trunk inference or a new operating point; records are `independent-audit.json`
and `independent-audit.log`.

Local structured disposition is **NEGATIVE_CONTROL** for this complete
readout recipe. Keep A and the existing demonstration; preserve C as prior
rescue evidence under its original component limits. Stop this pilot with
no threshold, loss, regularization, feature, seed or partition successor.

Supported registration failed on the existing
`experiments/index.jsonl:303 input_fingerprint does not match input_refs`;
inheritance assignment subsequently returned `unknown terminal id`.
Both receipts are retained. The local disposition records the intended role
and metadata-pending status; it is not a successful global ledger assignment.
No unrelated metadata was edited or bypassed.
