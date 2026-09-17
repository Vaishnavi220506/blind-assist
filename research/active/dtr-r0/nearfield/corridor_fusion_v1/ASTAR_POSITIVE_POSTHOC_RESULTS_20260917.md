# A* plus public evidence: no clear-task increment, one boundary event tradeoff

**Prefer standalone A* as the next effect-version candidate.** The requested
exact saved-output OR recovers none of the four old-A clear true frames lost by
A*. All four have zero usable ToF returns, so the public branch is UNKNOWN.
A* and A*+positive both score **96/3/12, F1 92.75%, recall 88.89%, precision
96.97%** on216 clear frames (75% coverage). The branch's76 clear activations are
already A* true positives; there is no clear complementarity in this cohort.

The combination does recover one boundary event at the cost of one boundary
false frame: strict107/6/37 becomes108/7/36, events24/30 become25/30. Preserve
this limited event tradeoff rather than calling the branch entirely inactive.
It does not change the clear-task recommendation or erase A*'s old-A costs.

## Scope and fixed operation

This is **posthoc consumed Development**, requested after the completed
[single-version confirmation](PUBLIC_SINGLE_RESULTS_20260917.md). The original
confirmation remains byte-identical. The separate [analysis protocol](ASTAR_POSITIVE_POSTHOC_PROTOCOL_20260917.md)
applies only `combined = saved control OR saved positive` on all288 frames.
There is no capture, fitting, model inference, threshold search, label change,
temporal filter or change to the public runtime's existing output meanings.

All flags are recomputed from saved scores and their original thresholds:
old A0.3917890013717321, A*0.5568065433174727, positive5.8390960693359375.
Model hashes, original prediction/summary seals, raw/native receipt and all
confirmation files are authenticated before analysis and unchanged afterward.
New predictions, summary, protocol binding and audit live separately under
`artifacts.local/work/corridor-astar-positive-posthoc-20260917/`.

## Matched final-alert table

| Method | Clear TP/FP/FN; F1 | Strict TP/FP/FN; F1 | Boundary TP/FP/FN; F1 | Core / strict events |
| --- | --- | --- | --- | --- |
| Frozen A | 97/25/11; 84.35% | 111/37/33; 76.03% | 14/12/22; 45.16% | 17/18; 25/30 |
| A+positive | 97/25/11; 84.35% | 112/38/32; 76.19% | 15/13/21; 46.88% | 17/18; 25/30 |
| **A*** | **96/3/12; 92.75%** | 107/6/37; 83.27% | 11/3/25; 44.00% | **18/18**; 24/30 |
| A*+positive, posthoc | 96/3/12; 92.75% | 108/7/36; 83.40% | 12/4/24; 46.15% | 18/18; 25/30 |

Clear family results are identical for A* and the combination: BODY35/1/1,
HEAD35/0/1, rod26/2/10. Strict combined precision93.91%, recall75.00%; A*
94.69%,74.31%. Strict F1 increases only0.13 percentage points from A*.
One of24 configurations changes, shallow scene2: one added TP and one added FP
leave its total correct count unchanged. All other configuration outputs match.

Both A* variants retain the clear negative burden3/108=2.78%,3 negative episodes
with an alert and3 false segments, versus old A25/108=23.15%,12 episodes and13
segments. Across strict negatives, the extra boundary false frame raises A*'s
burden6/144=4.17% to7/144=4.86% and false segments5 to6. All A* alerts remain.
The two isolated boundary additions also increase total alert transitions28 to30
and boundary-internal transitions11 to13; clear events and clear nuisance stay
unchanged. There is no smoothing or stable-alert claim.

## The four clear losses and the two additions

| Old A TP lost by A* | Usable ToF returns | Positive logit | Recovered by OR |
| --- | ---: | ---: | --- |
| HEAD scene2 in, t=.75s (`in_03`) | 0 | -30 | No |
| Rod scene5 in, t=0s (`in_00`) | 0 | -30 | No |
| Rod scene5 in, t=.25s (`in_01`) | 0 | -30 | No |
| Rod scene5 in, t=.75s (`in_03`) | 0 | -30 | No |

The -30 value is a missing-return sentinel, not calibrated negative evidence.
All69 zero-return frames preserve A*. The present returned-evidence branch has
no measurement with which to recover these four cases. This observation does
not establish that RGB/Radar/temporal evidence is absent or unusable.

Exactly two new outputs relative to A* occur in one paired configuration:

| Case | Truth | A* score | Positive logit | Max return | Native corridor witness |
| --- | --- | ---: | ---: | ---: | --- |
| Shallow scene2 enter, t=0 | Negative boundary | .123238 | 6.497169 | 90 | No |
| Shallow scene2 exit, t=0 | Positive boundary | .389910 | 6.421956 | 90 | Yes |

The true addition recovers an event A* missed throughout its positive interval.
It is an actual event gain, but only one TP sample; it does not imply stable
tracking. The false addition is counted alongside it. These are two frames in
two episodes of one physical configuration, not two independent configurations.

## Event timing and retained costs

Relative to A*, no core event or onset changes. Core18/18 remains, including rod
scene5 first alarm at.5s rather than old A's0s. The combination cannot recover
that lost half-second. All18 core episodes remain left-censored.

Strict shallow scene2 exit changes from missed to detected at0s (old A first
alert.25s). No A* event or onset is lost/delayed. Its other strict onset changes
relative to old A remain: gains rod scene0 and shallow scene2/5 enter, losses
shallow scene0/1/3 enter, and delayed rod scene5. Thus25/30 now equals old A's
event count, but **three gained and three lost events remain different events**.
Equal aggregate event count is not preservation of every old-A event.

The six strict exit first-off delays remain all0s, identical to A*; all core
release opportunities remain unavailable. Short4Hz intervals and left-censored
obstacles do not demonstrate advance warning or stable real-world release.

Relative to old A, combined clear changes remain3 rescued FN,4 lost TP,22 removed
FP and0 new FP. Strict changes are9 rescued FN,12 lost TP,32 removed FP and2
new FP. Adding the branch therefore does not erase the primary A* tradeoff.

## Retained version and checks

A* is the preferred effect-version candidate for its measured clear performance
and much lower nuisance burden. Keep the public branch as a reusable component
with earlier Development evidence and this boundary-event tradeoff; do not add
it to the preferred online effect path solely for complexity. A*+positive is a
recorded posthoc alternative, not a newly confirmed or deployed method.

The existing [runtime](SINGLE_VERSION_USAGE_20260917.md) returns A* as `control`;
`alert` still means old A OR positive. This analysis deliberately does not
silently change that contract or relabel a component sum as standalone runtime.
A*'s earlier measured comparison mean66.59ms includes old-A head overhead;
the positive branch's2.20ms is prior component timing. No optimized A* or new
combined latency is measured or claimed in this saved-output computation.

Reproduction: run `analyze_astar_positive.py` only into its absent dedicated
output, then `audit_astar_positive.py`. The audit independently reconstructs
scalar counts, native geometry/5cm strata and contiguous event onsets; all four
methods' original results and exact OR pass, all frozen input hashes remain.
No new test suite or inference run is needed for this deterministic accounting.
The machine-readable result retains all per-frame additions, configurations,
families, core/strict events, releases and the four-frame intersection.

Structured registration/inheritance remains subject to the shared ledger303
input-fingerprint error and unknown terminal; supported-command receipts are
retained separately. No shared ledger repair or bypass is performed. The
analysis process exits with no persistent resource or new data allocation.
