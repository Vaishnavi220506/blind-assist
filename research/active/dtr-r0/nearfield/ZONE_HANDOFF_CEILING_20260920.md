# Zone-Handoff Ceiling: frozen consumed-data diagnostic

2026-09-20. EXPLORE. One pass over the existing 36 clips / 432 frames.

Question: does the frozen per-frame maximum lose distributed adjacent support
in the four newly split FP segments (five withheld frames)? Calibration already
integrates footprint overlap. Keep its formula, depth envelopes, threshold
0.007085703945147101, definite bypass and saved outputs unchanged.

Source: `artifacts.local/work/ba-core-transfer-20260920/`; gap identities come
from the sealed `ba-calibration-footprint-audit-20260920/audit.json`.
Baseline is 144/146/0, 24/24 events, FP segments40, OUTSIDE-layout FP87.
This is consumed simulated Development, not fresh confirmation or hardware truth.

## Frozen diagnostic

Use each saved possible zone with strictly positive joint score (floor exactly
zero). Rank descending joint score, break ties by zone id. Record top1/top2,
their sum, grid adjacency, original depth compatibility and component membership.
Retain all original intervals and values. No RGB, fitting, temporal state,
threshold selection, new capture or alternative connectivity sweep.

An edge requires all of:

1. Four-neighbor 8x8 zone ids AND matching actual shared box edge. Diagonal-only
   touching is excluded.
2. Original axial-depth intervals overlap with positive length.
3. Their shared ray-slope edge has positive-length corridor intersection over
   a positive-length common depth interval inside Z=[0.3,3], X=[-0.3,0.3],
   Y=[-0.2,0.9]. No expansion of depth or footprint is allowed.

Form graph connected components. Record each sum of unchanged zone scores and
the maximum component sum. Pairwise compatibility permits a sloping surface;
transitive connectivity is not proof of one physical object or a global common
depth. Also record unconstrained all-positive-zone sum as a deliberately
permissive upper bound, not an alternative selected method.

The diagnostic hypothetical alert uses the same definite bypass and inclusive
threshold with maximum component sum. Baseline predictions are immutable. Seal
all observable features before joining evaluator labels/gap ids. Report all432,
TP144, baseline FP146, negative-withheld142, gap5, plus BODY/HEAD, layout,
event/onset, UNKNOWN, FP segments/duration and added alert identities.
TN is zero: negative-withheld rows are UNKNOWN abstentions, not certified TN.

Summing zone-normalized overlap fractions is an optimistic pooling hypothesis,
not additive physical probability, conserved area, independent likelihood or
proof of same-surface handoff. All scores are nonnegative, so this hypothetical
readout can only add alerts. It cannot reduce FP frames below146. Filling known
false-alert gaps adds FP; distinguish intended bridge frames from collateral FP.

## Conditional implementation and stop

Before outcomes, operationalize the user's '3/4 or 4/4 gaps, very little extra
FP' as: fully bridge at least3 of4 gaps (all five gap frames counted separately),
add at most5 total FP including bridge frames, introduce no new FP segments
containing zero baseline FP frames, and retain at most37 total FP segments.
Every original TP, event and first in-event alert must remain. If these pass,
implement the exact frozen component rule as a separate branch challenger with
parity checks and report its consumed-data status. Do not promote it from this
diagnostic alone. Otherwise close this exact aggregation explanation; do not
tune floors, connectivity, depth tolerances or thresholds, and do not launch a
temporal or score-dip successor automatically.

CPU small-graph/scalar diagnostic: `TASK_NOT_GPU_SUITABLE`. Canonical ignored
payload: `artifacts.local/work/ba-zone-handoff-ceiling-20260920/`. Preserve source
hashes, frozen protocol/code, features, graph edges/components, result and
registration/inheritance receipts. No persistent worker or paid resource needed.
