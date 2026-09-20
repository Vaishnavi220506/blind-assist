# Score-Factor Collapse Audit

2026-09-20. One consumed-data attribution audit, no new algorithm or successor.
Calibration remains144/146/0,24/24 events; overlap formula, threshold and all
predictions remain immutable. Zone-Handoff aggregation stays closed.

Use the four previously authenticated gaps (five withheld negative frames) and
their enclosing previous/next Calibration-alert frames. The two-frame gap shares
the same flanks for both calculations; do not treat its other gap frame as an
alert reference. Expand all13 window frames with dominant possible zone, d, o,
s=d*o, s/threshold and support interval. Rank saved joint score, ties by zone id.
The fixed threshold is0.007085703945147101.

For each gap frame calculate the user's two descriptive counterfactuals:

- `CF_depth = d_gap * median(o_before,o_after)` keeps depth and replaces overlap.
- `CF_overlap = median(d_before,d_after) * o_gap` replaces depth and keeps overlap.

With two flank values the median is their arithmetic mean. Use the actual
dominant zone at each flank for this requested comparison, then explicitly
record the cross-zone comparison. Report both d/d_ref and o/o_ref, and signed
log-score contributions `log(d_ref/d_gap)` and `log(o_ref/o_gap)`; their sum
equals `log((d_ref*o_ref)/s_gap)`. Do not choose a new numerical collapse cutoff.

Threshold attribution categories describe the arithmetic, not physical cause:
only replacing overlap passes -> OVERLAP_REPLACEMENT_ONLY; only replacing depth
passes -> DEPTH_REPLACEMENT_ONLY; both pass -> EITHER_REPLACEMENT_SUFFICIENT;
neither passes but replacing both passes -> BOTH_REPLACEMENTS_REQUIRED; if even
the reference-factor product fails -> REFERENCE_NOT_EXPLANATORY. Avoid forcing
ambiguous cases into a three-class physical-cause label.

Track each zone that is dominant anywhere in a gap window at every frame of that
window. Recalculate the same two CF using the GAP WINNER's own flank factors when
both are available. Missing returns are missing, not zeros. A valid return with
no overlap in0.3..3m has d=0 and conditional o undefined, not a fabricated o=0.
Keep fixed-zone CF not-evaluable when a required factor is undefined. Include all
saved anchors for each selected zone even when it is not possible or dominant.

Verify source seals and original score/code hashes. Read only saved observations,
scores, intervals and authenticated gap/frame records; no RGB, model or native
ownership oracle is needed for this descriptive factor question. Hash-bound
original predictions remain unchanged. CPU scalar JSON/math audit is
`TASK_NOT_GPU_SUITABLE`. Preserve code/input/protocol hashes, full tables,
counterfactuals, validation and metadata receipts under canonical
`artifacts.local/work/ba-score-factor-collapse-20260920/`.

Interpretation limits: d is the fraction of an axial support interval inside the
depth slab, not return confidence, near-surface purity or foreground energy.
o is conditional on that same depth interval and zone footprint; factors are
coupled. A dominant-zone change and cross-frame factor substitution are not
independent interventions or proof of hardware dropout/calibration failure.
All five gaps are negatives; a lower score can be correct suppression, so none
is a missed true obstacle. No successor, formula change or hypothetical output
stream will be implemented in this audit.
