# Existing-data lateral tolerance re-evaluation

User redirected work before training to stored predictions and native geometry.
No new model fit, inference, threshold, sensor, depth, capture or label selection.
The complete old and new288-frame cohorts remain consumed Development.

Keep x=.2..3.6m and z=.4..2.05m, and the nominal lateral half-width.30m.
For each native AABB satisfying original inclusive x/z intersection, define
`m = min(hi_y + .30, .30 - lo_y)`; frame margin is max over ALL such objects.
This is lateral positional penetration, not overlap length or object diameter.
Thin central obstacles remain positive. No returned-sensor or RGB-visibility
filter determines inclusion. If no object satisfies x/z, frame is clear negative.

Evaluate fixed t=.03,.05,.10m side by side, without choosing best F1:

- core positive: any object intersects closed half-width(.30-t) corridor;
- clear negative: every eligible object is outside closed half-width(.30+t);
- boundary: remaining frames, retain original strict label and score separately.

Core contact counts positive; outer contact remains boundary. t=0 exactly
reproduces original strict truth. Values come from native bounds, not assumed
source offsets or inferred sensor detections. These are prototype analysis
tolerances, not validated human safety thresholds.

Authenticate saved A/B/C/S1 old-domain decisions and A/S1 new-domain decisions.
Global B/C were not run on the new domain; do not invent them from sparse C calls.
Report strict whole-cohort metrics, clear-sample metrics AND coverage/counts,
boundary strict metrics/alert rate, family strata and signed-error distance bins.
Also show x/z-relevant object availability and unchanged complete strict errors.

Replay existing binary decisions without filtering: core-event detection and
first alert delay, boundary-to-core warning offset, clear-negative alert burden,
alert transitions and release at observed core-to-clear-negative exits. Episode
boundaries remain separate. Six4Hz samples span1.25s; sampled-duration accounting
uses.25s per frame including the last bin. Report censoring and absent release
opportunities. No newly chosen temporal success threshold or composite score.

New A and S1 arrays are identical; enforce identical results under every shared
definition. Score changes from excluding boundary samples are task-definition
sensitivity, never model improvement. Preserve all original labels/predictions.
