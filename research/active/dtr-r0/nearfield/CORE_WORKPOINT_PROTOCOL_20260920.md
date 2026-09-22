# Core alert operating-point feasibility on existing scores

2026-09-20. User-authorized EXPLORE diagnostic on consumed Core432.
Question: does the unchanged geometric score permit useful false-alert reduction
when preserving Core events and first in-event alert times, rather than every
old Thin96 boundary-positive frame? This is a new scoped diagnostic, not a retry
or amendment of the frozen Calibration/transfer experiments.

## Fixed inputs and decisions

Read saved `ba-core-transfer-20260920` predictions, frame-results and seals.
Keep the full432-frame denominator for reporting. Primary Core subset is the
existing complete INSIDE+OUTSIDE layouts:288 frames,72positive/216negative,
including all pre-entry negatives and the original four near-depth boundary-band
negatives. BOUNDARY layout144 remains a separate cost stratum. No truth edits,
scene filtering, sensor reconstruction, model calls, RGB veto or temporal rule.

Preserve the frozen score and the exact decision formula:
`definite_zones > 0 OR (raw.alert AND score >= threshold)`.
Threshold is one inclusive scalar in [original_threshold,1], shared by all
frames/scenes. Score is a ranking convention, not a collision probability.
UNKNOWN and raw observations remain unchanged; silence never means clear space.

First count baseline false alerts by definite bypass versus score-gated path,
separately for OUTSIDE layouts and INSIDE pre-entry negatives. This measures
controllability by this threshold, not a physical cause of the false alert.

Compute three fixed roles without fitting any score/model:
- Baseline: original Calibration threshold0.007085703945147101.
- Event/onset ceiling: largest threshold retaining every baseline Core event
  and the baseline first in-event alert. With the unchanged definite bypass,
  this is the minimum score of required first-alert frames without a bypass,
  or1 if none. Report the binding frame identities.
- Full-Core-frame control: largest threshold retaining all baseline Core TP
  frames, using the analogous minimum. This stricter diagnostic shows whether
  any benefit requires later positive-frame losses.

For context, enumerate only the exact distinct decision sets of this fixed
score above the original threshold (inclusive scores and nextafter boundaries,
plus1). Retain tied frames together. The curve is descriptive consumed-data
headroom; do not pick a separate threshold per family or use boundary outcomes
to optimize the Core criterion. No second score, thresholds on other features,
gap filling, RGB reopening or temporal successor is included.

## Accounting and stop

For each role report Core events and per-event onset; every lost positive frame;
each event's alerted fraction, longest contiguous silent run (including trailing
silence), and all event identities. Report TP/FP/FN, precision/recall/FPR,
false-alert segments and sampled duration separately for OUTSIDE and INSIDE
negative periods, and full432/BOUNDARY costs. Segment masks retain full timelines.
Event-internal first-alert retention does not establish accurate onset: retain
pre-existing-alert counts and clip-first times. Sample bins are0.2s, not latency.

The diagnostic ends with the size and costs of the available headroom. A lower
FP count alone does not establish a preferred policy if events become mostly
silent or false segments increase. No numerical tolerance for those costs is
invented here. A practically useful candidate must subsequently be frozen and
evaluated on new complete layouts before any transfer claim. This diagnosis
does not silently replace the baseline or turn this consumed cohort into that
validation. If event/onset-preserving headroom is negligible, close this scoped
route and do not relax the constraints or add another threshold sweep.

Store input hashes, frozen protocol, complete curve, decisions and source IDs in
`artifacts.local/work/ba-core-workpoint-20260920/`. Scalar/JSON work uses CPU
(`TASK_NOT_GPU_SUITABLE`); no persistent processes or paid allocation are needed.
