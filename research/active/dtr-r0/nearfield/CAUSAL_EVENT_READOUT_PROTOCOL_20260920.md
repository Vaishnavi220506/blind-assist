# Fixed causal event readout on consumed scores

2026-09-20 EXPLORE. Test the user's rising-edge plus one-frame continuity
hypothesis on the two complete, already consumed Core432 cohorts. The previous
fixed-threshold failure remains unchanged. This is a new observable temporal
readout, not another threshold selection or fresh confirmation.

## Exact method and information boundary

Use T=0.4071309640537889, dt=0.2s; keep the original Calibration comparator.
For one anonymous scalar sequence, current score s, previous score p:

```
strong = definite_zones > 0 OR (raw.alert AND s >= T)
rise = previous_sample_exists AND raw.alert AND s < T AND s > p AND 2*s-p >= T
hold = previous_sample_exists AND previous_strong
combined = strong OR rise OR hold
```

Hold can bridge exactly one unsupported sample; only genuine `strong` refreshes
it. Neither a rising prediction nor a held alert can renew the hold. Thus a run
`strong, weak, weak` emits `alert, alert, silent` unless the third sample independently
qualifies as rise. Rise is the exact local extrapolation rule, not a separate
latched event. No cap, clipping, extra low threshold, slope threshold, hysteresis,
window, category/zone parameter, RGB, learned weights or label input is added.
The initial sample has no previous state. Reset on clip boundaries or any sampling
discontinuity. Clip IDs are grouping only; their text never enters the predictor.

The current raw gate applies to rising evidence; continuity may emit without
current raw support, explicitly attributed to one prior strong sample. Keep current
UNKNOWN unchanged: remembered/extrapolated alerts do not create independently
valid current support or certify clear space. Ground-truth entry is never used
to arm a hold or suppress pre-entry alerts.

Arms fixed before replay: Calibration, strong, strong+rise, strong+hold,
strong+rise+hold. The two single-mechanism arms explain contribution; no best-arm
selection or follow-up tuning. The predictor receives only scores/raw flags and
relative sample continuity. Seal complete predictions before joining evaluator
truth, layer, layout, geometry or previously identified missed-frame IDs.

Nonnegative scores imply `2*s-p <= 2*s`: a current score below T/2 cannot be
rescued by this rising rule even under the most favorable previous sample.
An isolated missed frame also need not follow genuine strong evidence. These are
falsifiable limitations, not reasons to change the proposed formula.
Score0 without possible support is a readout zero, not a zero range. Report rise
triggers after such missing support separately; max-score differences do not
establish that one physical surface is being tracked. Report consecutive rise
lengths: rise can independently fire repeatedly even though hold cannot renew.
Extrapolating a next-step score does not relabel current negative frames as positive.

## Reporting and decision

Primary consumed cohort: ba-core-workpoint-transfer-20260920 (432 total,
Core288=78P/210N); contextual regression: ba-core-transfer-20260920 (432 total,
Core288=72P/216N). Both include all negative periods and all boundary layouts.
Do not describe the former fresh source as fresh validation for this new method.

For every arm/cohort report TP/FP/FN, precision/recall/FPR, false segments and
sampled duration; OUTSIDE and INSIDE-negative separately, Core/Boundary/full,
BODY/HEAD and background. Include all added FP and recovered FN identities,
12 Core event first times, clip-first times, pre-entry ongoing alerts, per-event
alerted fraction and longest silence including tails. Record each temporal
trigger's source, previous score, extrapolation and previous strong state.

Necessary target-recovery checks on the primary cohort: retain12/12 Core events,
BODY36/36 and every strong alert; restore both previously delayed first in-event
alerts to the Calibration time; worst horizontal HEAD event reaches at least6/7
positive coverage (the preceding pilot's5/6 condition). Report recovery of all
four newly lost TP individually and any recovery of the baseline's own FN.
No numerical acceptable-FP budget is invented: report added FP/segments/time
against8FP high threshold, and remaining reduction against128FP Calibration.
Target recovery is necessary, not sufficient for promotion with arbitrary FP cost.
Boundary is reported separately, never used to pick parameters or veto Core merit.

If necessary target recovery fails, close this exact mechanism as a negative
control; explain which premise failed. If it succeeds, retain only disclosed
Development evidence and explicit FP tradeoffs, not a default or transfer claim.
One deterministic replay per cohort, no new capture, parameter sweep, automatic
connected-support/zone-mass analysis or successor. End with report and scoped
delivery. Data/figures/seals/receipts stay in
artifacts.local/work/ba-causal-event-readout-20260920/.
CPU scalar/JSON work is TASK_NOT_GPU_SUITABLE; no persistent processes needed.
