# Sensor-native temporal frontier: entry audit

The proposed additive topology is useful: `A* OR (raw HGB AND confirmation)`
preserves every original A* alert by construction. The supplied roadmap needs
the corrections below before its temporal-ceiling claim is testable. This is
an entry audit of consumed Development evidence, not a completed temporal
ceiling experiment or a negative result for temporal occupancy.

## Actual frontier

Saved ordered predictions reproduce clear A* **96/3/12** and raw **99/5/9**.
The disagreement contains **three added TP and three added FP**, while raw
also removes one A* FP. An additive rescue cannot remove that existing FP.
Consequently an unrestricted union is **99/6/9**; reaching **99/3/9** requires
accepting all three added TP and rejecting all three added FP. A net difference
of two FP must not be mistaken for two candidate false positives.

The three TP are near-rod scene1 in02/in03 and scene5 in03. In their causal
history windows of up to 1.0 second, none has a saved native corridor ToF
witness. Scene5 in03 has no usable ToF return anywhere in its entire available
window. Thus accumulating earlier native corridor ToF evidence does not
currently explain these particular opportunities. This does not rule out
Radar or wider spatial evidence, and absent returns remain UNKNOWN.

Radar packets do contain history for these candidates. Their existence alone
does not establish occupancy in the body corridor: persistent background can
also survive. No Radar attribution or temporal separability result is claimed.
The original generator moves target lateral position with camera lateral
position; this is controlled synthetic motion, not a natural stationary-rod
trajectory. Public IMU deltas supply rotation, not metric translation.

## Current route corrections

- [MZ176](../MZ176_RESULTS_20260918.md) already completed once. ALL added no TP,
  failed its gain gate and is closed without a consensus successor.
- [MZ177](MZ177_CONTINUOUS_APPROACH_RESULTS_20260918.md) already names the
  completed 24-sequence/1920-frame continuous diagnostic. Do not reuse its ID.
  Its A* head-turn misses include 106 frames with valid target ToF support;
  72 are BODY. This is a different, directly observed opportunity for a
  rotation-conditioned body-frame readout, not proof of the old frontier goal.
- Freeze A*/raw models and thresholds; keep L10 independent. Neither a new
  learned head nor a capture nor an App change is part of this audit.

## Scope of a meaningful next probe

Retain per-sensor facts and explicit availability. Associate past observations
using only information available by the decision time; publish a rotation-only
arm separately from a privileged pose oracle. Do not call IMU-only transport
full ego-motion compensation. A future observation may confirm a later alert,
but must incur that delay and cannot retroactively justify the earlier alert.
Do not demand Radar approach or two-sensor agreement from otherwise valid ToF
occupancy. Unobserved/invalid ToF is not a free ray; coarse footprint returns
do not certify all unreturned space as clear.

The old six clear disagreement frames provide only two positive and two
negative scene groups. A separator fitted or selected on these six examples
is descriptive, not a generalization result. Report causal feature coverage,
native-support retention, strict/boundary errors, event identity and delay,
in addition to clear TP/FP. Failure caused by missing translation or missing
observations is NOT_EVALUABLE for the full mechanism, not evidence that all
sensor-native temporal representations fail.

Given the observed evidence, first inspect causal Radar/spatial histories for
the old frontier and keep the continuous head-turn question separate. There
is no justification yet to claim the proposed five-dimensional features
separate the candidates or to launch a learned rescue head.

## Reproduction and limits

`python research/active/dtr-r0/nearfield/corridor_fusion_v1/audit_temporal_frontier_entry.py --output artifacts.local/work/temporal-frontier-entry-20260918/audit.json`

The delivered output exists, so use a fresh output filename for reproduction.
The audit verifies all 288 unique ordered IDs, public/saved episode and time
parity, strictly increasing episode times, frozen score/threshold decisions,
clear counts and the six candidate identities. It retains each candidate's
causal history plus input SHA256 hashes. CPU JSON accounting requires no GPU.
Native witness fields are reused from the prior saved audit; native geometry
is not independently reconstructed here. No temporal model or oracle was run.
No research terminal is assigned and no existing ledger is changed.
