# MZ133: angular information versus readout, one bounded Development contrast

User authorization 2026-09-14: execute proposal audit, angular-resolution
contrast, and small joint-attribution exploration. This is a new question, not
a retry of MZ131 or MZ132. Those consumed terminals remain unchanged.

Question: does finer finite-footprint angular sampling reduce off-corridor
warnings with the same simple geometry, physical angular domain and corridor?
No training, weight/threshold sweep or hardware purchase.

Source: all 288 consumed MZ123 frames, same scene geometry, camera, packet loss,
observed IMU yaw and frozen MZ129 Radar/RGB/flow branch. Geometry generator reads
source cubes and reflectances; predictor receives only public zone packets and
the frozen observable yaw/Radar bit. Source identities, ray hits and labels stay
in evaluator artifacts. No interpolated old ToF or true points in prediction.

Admission: regenerate legacy 8x8, 3x3-per-zone first hits analytically and compare
every ray's hit availability and range against the saved native UE trace. Range
tolerance is 1e-5 m, fixed before execution. Reproduce old public ToF outputs
using the original histogram/noise/seed model; exact status/count/quantized-range
agreement and <=1e-7 absolute signal discrepancy are required. If admission fails,
retain the failure and diagnose geometry; do not present it as method evidence.
This is an admitted analytic cube-scene capability test, not a new UE capture.

Fixed arms, saved before scoring:

- legacy8: 8x8, original 3x3 integration, original public model, simple readout.
- dense8: 8x8, 12x12 integration, normalized zone signal, sigma .04 m.
- fine32: 32x32, 3x3 integration, normalized zone signal, sigma .04 m.
- budget32: same fine32 rays, signal scaled by 1/16 before .01 detection floor;
  range sigma .16 m (fourfold shot-noise stress assumption).

Dense8 and fine32 integrate the identical 96x96 full-FOV lattice. Integration
weights sum to 1 per zone in normalized arms. They retain the same .05 m bins,
.60 m peak merging, .10 m spread, two-target cap and .02 m quantization. Noise
is fixed by episode seed and deterministic zone order; dimensions alter draw
assignment, so this is a fixed-seed contrast, not an uncertainty interval.
Budget32 is a disclosed hypothetical sensitivity bound, not a calibrated photon
or ST model. Signal and sigma choices are not tuned to outcomes.

Readout: full 45x45-degree FOV, central horizontal [-11.25,11.25] degrees have
unit weight (four coarse or sixteen fine columns), alert threshold 1. Preserve
the all-FOV certain-support guard. Every return retains its full zone; no RGB
contraction. SIM_MERGED retains .02-4 m; SIM_VALID uses measured +/-3 sigma.
Same interval geometry, pitch +/- .5 degrees, observed yaw +/-(.5+.2*time),
body corridor x .2-3.6, y +/-.3, z .4-2.05 m. Report ToF-only and ToF OR the
unchanged independent MZ129 Radar branch, plus original MZ129 as context.
The matched simple8 is the resolution comparator, not an assumed MZ129 replica.

Acceptance: report TP/FP/FN, all-family totals, UNKNOWN, positive-event first
alerts, false segments, detected native contributor coverage, status and signal
counts, runtime and prediction/source hashes. Preserve every boundary frame.
Distinguish alert retention from actually hitting the hazard. No-alert remains
UNKNOWN and counts as FN where truth is positive.

A useful normalized gain is fewer FP with no lost TP or later/lost events;
retain only as controlled capability evidence. A loss or signal-budget collapse
prevents hardware/performance promotion. No gain under fixed MERGED/pose readout
does not prove resolution generally useless; diagnose which return states remain.
Stop after these fixed arms and mechanical fixes. Any further new-source final
confirmation needs its own unchanged method and frozen source protocol.
