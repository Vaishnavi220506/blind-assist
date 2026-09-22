# MZ177 continuous approach protocol

This round asks whether the frozen public pipeline gives a timely, stable reminder while a wearer approaches a thin rod, a suspended high obstacle, or a body sized obstacle. It is a controlled UE Development probe; it is not hardware, natural-distribution, user-benefit, or safety evidence.

The consumed 288-frame work point is closed. The corrected alignment audit shows that unchanged A* remains 96/3/12; the two Oracle support interventions yield 93/3/15 at the same threshold. A separate B2 diagnostic happens to have the same totals. See `MZ177_WORKPOINT_ALIGNMENT_20260918.md`. No training, model refit, threshold sweep, posthoc rescue, or automatic successor is allowed in this round.

## Source freeze

`continuous_approach_source.py` freezes 24 sequences before capture:

- families: `thin_rod`, `suspended_bar`, `body`;
- processes: approach/collision direction, lateral side pass, stop/back, and pure camera head turn;
- two mirrored layouts per family/process;
- 80 nominal frames per sequence, `dt=0.1 s` (nominal 10 Hz), an 8.0 s sampling window with timestamps spanning 0–7.9 s;
- one target plus a distant context wall, with fixed 640x360 RGB, 64-zone dual-return ToF, four Radar slots and causal IMU.

The target enters the alert corridor during approach and stop/back, stays outside the corridor during side pass, and remains a stationary corridor obstacle during head turn without wearer translation. Head-turn sequences measure stationary occupancy and output transitions, not successful prediction of a future collision. Source corridor/contact labels are evaluator-only and are frozen before UE output.

The existing MZ115 adapter is reused. Its 24 render warm ticks per nominal frame are retained for capture readiness; nominal timestamps are not a measured transport rate. The planned capture attempt budget was one. Actual execution required mechanical recovery: inherited 600 s timeout stopped the first process at 526 frames without JSONL; the second stopped at 728 durable rows without a final receipt. Only 720 complete-episode rows are retained from the second process. Remaining episodes resume from their original seeds, with no source/model/threshold change. This is explicitly a recovered Development capture, not compliance with the original one-process plan. Failure directories and `execution-recovery.json` are retained.

## Frozen predictor comparison

The public replay consumes only `raw.jsonl` and its RGB files. The evaluator JSONL and source specification are never opened by the predictor process.

1. **A***: sealed `A-star.pkl`, 2485-feature public frontend, threshold `0.5568065433174727`.
2. **raw HGB**: sealed `raw-final.pkl`, 2224-column raw subset of the frozen representation frontend, threshold `0.4874581810097215`.
3. **A* hysteresis**: the same A* score and model; turn on at the frozen A* threshold and turn off at `threshold - 0.02`. State resets at every episode. The 0.02 gap is fixed before the replay and is not tuned from this capture.

Missing or invalid sensor support remains explicit in the public frontend. An absent alert is not converted to a clear-space label by the predictor.

## Evaluator timeline and metrics

The evaluator joins rows by frame ID and uses only native bounds/private ToF lineage for offline attribution. It must preserve `UNKNOWN` when a native target attribution or timing boundary is unavailable.

For each sequence and method, save the complete timeline:

`target first visible -> first valid target-attributed ToF -> first alert -> contact/nearest pass -> release after stop/leave`.

“First visible” is an evaluator-only projected native target-bound visibility proxy, with left censoring when present at frame zero; it is not a measured RGB visibility onset. “First valid ToF” requires an actual public returned slot and received packet, attributed to the target through evaluator-only returned lineage. Private native ray hits are reported separately. Contact and nearest pass use native target bounds against the predeclared body envelope/corridor geometry. Risk accounting uses corridor OR contact so the corridor's inner cutoff does not label an imminent contact as clearance. These labels never reach the predictor.

Report, by family and process and for every sequence:

- complete-event misses, including sequences with no alert;
- first-alert time and target forward clearance at that frame, or `UNKNOWN`;
- first correct risk-interval alert and advance warning before contact, separately from an alert at/after contact;
- side-pass false-alert segment count and total alert duration;
- stop/back release delay after the final contact/corridor interval, or `UNKNOWN` when the opportunity is censored;
- frame/event coverage and `UNKNOWN` counts.

All sequences start with a geometric projection opportunity, so visibility onset is left-censored. The stop/back design leaves only 0.1–0.2 s of final non-risk samples. A first observed release in that window is reportable; stable release is not established. Do not relabel the preceding approach or stationary hazard as false merely because the wearer subsequently retreats.

Do not average only successful detections. Aggregate counts retain missed and censored sequences, and per-sequence timelines remain the primary evidence.

## Decision boundary

This round only chooses whether one of the three frozen output mechanisms is worth a single follow-up mechanism probe. A lower false-alert count alone is insufficient: alert precision, recall, contact timing, side-pass discrimination, release behavior, native target-attributed ToF retention and `UNKNOWN` coverage must be reported together. A negative or incomplete result closes this probe without threshold tuning.
