# MZ177 continuous approach results — 2026-09-18

24 controlled UE Development sequences / 1920 frames completed. Preserve frozen A* and raw HGB; no training, threshold sweep, residual, pixel-query, tail-rescue or surface-attribution successor was run.

## Decision

All three fixed methods warn before contact in all 12 moving contact sequences, and detect all 6 stationary head-turn occupancy sequences at least once. This does not establish stable reminders: A* loses 130/480 head-turn risk frames, including 106 frames with actual valid target-attributed ToF returns. These account for 106/127 of its supported false negatives overall. Head-turn A* has 93 alert transitions across six sequences; fixed hysteresis reduces that to 79 but still loses 98 supported head-turn frames. BODY supplies 72 of the 106 supported head-turn misses. Thus the next single candidate is **rotation-conditioned body-frame evidence retention/readout**, not an earlier-collision network or another surface model. This is a diagnostic choice, not a demonstrated causal cure; no successor is launched.

Timing is already early in these deliberately simple geometries. All A* approach/stop-back sequences first alert at t=0, before the corridor becomes positive; their actual onset is therefore left-censored. The table below distinguishes the first alert from the first correct in-risk alert. The latter provides 2.8–5.7 s nominal contact lead, not measured device latency. We cannot establish how much earlier the original alert began.

## Fixed comparison

| Method | TP / FP / FN / TN | Precision | Recall | Moving contact warned in advance | Side-pass alert segments / seconds | Target-supported FN |
|---|---|---|---|---|---|---|
| astar | 1086 / 59 / 298 / 477 | 94.85% | 78.47% | 12/12 | 9 / 1.2 | 127/971 |
| raw_hgb | 1065 / 65 / 319 / 471 | 94.25% | 76.95% | 12/12 | 16 / 2.3 | 154/971 |
| astar_hysteresis | 1100 / 59 / 284 / 477 | 94.91% | 79.48% | 12/12 | 9 / 1.2 | 118/971 |

Complete moving-risk misses: 0/12 for each method. Stationary occupancy ever-detected: 6/6 for each, reported separately from collision prediction. Side-pass alert sequences: A* 5/6, raw 6/6, hysteresis 5/6. Hysteresis adds 14 TP and no FP on this cohort but changes no first-alert time; it is a small Development continuity gain, not a new mechanism or deployment promotion. Frame counts are correlated observations, not independent events.

## Per-sequence timeline

Times are nominal seconds. V is projected-bound visibility opportunity (all left-censored at 0), T is first actual target-attributed valid public ToF return, C is first contact, N is nearest body-envelope pass. Each method cell is `first alert / first correct risk alert / pre-contact lead`; `—` is unavailable or not applicable, never zero. See the full HTML for all 80 frame states per sequence and release status for every method.

| Sequence | V | T | C / N | A* | raw HGB | hysteresis |
|---|---|---|---|---|---|---|
| thin_rod_approach_layout0 | 0.0 | 0.0 | 6.4 / 6.4 | 0.0 / 0.7 / 5.7 | 0.0 / 0.7 / 5.7 | 0.0 / 0.7 / 5.7 |
| thin_rod_approach_layout1 | 0.0 | 0.3 | 6.4 / 6.4 | 0.0 / 0.7 / 5.7 | 0.0 / 0.7 / 5.7 | 0.0 / 0.7 / 5.7 |
| thin_rod_side_pass_layout0 | 0.0 | 0.0 | — / 5.1 | 3.5 / — / — | 2.0 / — / — | 3.5 / — / — |
| thin_rod_side_pass_layout1 | 0.0 | 0.1 | — / 5.1 | 7.6 / — / — | 3.4 / — / — | 7.6 / — / — |
| thin_rod_stop_back_layout0 | 0.0 | 0.0 | 3.3 / 3.3 | 0.0 / 0.5 / 2.8 | 0.0 / 0.5 / 2.8 | 0.0 / 0.5 / 2.8 |
| thin_rod_stop_back_layout1 | 0.0 | 1.9 | 3.3 / 3.3 | 0.0 / 0.3 / 3.0 | 0.0 / 0.3 / 3.0 | 0.0 / 0.3 / 3.0 |
| thin_rod_head_turn_layout0 | 0.0 | 0.1 | — / 0.0 | 0.0 / 0.0 / — | 0.0 / 0.0 / — | 0.0 / 0.0 / — |
| thin_rod_head_turn_layout1 | 0.0 | 0.1 | — / 0.0 | 0.1 / 0.1 / — | 0.1 / 0.1 / — | 0.1 / 0.1 / — |
| suspended_bar_approach_layout0 | 0.0 | 0.0 | 6.3 / 6.3 | 0.0 / 0.6 / 5.7 | 0.0 / 0.7 / 5.6 | 0.0 / 0.6 / 5.7 |
| suspended_bar_approach_layout1 | 0.0 | 0.0 | 6.3 / 6.3 | 0.0 / 0.6 / 5.7 | 0.0 / 0.7 / 5.6 | 0.0 / 0.6 / 5.7 |
| suspended_bar_side_pass_layout0 | 0.0 | 0.0 | — / 5.0 | 7.7 / — / — | 5.6 / — / — | 7.7 / — / — |
| suspended_bar_side_pass_layout1 | 0.0 | 0.0 | — / 5.0 | — / — / — | 0.2 / — / — | — / — / — |
| suspended_bar_stop_back_layout0 | 0.0 | 0.0 | 3.2 / 3.2 | 0.0 / 0.2 / 3.0 | 0.0 / 0.2 / 3.0 | 0.0 / 0.2 / 3.0 |
| suspended_bar_stop_back_layout1 | 0.0 | 0.0 | 3.2 / 3.2 | 0.0 / 0.2 / 3.0 | 0.1 / 0.3 / 2.9 | 0.0 / 0.2 / 3.0 |
| suspended_bar_head_turn_layout0 | 0.0 | 0.0 | — / 0.0 | 0.0 / 0.0 / — | 0.0 / 0.0 / — | 0.0 / 0.0 / — |
| suspended_bar_head_turn_layout1 | 0.0 | 0.1 | — / 0.0 | 0.0 / 0.0 / — | 0.0 / 0.0 / — | 0.0 / 0.0 / — |
| body_approach_layout0 | 0.0 | 0.0 | 6.1 / 6.1 | 0.0 / 0.4 / 5.7 | 0.2 / 0.4 / 5.7 | 0.0 / 0.4 / 5.7 |
| body_approach_layout1 | 0.0 | 0.0 | 6.1 / 6.1 | 0.0 / 0.4 / 5.7 | 0.1 / 0.4 / 5.7 | 0.0 / 0.4 / 5.7 |
| body_side_pass_layout0 | 0.0 | 0.0 | — / 4.8 | 5.4 / — / — | 2.4 / — / — | 5.4 / — / — |
| body_side_pass_layout1 | 0.0 | 0.0 | — / 4.8 | 5.2 / — / — | 0.2 / — / — | 5.2 / — / — |
| body_stop_back_layout0 | 0.0 | 0.0 | 3.1 / 3.1 | 0.0 / 0.1 / 3.0 | 0.1 / 0.1 / 3.0 | 0.0 / 0.1 / 3.0 |
| body_stop_back_layout1 | 0.0 | 0.0 | 3.1 / 3.1 | 0.0 / 0.1 / 3.0 | 0.0 / 0.1 / 3.0 | 0.0 / 0.1 / 3.0 |
| body_head_turn_layout0 | 0.0 | 0.0 | — / 0.0 | 0.0 / 0.0 / — | 0.0 / 0.0 / — | 0.0 / 0.0 / — |
| body_head_turn_layout1 | 0.0 | 0.0 | — / 0.0 | 0.0 / 0.0 / — | 0.0 / 0.0 / — | 0.0 / 0.0 / — |

## Stability, support and release limits

A* risk FN by process: approach 51 (11 target-supported), stop/back 117 (10 supported), head-turn 130 (106 supported). The stop/back design actually reaches the declared contact envelope before stopping; it is not an independent safe stop-before-contact condition. Contact/corridor remain risk even while stationary. Do not retroactively count those warnings as false due to subsequent retreat.

The final non-risk window is only 0.1–0.2 s. A*/hysteresis show an observed immediate release in 2/6 stop/back sequences, with 4/6 still alerting at the last sample; raw shows 1/6 and 5/6 respectively. Stable release and mean population release delay remain UNKNOWN; the observed zero delays are not a success claim.

1217/1920 frames have valid public target returns, versus 1341 private-hit frames; 164 public packets are missing. Geometry, projected visibility and returned-lineage UNKNOWN counts are all zero for this controlled dataset, but absent returns never mean clear space. Thin-rod stop/back layout1 alerts before its first target ToF return at1.9s, so ToF-to-alert differences alone do not prove causation; RGB/Radar remain valid independent inputs. Head-turn has no translation or contact: its failures concern stationary body-frame occupancy under rotation, not demonstrated future collision discrimination.

## Evidence and validation

Artifact root: `artifacts.local/work/continuous-approach-20260918/`.

- `alignment-v2/`: unchanged A*96/3/12; Oracle support interventions93/3/15 lose the same three rod TP. See [workpoint audit](MZ177_WORKPOINT_ALIGNMENT_20260918.md).
- `capture-complete/receipt.json`: all24 episodes,1920 ordered IDs, source/time/RGB/JSONL hashes PASS. This is mechanically assembled evidence: 720 complete prefix frames plus1200 resumed frames. Prefix has no engine completion receipt; source and complete streams are authenticated during assembly. First526-frame timed-out run and incomplete8-frame tail were excluded. Three mechanical launches deviate from the original one-process plan; source/model/threshold unchanged.
- `baselines-v1/completion.json`: public-only inference PASS before native evaluation; zero training. Frozen CPU NumPy/OpenCV/sklearn interface (`FROZEN_PROTOCOL_CPU_ONLY`).
- `evaluation-v1/completion.json`, `summary.json`, `timelines.jsonl`, `timelines.html`: full evaluation and24 rendered timelines. Offline native geometry/lineage only, never model inputs.
- `runner-smoke-old-v1/parity-audit.json`: both models exactly reproduce all288 old probabilities.
-21 synthetic evaluator tests PASS, including return attribution, missing packets, UNKNOWN, contact timing, release censoring, sequence joins and sealed-input validation. Synthetic HTML contains24 sequence sections and144 tracks.

Nominal10Hz timestamps and 24 UE render warm ticks do not measure real-time transport or end-to-end latency. Visibility is an AABB projection proxy, not pixel visibility or occlusion proof. This is controlled Development with two layouts per cell, not natural/hardware/safety evidence. Retain A*/raw as comparators; no App default changes.

Registration remains pending because the existing experiment ledger line303 fails input_fingerprint validation. Do not repair or bypass that ledger. Intended disposition: retained frozen A*/raw core; hysteresis a Development component; head-turn instability a diagnostic supporting one unlaunched rotation-conditioned candidate. Structured registration/inheritance is not claimed complete; set-terminal-inheritance also returned unknown terminal id. Both tool errors are retained.

Cleanup: inventoried and hashed526 failed first-attempt images, then removed only that frame tree (118,699,790 logical bytes; not a measured disk-space delta). Final raw evidence, both retained source segments, failure logs, seeds, receipts and hashes remain. Resumed UE process tree reports released=true with no survivors; model/evaluator processes exited.
