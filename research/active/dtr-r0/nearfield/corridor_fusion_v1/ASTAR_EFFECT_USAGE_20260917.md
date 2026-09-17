# Standalone A* effect version

The standalone entry now runs the retained A* directly. Its **`alert` means
A* score >=0.5568065433174727**. It loads one HGB and uses the inherited public
RGB/ToF/Radar features and causal IMU orientation. It does not load or execute
old A, the positive-return head, Torch or DA-V2. No model was retrained.

This is the host research/showcase path. Android defaults and the original
`replay_single_public.py` contract are unchanged: in that older comparison
entry, `alert` still means old A OR positive and `control` means A*.

## Run on public capture records

From the checkout root, with the existing BlindAssist Python environment
(NumPy, OpenCV, scikit-learn and threadpoolctl):

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python `
  research/active/dtr-r0/nearfield/corridor_fusion_v1/replay_astar.py `
  --bundle artifacts.local/work/corridor-astar-effect-20260917/bundle `
  --raw PATH_TO_CAPTURE/raw.jsonl `
  --rgb-root PATH_TO_CAPTURE `
  --output artifacts.local/work/YOUR_NEW_ASTAR_REPLAY
```

Use a new output directory. `raw.jsonl` and its RGB files must use the existing
BlindAssist public 64-zone/two-return ToF, four-slot Radar, RGB calibration and
IMU schema. No evaluator, scene specification or label is required by this
entry. Missing ToF does not force a negative; A* keeps access to the other public
features. No-alert means **not reminding**, not a certified clear corridor.

The two-file bundle contains only `A-star.pkl` and `config.json`. Model identity
is checked on load: SHA256
`dffd5c5a2d6d59c5a0546513148b21736d5aa8601e5a034bcdab9337295cd59c`.
It is the same1344-frame seed185017 model retained in the prior experiment.

| Output | Consumer contract |
| --- | --- |
| `predictions.jsonl` | One frame ID, timestamp, score, fixed threshold, model ID and canonical `alert` |
| `reminders.jsonl` | Exact same `alert`, plus `reminder_onset` for rising edges within each episode |
| `summary.json` | Model load time, all-frame latency, alert/onset counts and runtime scope |
| `input-seal.json`, `completion.json` | Public input/model/source identities and output hashes |

Display badges and reminders must use the canonical `alert`; never infer a
second decision from status text, GT, a rounded score or the old comparison
entry's `alert`. Every episode resets causal yaw and the reminder edge state.
No smoothing, latch or hysteresis is introduced.

The optional `--warmup-raw` and `--warmup-rgb-root` use the first three frames
of a specified consumed source before timing. Without them, no frame is dropped
as warmup and the summary reports zero warmup frames. The delivered benchmark
uses three prior MZ170 frames, never selectively warms on confirmation outcomes.

## Delivered replay and measured parity

Artifacts: `artifacts.local/work/corridor-astar-effect-20260917/`.
`replay/` contains the full existing confirmation replay:288 frames,48 complete
episodes. `runtime-audit.json` confirms every score matches the sealed old
`control_score` bitwise; all alerts, clear/strict counts and event times match.
Both reminder flags and onset transitions follow the same new A* output.

| Evaluation | TP / FP / FN | F1 | Precision | Recall | Events |
| --- | --- | ---: | ---: | ---: | ---: |
| Clear216,75% coverage | 96 / 3 / 12 | 92.75% | 96.97% | 88.89% | 18/18 |
| Strict288 | 107 / 6 / 37 | 83.27% | 94.69% | 74.31% | 24/30 |
| Boundary72 | 11 / 3 / 25 | 44.00% | 78.57% | 30.56% | Reported within strict |

These are retained controlled same-simulator results, not a new accuracy
experiment. The independent executable preserves the known four old-A clear
TP losses, three rescued frames and one rod event delayed0.5s. All18 core
episodes start with an obstacle already present; event detection is not advance
warning. The prior [confirmation](PUBLIC_SINGLE_RESULTS_20260917.md) and
[posthoc OR comparison](ASTAR_POSITIVE_POSTHOC_RESULTS_20260917.md) are unchanged.

## Independent-path latency

One process, one resident A*, four CPU numerical threads,288 timed frames after
three consumed warmup frames. No slow frame is discarded.

| Measurement | Mean ms | p50 ms | p95 ms | Observed max ms |
| --- | ---: | ---: | ---: | ---: |
| RGB read/decode + frontend + A* | **66.67** | **66.04** | **74.64** | **85.00** |
| Public feature extraction + A* | 63.77 | 63.12 | 71.71 | 82.26 |
| RGB read/decode | 2.87 | 2.81 | 3.29 | 4.53 |

Cold model load is2.44s, separate from per-frame timing. Complete timing is
direct wall time around decode and prediction; component percentiles do not
add. JSON saving, video rendering, display/audio, sensor capture/transport and
phone execution are excluded. CPU is used because the unchanged NumPy/OpenCV
frontend and sklearn HGB have no corresponding GPU execution backend.

The66.67ms measurement is close to the prior66.59ms comparison-path timing.
They are separate host runs and the former removes extra heads; this does not
demonstrate a speedup. Do not subtract the2.20ms branch cost from either number.

## Complete A / A* demonstration

`demo/` contains the synchronized comparison produced from all288 saved frames
in original order, with complete BODY, HEAD, rod and boundary episodes and both
in/out members. Old A uses the original sealed `A` flag; A* uses the new
standalone `alert`. Native truth appears only as an offline scoring annotation,
never as an input or display decision. The demo includes successes and known
costs; no outcome-based clip selection or new capture is performed.

The supplied HTML player and MP4 are offline artifacts. The page keeps optional
reminder sound muted by default; A* sound uses the saved reminder signal. Its
episode navigation allows replay of complete configurations rather than
presenting a montage of successful frames. Playback speed is a presentation
control and does not alter prediction timestamps or measured latency.

Open `demo/index.html` with its sibling files present, or play
`demo/A-vs-Astar-all48.mp4`. The video has no audio track. The HTML adds optional
short tones on saved A* rising edges, reset per episode; high playback speeds
may skip tones because of browser scheduling. It does not send spoken alerts.
The video is 1440x900 at the source's 4Hz sample cadence,72s total. Each six-frame
episode includes a final0.25s sample hold; it is not a continuous walking run.

Reproduce the display from the saved predictions without running a model:

```powershell
python research/active/dtr-r0/nearfield/corridor_fusion_v1/render_astar_demo.py `
  --source artifacts.local/work/corridor-public-single-20260917 `
  --predictions artifacts.local/work/corridor-astar-effect-20260917/replay/predictions.jsonl `
  --out artifacts.local/work/YOUR_NEW_ASTAR_DEMO `
  --font PATH_TO_CHINESE_FONT --ffmpeg PATH_TO_FFMPEG
```

The renderer also requires Pillow and ffprobe alongside ffmpeg. Verification
checks all288 decoded frames,48 complete episodes, source order and prediction
identity. `frame-manifest.json` maps each video frame to both saved decisions;
`episodes.json` supplies navigation. Browser QA verified playback, pause, speed,
family filtering and episode seek using a localhost server supporting HTTP byte
ranges. Direct file navigation is blocked in the automated browser and was not
tested there. A server without byte-range support can play but fail to seek.
Optional sound was checked for signal mapping and control behavior, not audibility.

Reproduce packaging with `prepare_astar_effect.py` only if its dedicated bundle
does not already exist; it authenticates and copies the retained checkpoint.
Reproduce parity with `audit_astar_effect.py`. These are engineering extraction
and display checks, not another training, threshold-selection or confirmation
experiment. Durable weights, replay/metadata and original evidence are retained.
