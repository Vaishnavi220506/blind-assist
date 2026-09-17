# Run the single public positive version

This research runtime loads one fixed A, one1537-parameter return head and one
positive threshold. A separately retrained HGB is included only as the matched
data-amount comparison; it never changes the A+head alert. There is no fold or
scene router, online fitting, depth model or evaluator file in the public entry.

Artifacts live under the checkout's canonical junction:

`artifacts.local/work/corridor-public-single-20260917/bundle/`

| File | Role |
| --- | --- |
| `A.pkl` | Original frozen2485-feature HGB, threshold0.3917890013717321 |
| `positive.pt` | Full1344-frame shared ToF return head |
| `config.json` | Model hashes and fixed positive logit threshold5.8390960693359375 |
| `A-retrained.pkl` | Separate same1344-data HGB comparison, threshold0.5568065433174727 |

The bundle authenticates weights on load. The full-data head is new fitting,
not a selected outer fold. Its one threshold
was chosen on1152 scene-held-out Development predictions before confirmation.
Development selection and new-configuration results have different authority;
see the [fixed protocol](PUBLIC_SINGLE_PROTOCOL_20260917.md) and
[confirmation report](PUBLIC_SINGLE_RESULTS_20260917.md) before quoting performance.

From `E:\linnan\linnan`, with the existing research environment:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe' `
  research/active/dtr-r0/nearfield/corridor_fusion_v1/replay_single_public.py `
  --bundle artifacts.local/work/corridor-public-single-20260917/bundle `
  --raw PATH_TO_CAPTURE/raw.jsonl `
  --rgb-root PATH_TO_CAPTURE `
  --output artifacts.local/work/YOUR_REPLAY/predictions.jsonl
```

Use a new output path. The command keeps the models resident across all rows,
resets causal IMU yaw at each episode, and writes scores/alerts/missingness per
frame. It reads public raw records and their RGB files; no `evaluator.jsonl` or
scene specification is required. Supported inputs use the existing BlindAssist
public64-zone/two-return ToF, Radar, RGB calibration and IMU schema; this is not
an arbitrary camera-only detector or an Android integration.

Output `alert` is the main A OR positive decision. `control` is the independent
retrained-A comparison. `state=UNKNOWN` with zero usable returns means the
positive branch contributes nothing; the main output retains A. A nonalert
does not certify the corridor is clear. No temporal filter is introduced.

The subsequent [saved-output complementarity check](ASTAR_POSITIVE_POSTHOC_RESULTS_20260917.md)
prefers standalone A* (`control`) as the effect-version candidate: OR with
`positive` adds no clear gain, only one boundary event and one false frame.
That posthoc combination is saved separately; `alert` keeps its original meaning.
The current entry still computes comparison heads, so consuming `control` does
not imply an optimized A*-only runtime or transfer its latency to another path.

For confirmation replay with timing and native scoring use the separate
`run_single_confirmation.py --capture PATH_TO_CAPTURE`, followed by
`audit_single_positive.py --capture PATH_TO_CAPTURE` and
`audit_single_native.py --capture PATH_TO_CAPTURE`. That evaluator is intentionally
one-run-only and writes under the experiment's `confirmation/` directory; ordinary
public reuse should use `replay_single_public.py` above. Never overwrite a sealed
confirmation to retune the threshold or replace a result.

Cold model loading, public algorithm timings and capture/transport time are
different measurements. The benchmark warms on consumed input, includes every
new frame, and times RGB decode+A frontend/HGB+positive frontend/head. It does
not measure sensor acquisition, networking or phone performance. The comparison
HGB timing is reported conservatively including the original A head overhead.
