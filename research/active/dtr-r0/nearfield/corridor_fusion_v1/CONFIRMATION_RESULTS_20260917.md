# Frozen S1 targeted confirmation: source not evaluable

**SOURCE_NOT_EVALUABLE_CAPTURE_TIMEOUT.** The single authorized capture stopped
at its frozen1200-second limit with235/288 RGB frames. No complete public sensor
records or evaluator file were finalized. No new A/S1 inference or performance
evaluation ran. This is an acquisition failure, not evidence for or against S1.
The prior consumed Development gain137/28/7to137/24/7 remains unchanged; fresh
confirmation is still outstanding. No gate tuning, recapture or CNH successor.

[Frozen protocol](CONFIRMATION_PROTOCOL_20260917.md),
[prior S1 result](SPECIALIST_RESULTS_20260917.md),
[balanced A](BALANCED_BASELINE_20260917.md).

## What was frozen and checked

Code commit `2bd1632391e42c51ee7e5694b6564090da31880e` contains the source,
protocol, preparation wrapper and public-only conditional runner. A/C/PCA/gate,
thresholds and inference dependency hashes are unchanged. CPU cached parity
reproduces all288 prior A/C/gate/final/invoked/veto values bitwise, without token
execution or fitting. Five runner tests and three source tests pass. Independent
source review confirms all-object labels and no source metadata in gate inputs.

The planned source has24independent configurations,48paired episodes,288frames,
144source-positive/144negative and30positive events. Each family covers background
front distances3.3/3.6/3.8/4.0/4.2/4.5m. Backgrounds comprise8portals,5staggered
panels,7recessed openings and4full walls. Near backgrounds retain true corridor
openings; full near walls would invalidate negative labels. Paired frames change
only target lateral position. Width, reflectance and texture grids/seeds vary.

The4m simulated ToF maximum means4.2/4.5m physical backgrounds may have no direct
return. Wall distance is not q100; above-threshold positive coverage remains
unmeasured. Range/topology/width/reflectance co-vary, so this is a targeted stress
design rather than a causal factorial. Appearance uses grayscale unlit texture
and hypothetical reflectance, not measured material BRDF. Same-renderer source
disjointness cannot establish natural-distribution or hardware effectiveness.

## Acquisition outcome and cause

The registered worker started17:40:06 and terminated18:00:09 Hong Kong time on
2026-09-17. Its terminal is `failed`, exit1, `TimeoutExpired(...,1200)`. The final
progress receipt reports235frames and720.735seconds inside capture. Process
lifetime including cleanup is1203.238seconds. Logs show first-start shader
compilation; sampling began around eight minutes after launch.

This task reused the older capture's1200-second overall allowance without
adequately accounting for the more complex backgrounds. Source-derived average
per-frame object plus texture-tile count rose94.25to192.17 (new range90–276).
Startup compilation and slower per-frame scene construction exhausted the
allowance. This is an execution-budget estimation error. It does not justify
changing a model, selecting easier scenes, or silently extending a frozen run.

The capture engine finalizes raw/evaluator/manifest/receipt only at completion.
All four are absent in this terminal. The235 RGB images alone cannot establish
paired sensor performance or truth; no partial-denominator scores are reported.
New TP/FP/FN/F1, invocation, called-TP/FP discrimination, event/onset, latency,
q100 and physical-ambiguity outcome tables are therefore **NOT_EVALUABLE**.
The implementation for those measurements exists but has not run on new data.

## Delivery and retained evidence

Controller artifacts: `artifacts.local/work/corridor-depth-confirmation-20260917/`.
The immutable runner freeze, source/spec/audit, full bundle, exact prior parity,
failure transfer receipt and returned logs are retained. The failure ZIP SHA256
is `801e8907dac5c438e1998633e3c4fea5caf8621d745011e8b3285a57c3522f44`;
19original receipt/log/source files were hash-verified after transfer. The partial
RGB manifest retains235hashes and53,031,275bytes on the worker. No media download
was needed to make this source-failure decision.

Worker owner: this S1 confirmation. Its durable payload remains at
`G:/DevWorkspace/BlindAssist/artifacts/work/corridor-depth-confirmation-20260917/`;
job receipts are under `artifacts/evidence/jobs/s1-confirmation-capture-20260917`.
These files are diagnostic evidence, not a resumable admitted dataset. No active
job remains. Thirty tracked process identities exited, no survivors; port26062
and the owned scheduled task are absent. The task-owned reproducible DDC was
removed after release verification:453,315,479logical bytes (~432.32MiB).
Raw RGB, logs, hashes, model inputs and failure receipts were preserved.

Disposition: preserve **S1 COMPONENT_OR_CHALLENGER / CHALLENGER**, awaiting an
unchanged-method confirmation. The source-failure terminal is a diagnostic
negative control for the1200-second capture allowance, not a negative control
for S1's algorithm. Existing ledger303input-fingerprint error still prevents
supported experiment registration; structured terminal assignment remains
pending, with no manual ledger bypass.

A later explicitly scoped acquisition could keep this exact method/design and
use an engineering allowance that covers observed startup plus all288frames;
that would be a new recorded capture attempt, not a continuation of this frozen
terminal. It was not started here. A remains the balanced research baseline;
App defaults and prior positive/negative scientific terminals remain unchanged.
