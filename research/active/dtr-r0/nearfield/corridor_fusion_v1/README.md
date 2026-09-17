# Frozen depth corridor readout (E0/E1)

[Saved-score FP-budget diagnosis](RESIDUAL_BUDGET_DIAGNOSIS_20260918.md): at
Clear FP<=3/5, maximum TP is raw99/99, A*98/99, B088/91, B197/98, B293/95.
Residual event gains coexist with lost events and delayed rod alerts; full
maximum-TP plateaus are retained. No low-FP joint advantage to rescue through
threshold transfer. Keep the fixed recipe closed; no training or threshold change.

[Texture counterfactual training](BG_INVARIANCE_RESULTS_20260918.md): authenticated
background pairs reduce B2 held score drift versus B1 but do not improve alerts.
Old clear B2104/22/4 loses the low-FP tradeoff versus raw99/5/9; held B1/B2 alerts
are identical. Keep raw/A*, close the fixed recipe, retain source and drift evidence.
One288-frame capture,30inner HGB/21residual fits complete; owned resources released.

[Three-arm representation ablation](REPRESENTATION_RESULTS_20260918.md): matched
1344-frame HGB retraining gives raw/single clear99/5/9 versus multi96/3/12.
Multi-association superiority is not established; lower FP trades off recall
and boundary events. Keep A* unchanged. All21 fits and exact baseline replay
complete; consumed Development only, with native/event costs fully reported.

[Minimal CCRL result](CCRL_RESULTS_20260918.md): the fixed ranking residual
gives clear101/19/7,F188.60%, versus matched BCE101/10/7,F192.24% and
A*96/3/12,F192.75%. Ranking does not retain low FP or all event timing;
retain A*.672lateral pairs authenticate, but no exact nuisance pairs exist,
so invariance remains NOT_EVALUABLE. One consumed Development run only.

[Standalone A* effect version and complete demo](ASTAR_EFFECT_USAGE_20260917.md):
`replay_astar.py` loads one retained HGB; its `alert` drives storage, reminders
and display. All288 scores/events exactly match frozen A*: clear96/3/12,
F192.75%,core18/18; strict107/6/37,24/30. The complete48-episode A/A* demo
retains all successes and costs. Direct mean66.67ms,p5066.04,p9574.64 is measured
on the standalone host path; no speedup claim. Old comparison semantics remain
unchanged. This is engineering delivery, with no new training/capture/threshold.

[A* plus public evidence, posthoc](ASTAR_POSITIVE_POSTHOC_RESULTS_20260917.md):
exact saved-output OR leaves clear96/3/12,F192.75% unchanged. None of four
old-A true frames lost by A* is recovered; all four have zero usable ToF.
The combination adds one boundary TP and one FP, strict108/7/36 and25/30events
versus A*107/6/37,24/30. Prefer standalone A* for the next clear effect-version
candidate; retain the boundary-event tradeoff and public component separately.
No training/capture/threshold change; original frozen confirmation is unchanged.

[Single frozen version and new-configuration confirmation](PUBLIC_SINGLE_RESULTS_20260917.md):
one1344-frame public head/threshold and a same-data retrained A* are runnable
through the [public entry](SINGLE_VERSION_USAGE_20260917.md). New clear A/OR both
97/25/11; none of11 A misses has returned corridor support, so clear rescue
transfer is untested. The head adds1 boundary TP and1FP, +2.20ms mean algorithm
cost. A* gives96/3/12,F192.75%, core18/18, but loses4 clear A TP, delays one rod
onset.5s and strict events25/30to24/30. Retain the head component and A* as a
precision-oriented challenger with costs; no App change or post-result tuning.

[Public positive development v2](PUBLIC_POSITIVE_V2_RESULTS_20260917.md): reuse
1,344 existing frames; pooled inner scene-held-out calibration turns the unchanged
1,537-parameter BCE head into changed clear103/27/5 (F186.55%) without any
additional FP across four reporting cohorts. Old/MZ146/MZ158 remain A-identical.
The matched peak-loss arm adds one old TP but three boundary FP. Retain BCE plus
pooled calibration as a Development challenger; no App or fresh-confirmation claim.

[Public positive evidence head](PUBLIC_POSITIVE_RESULTS_20260917.md): implemented
1,537-parameter public-ToF MLP and six whole-scene fit/calibration/report folds.
Selected OR leaves A unchanged on both 288-frame cohorts. A single posthoc
logit0 diagnostic gives changed clear 103/33/5 (84.43%), recovering 8 FN but
adding 6 FP, with substantial boundary FP growth. Preserve the learned signal
and the calibration failure separately; no public final-method promotion.

[Existing-evidence tri-state probe](TRISTATE_EVIDENCE_RESULTS_20260917.md):
privileged positive OR gives clear 103/27/5, F1 86.55%; positive plus outside-only
veto gives 102/17/6, F1 89.87%. All zero-return frames retain A. Veto loses one
rod TP and delays its first alert by 0.75 s despite 50 usable background returns.
Retain spatial-readout headroom; outside-only is not certified free space.
This is an existing-data oracle diagnostic, not a public-input method result.

[Surface-support oracle](SURFACE_ORACLE_RESULTS_20260917.md): on existing new288,
replacing only A's per-return support endpoints with native full-face extents
leaves clear95/27/13 unchanged. A barely uses this feature seam. Eight clear
misses already have returned corridor points; five rod misses lack corresponding
returns. Complete extent adds zero clear reachability beyond sampled points,
and four boundary-only opportunities. No RSSF/CNH/model successor was started.

[Existing-data tolerance re-evaluation](TOLERANCE_RESULTS_20260917.md)
separates clear corridor decisions, strict boundary pressure and observed alert
behaviour. At5cm,75%of each cohort remains: old A107/22/1,F1 90.30%; changed
A=S195/27/13,F1 82.61%. All40remaining changed-domain errors are beyond10cm
laterally from the nominal boundary. No models/thresholds changed. Online
DA-V2 stays closed and the untrained intrusion pilot is deferred. Run
`run_tolerance.py` only for a new explicitly owned output; audit the existing
immutable result with `audit_tolerance.py`. No new source is needed for this
analysis. See [transfer diagnosis](TRANSFER_DIAGNOSIS_20260917.md).

This is the first bounded experiment requested in the 2026-09-17 obstacle
masterplan. It does not implement E2–E6 or alter the Android/default alert path.
See [protocol](E1_PROTOCOL_20260917.md) for cohorts and symmetric selection.

The user subsequently adopted [A as the balanced research baseline](BALANCED_BASELINE_20260917.md).
The separately authorized [conditional specialist](SPECIALIST_RESULTS_20260917.md)
retains a small consumed Development gain,137/28/7to137/24/7, without changing
A/C weights. Run `run_specialist.py --output <artifact-directory>` and audit
with `audit_specialist.py <artifact-directory>`. See its fixed
[protocol](SPECIALIST_PROTOCOL_20260917.md); no App promotion is implied.

Completed [E1 results](E1_RESULTS_20260917.md) and the subsequently authorized
[intermediate diagnostic](INTERMEDIATE_RESULTS_20260917.md) show no overall
increment. The latter uses `run_intermediate.py --output <artifact-directory>`;
its [protocol](INTERMEDIATE_PROTOCOL_20260917.md) fixes one token/PCA adapter.
`audit_intermediate.py <artifact-directory>` verifies saved outputs without fits.

The only learned task head is scikit-learn HGB. A receives the inherited 2485
public sensor/geometry features; B also receives 924 local relative-depth
statistics. ToF dual slots and Radar remain in both inputs. DA-V2's output is
never treated as metres, surface truth, or a hard sensor veto.

## Run

Use the configured research Python (Torch CUDA, NumPy, OpenCV, sklearn,
torchvision, threadpoolctl). Place the official
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) source in
`WORK/upstream` and its official `depth_anything_v2_vits.pth` in `WORK`.
WORK must resolve under the checkout's ignored `artifacts.local/` tree. Source
commit, checkpoint SHA256 and native/processed image shapes are recorded.
The inherited MZ136/MZ143/MZ170 local artifacts must exist; they are not bundled
with this source repository. No simulated data is generated by this runner.

```powershell
& E:/codex-tools/bin/blindassist-research-gpu.cmd research/active/dtr-r0/nearfield/corridor_fusion_v1/run_e1.py --work artifacts.local/work/corridor-depth-e1-20260917 --run-name run-v2
```

An existing completed run cannot be overwritten. Interrupted feature extraction
can reuse per-frame caches only with the same frozen source/config/input/model
identities. The process writes failure or completion evidence and exits; models
and arrays do not require a resident service. Generated data, logs and weights
are retained for reproducibility, not scheduled for deletion.

Outputs include selection and prediction seals, A/B models, feature caches,
per-frame CSV, PR curves, family/pressure metrics, event and release details,
native ToF contributor accounting, CUDA peak allocation, actual online latency
samples and a comparison video. No-alert means UNKNOWN. Native Radar lineage
is unavailable. A binary-only reference has no comparable ranking PR-AUC/AP.

Complete latency includes RGB decode, inherited feature extraction, DA-V2
preprocessing/forward/upsampling, depth statistics and HGB. B conservatively
also includes the tiny A-head call. It excludes physical capture/transport and
the inherited causal references' waiting time. The 24 measured frames are the
first complete episode per family, chosen without outcomes.

All evidence is consumed, same-generator controlled Development. The report
cohort is untouched by this run's HGB selection, but has been inspected in prior
research; this is not a fresh test or a real-world/phone performance claim.
