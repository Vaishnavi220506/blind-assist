# Project state

Updated: 2026-09-07

BlindAssist is a runnable Android showcase research prototype and thesis project.
The primary goal is useful, measurable technical effect and a credible, stable
controlled demonstration. Innovation depth depends on the problem; major novelty
is not a requirement. Established methods, integration, and incremental improvements
are valid choices when their effect and cost justify them. Describe the actual
contribution honestly, and pursue new mechanisms when they address a concrete gap.
Natural-distribution and safety claims require their own evidence; a build or a narrow replay does not establish them.

## Current research lines

Forward-perception architecture correction (2026-09-13): the user-confirmed
simulation mainline is **one RGB camera + ToF + Radar + IMU**. Prioritize measurable
current-corridor alert benefit under the fixed cheap hardware budget. The
[MZ136 direct paired-training experiment](../research/active/dtr-r0/nearfield/MZ136_RESULTS_20260914.md)
does not improve the joint alert tradeoff; keep MZ129 and its independent native
evidence. Unique return attribution is not a required intermediate task.
The subsequent [TRAIN fit repair](../research/active/dtr-r0/nearfield/MZ136_TRAIN_FIT_REPAIR_20260914.md)
improves the same192 training frames from154 to191 correct with385new readout
parameters. Its frozen dev48 transfer needs20FP versus MZ12913 at matched recall
and timing, so retain it as a fitting diagnostic only. No new capture was used.
The [grouped readout follow-up](../research/active/dtr-r0/nearfield/MZ136_GROUPED_READOUT_20260914.md)
reduces head instability but still needs22FP versus13 at matched dev recall/timing.
The [boundary-input audit](../research/active/dtr-r0/nearfield/MZ136_BOUNDARY_INPUTS_20260915.md)
reduces edge MAE2.886 to1.227px on the same38 TRAIN frames, with fine edges
available38/48. Retain the geometry component only; MZ129 stays the alert baseline.
The [MZ137 end-to-end contrast](../research/active/dtr-r0/nearfield/MZ137_EDGE_CORRIDOR_20260915.md)
compares coarse/fine edges under one public-range plane on consumed dev48:
all arms24TP/13FP/0FN,5/5events unchanged. One corrected plane crossing never
changes the final alert; native support loss rejects this fixed integration.
MZ129 and the conditional MZ136 component remain; no tuning or test run follows.
MZ101--106 are a separate stereo+ToF branch, not evidence about this four-sensor
system. Follow the [four-sensor mainline](../research/active/dtr-r0/nearfield/FOUR_SENSOR_MAINLINE.md)
and its paired ToF+Radar+IMU versus +RGB comparison. Architecture changes require
an explicit new user decision; historical experiment suggestions do not change it.

| Line | Capability and present emphasis | Owning current |
| --- | --- | --- |
| `L10_R0_ACTIVE` | Recover and retain the requested target with useful evidence and observation cost; distinguish missing support, identity contradiction and endpoint extent. | [L10 current](../research/active/l10-r0/CURRENT.md) |
| `DTR_R2_DYNAMIC_RETAINED` | Current work: cane-complementary, class-agnostic forward obstacle awareness. Prioritize walls, body/head, suspended hazards and poles; ultra-low obstacles are secondary. DTR motion findings remain historical. | [Perception current / DTR history](../research/active/dtr-r0/CURRENT.md) |

These lines have independent evidence, budgets and decisions. Existing experimental
versions and detailed results belong in the owning current/ledger; this page does
not duplicate their trajectories. Uncommitted candidates do not change authority.

## Start and proceed

1. Read [current cross-route decisions](CURRENT_DECISION.md) and the affected route
   current; follow result/protocol/code links only for the present question.
2. Use [the research workflow](../research/WORKFLOW.md) to choose exploration,
   confirmation or engineering and the smallest check that changes a decision.
3. Implement and evaluate against a credible baseline. Report task effect together
   with relevant errors, UNKNOWN/coverage and observation or runtime cost.
4. Decide whether to retain, change, integrate or stop, then finish the remaining
   authorized delivery. Preserve historical results and release task-owned capacity.

## Demonstration and engineering

Semantic Anchor to Marker Pose remains a separate live-device showcase closure;
it does not transfer evidence to L10 or DTR or change their integration priority.

- Workstation entrypoint: `tools/ba.ps1`.
- Android builds: `scripts/run_android_gradle.ps1`.
- [Code ownership](CODE_MAP.md), [CARLA integration](CARLA_PLAYBOOK.md),
  [artifact routing](LOCAL_ARTIFACTS.md), [device evidence](DEVICE_REGRESSION.md).

## Evidence boundaries

- `UNKNOWN` and `NOT_EVALUABLE` are neither negative method evidence nor known-safe.
- `referent != affordance != waypoint != arrival != handoff`.
- Synthetic, replay, curated Development, registered-source, live-device and natural
  evidence retain their actual scopes. Disclosed reuse never restores freshness.
- [Formal governance](formal/RESEARCH_GOVERNANCE.md) applies to protected claims;
  it does not turn nearby reversible engineering into a final evaluation.
- [History index](history-index.md), owning ledgers/results and Git preserve history.

The full previous project narrative is retained at Git
`daf5720064d98a93b75336469d18e9a2fe0023e5:docs/PROJECT_STATE.md`.
