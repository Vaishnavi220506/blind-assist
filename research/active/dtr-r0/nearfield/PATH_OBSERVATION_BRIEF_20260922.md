# Fixed-path observation and quantization mechanisms: frozen brief

Date: 2026-09-22. Phase: EXPLORE, controlled same-generator synthetic Development.
This brief freezes ideas 1 and 2 before the new cohort is observed. No training,
shape-bank expansion, sensor change, quantization sweep, or runtime promotion.

## Shared source and fixed costs

The root experiment supplies 90 boundary pairs / 180 unique scenes: sides +/-1,
widths .07/.19/.37 m, depths .95/1.45/1.95/2.45/2.95 m, and boundary margins
.007/.017/.027 m. Box thickness .04 m, wall depth 4.2 m, eight fixed rays,
range quantization .1 m, and the existing full-extent world-fixed query remain.
No result of this new source may choose a path, threshold, or attribution rule.

Primary trajectory for both ideas is a fixed +X straight .12 m movement.
Endpoint baseline stores origin and endpoint (2 views / 16 ray readings);
path sampling stores origin and every .01 m through .12 m (13 views / 104 ray
readings). Thus travel remains .12 m, with 11 additional views / 88 readings.
No physical frame-rate, motion duration, pose accuracy, or latency is established.
All four axis spokes are frozen secondary diagnostics. Diagonal observations
belong to the separately specified two-step experiment and are unused here.

Before any new cohort observations, root and mechanism author changed the initially
proposed quantization primary from +Z to +X on geometric grounds: the paired front
faces share depth, while a lateral-face radial range is invariant to +Z motion.
Lateral +X motion can change lateral-face range while preserving face identity.
This is a pre-observation mechanism correction, not a scored-path selection.
+Z is retained as a structural negative / secondary diagnostic.

## Idea 1: information along the path

Compare complete quantized sequences, retaining all endpoint information.
Report separation over all 90 designated pairs and over initially identical pairs,
and label-purity over the full 180-scene cohort. A separated designated pair is
not necessarily distinguishable from every other opposite-label scene.

PASS requires at least one newly separated pair AND one newly pure cohort scene
under the fixed +X path. Improvement on only one measure is PARTIAL; neither is
NO_INCREMENTAL_INFORMATION. These are finite-cohort information gains, not an
implemented recognition accuracy result. Pair separation and pure-scene support
cannot decrease because the sampled signature retains both baseline observations.

Report four-spoke per-pair and common-initial-observation-class oracle ceilings
for endpoints and sampled paths. Also report full-cohort opposite-label purity
ceilings with one common spoke per initial class. All such choices use evaluator
labels/outcomes and are explicitly not executable policies.

## Idea 2: quantization crossing rather than visibility changes

For a pair aliased at both endpoints but separated along the path, attribute a
strict quantization gain only if at least one separating ray satisfies ALL:

- Raw ranges differ by more than 1e-12 m at origin or endpoint on that same ray.
- The first-return identity is the same for both members at every sampled pose,
  and never changes along either member's path.
- Identity includes wall versus box index AND exact rectangle face; corner/tied
  surfaces are rejected, with 1e-9 m face identification tolerance.
- Quantized sequences differ at intermediate samples and contain a bin change.

Raw ranges and geometry are evaluator-only attribution, never policy inputs.
Report each member's per-ray quantized transition brackets between adjacent
sample positions; these are .01 m intervals, not exact crossing positions.
All other path gains remain NOT_STRICTLY_ATTRIBUTED, without pretending they
are proven visibility gains. Secondary spokes use the same frozen rule.

The fixed +X primary passes this mechanism check only with at least one strict
crossing gain. No such gain means this specified mechanism is unsupported here,
even if path sampling helps through another mechanism. The two ideas share data,
cost, and potentially cases; their gains must not be added or treated as independent.

## Execution and stopping

One shared cohort evaluation after source/code/protocol sealing; no re-score after
scientific changes. Mechanical fixes preserve failed receipts and their reason.
Fixtures test sequence retention, complete-cohort ambiguity, no hidden-pair class
selection, surface-switch exclusion, and quantized-input-only signature creation.
Small scalar geometry work uses CPU / TASK_NOT_GPU_SUITABLE. Root owns seals,
results, experiment receipts, integration, delivery and final stop.

Integration: `path_observation_analysis.analyze(pairs, sources, observations)`
accepts root rows and returns JSON-compatible `denominators`, `primary`,
`fixed_spokes`, `quantization_attribution`, `endpoint_oracle`, `sampled_oracle`.
Source rows are `{id,scene}`, pair rows `{id,members}`, observation rows
`{id,views:[{camera:[x,z],bins:[8 integers],raw_radial_m:[8 floats]}]}`.
