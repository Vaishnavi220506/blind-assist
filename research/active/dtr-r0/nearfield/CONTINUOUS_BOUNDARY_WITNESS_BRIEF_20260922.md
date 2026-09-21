# Continuous boundary witnesses and explicit UNKNOWN reasons

2026-09-22, EXPLORE. Frozen component implementation before any fresh-cohort
inference. This implements ideas 3 and 5 only: construct real-valued opposing
rectangles from public observations and report why an answer remains UNKNOWN.
It does not add a selector, train a model, or generate the fresh evaluation cohort.

## Model and public interface

`infer(observations)` accepts a list of records with exactly `camera` and `bins`:
known [X,Z] camera translation and eight integer quantized radial ranges. Allowed
poses are the original reference plus the four fixed 0.12m translations. No IDs,
source geometry, source labels, family, raw ranges, or future actual observations
are accepted. Root may call it on initial and fixed +X two-view signatures.

The declared continuous model is one axis-aligned rectangle with variable lateral
faces L,R and variable front F, fixed thickness .04m, wall4.20m; center X in
[-.90,.90], width [.03,.60], center Z [.60,3.40]. Original eight ray angles,
.10m rounding bins, fixed corridor X[-.30,.30], Z[.30,3.00] and full extent truth
are unchanged. This model still excludes multiple objects, noise and unmodeled
geometry; continuous parameters do not establish complete real-world coverage.

For each ray, face-intersection geometry becomes linear constraints. A target
range bin bounds the maximum of front-face and near-side entry distances;
missed rays use a disjunction for passing to either side of the rectangle.
Binary variables encode only these disjunctions, not sampled object hypotheses.
Separate MILPs seek an IN witness and an OUT witness. This is a constructive
continuous solver, not an expanded grid or bank. Strict bin upper boundaries,
misses and nonintersection receive a jointly maximized numerical interior slack.

## Frozen work budget and validity

Exactly two SciPy/HiGHS MILP calls per unique public signature, one per requested
label. Each has time_limit=1.0s, node_limit=10000, mip_rel_gap=0, presolve=True.
No retries, restart, alternate seed, bank fallback, parameter sweep or solver
outcome tuning. CPU TASK_NOT_GPU_SUITABLE. The shared interior slack lies in
[0,.0001]m in each constraint's stated coordinate units. This favors interior
witnesses and may miss boundary-only solutions. SciPy1.17.1 is available through
E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe.

Any returned candidate is normalized by deterministic 12-decimal rounding and
domain clipping, then independently forward-checked against every supplied bin
and its claimed full-extent label. Invalid candidates are rejected with an
explicit numerical-validation reason and no retry. A time/node limit, infeasible
status or missing/invalid candidate is never a proof of geometric nonexistence.
Even one valid witness is not a certificate that its label is unique. All outputs
remain UNKNOWN; no certified IN/OUT alert is introduced by this component.

When both witnesses exist, evaluate those hypothetical rectangles at the four
original action poses. Report which action separates the pair. If none does,
report only that these two specific witnesses alias under those actions, never a
universal impossibility claim. These hypothetical forecasts are not extra actual
scene measurements or an oracle action fed into a new selector. If fewer than
two witnesses validate, distinguish solver incompleteness from two-witness
ambiguity. Retain every solver status, duration, constraint/binary count and
validated witness, with explicit model and completeness limits.

## Validation and scope

Focused synthetic unit fixtures (not the fresh cohort) check public input
whitelisting, exact quantization, full-extent boundary semantics, valid witness
replay, proof-free handling of missing/numerically invalid solutions, and both
separable and four-action-alias UNKNOWN reasons. Root freezes/scored-runs the
fresh180-scene cohort once and reports its complete denominator separately.
No actual fresh-cohort source geometry is provided to this solver. Source and
code snapshots accompany that run; any mechanical correction preserves receipts
and the fixed scientific settings. Outputs belong under the existing ignored
artifacts.local/work/ba-observation-mechanisms-20260922/witness* tree.
