# Frozen shared-bias models under slow drift on new synthetic scenes

2026-09-22 EXPLORE. The user authorizes investigating whether shared constant
sensor biases remain useful when error varies during observation. Prior
[shared-bias results](SHARED_BIAS_RESULTS_20260922.md) remain frozen. This pilot
changes generated inputs only, not inference, bounds, decoding, budget or path.

## Question and fixed design

Can the two frozen constant-bias models keep useful decisions on new geometries,
and does slow drift yield UNKNOWN or wrong commitments? PRIMARY is range_pose;
range_only is an attribution comparator, misspecified for pose-error conditions.
Drift is outside BOTH declared constant-bias models: conditional model labels do
not certify the physical scene under that mismatch. No claim of safe abstention
is assumed. Nominal and in-model constant conditions are validity controls.

Use only x_then_z: 13 nominal views, X6cm then Z6cm, nominal length12cm. Domain,
wall4.2m, thickness.04m, eight rays and.10m quantization are unchanged. For drifting
pose, actual trajectory length differs slightly; record actual lengths rather
than claiming equal physical travel. No fresh capture or hardware evidence.

New deterministic seed2026092201 generates180 single-box scenes:

- 120 general-domain scenes: uniform x[-.9,.9], z[.6,3.4], width[.03,.60],
  rounded to9decimals, accept first60IN and first60OUT by analytic query truth.
- 60 boundary challenges: left, right and far query faces, ten IN/OUT pairs per
  face. Pair shares width sampled[.06,.50], coordinates and positive margin
  sampled[.0001,.001]m. Lateral faces use z[.8,2.8]; far face uses x[-.2,.2].
  Place the relevant full-box face at boundary plus/minus the margin, round9.

Reject exact duplicates and overlap with the consumed180source geometries, never
using predictions or visibility for selection. Separate results for general120,
boundary60 and each20case face. The balanced180 total is descriptive of this
designed cohort, not a natural prevalence estimate. Newly sampled instances use
the same simulator and law; this is new-instance Development, not independent
source/final confirmation. Generator deterministic replay is an audit only.

Thirteen conditions: nominal plus three families(range, pose, combined) crossed
with four temporal shapes(constant_plus, constant_minus, drift_up, drift_down).
At view i=0..12 define a=1,-1,-1+i/6,1-i/6 respectively. Range bias is a*.002m
for range/combined, zero otherwise. X and Z offsets are each a*.001m for
pose/combined, zero otherwise. Actual camera=round(nominal+offset,12); add radial
bias before the unchanged quantizer. Signs reverse the same temporal profile.
These match maximum amplitude, not RMS/error energy; drift has smaller RMS.
Combined conditions are a declared additional within-bound constant control and
outside-model drift stress, not a fitted error correlation or hardware model.

## Sealing, scoring and decisions

Seal protocol, code, frozen inference dependency hashes and old-source hashes
before generation. Save evaluator-only geometry, truth, raw ranges and actual
poses separately. Inference gets ONLY ordered nominal camera+bins and preset;
no ID, condition, stratum, truth, actual pose or injection values. Deduplicate
identical entire public histories, never individual views/partial histories.
Seal every prediction before loading evaluator data into scoring. Record hashes
and preserve old source/result payloads. Replaying saved predictions is allowed;
no repeated scored solves after result inspection.

Report full TP, FP, falseOUT, correctnegative, UNKNOWN, precision, recall and FPR
with denominators for all conditions/models/strata. Report exact nominal-correct
identity retention and transitions to wrong versus UNKNOWN. Compare each drift
to BOTH constant signs in its family, avoiding favorable endpoint selection.
Count histories/views/bins changed relative to nominal and both constants, and
report errors even if observationally aliased with another condition. Break
UNKNOWN into opposing validated witnesses, both-class reported infeasibility,
and incomplete/numerical. No aggregate-only safety or lossless claim.

Any wrong commitment in a full-model constant control triggers implementation/
numerical audit and rules out claiming sound retention; no scored repair rerun.
Any drift FP/falseOUT refutes safe-abstention behavior on that declared stress.
If drift only abstains, retain that finite observation together with coverage
cost; zero errors is not proof for arbitrary drift. If quantization hides all
changes for an arm, label its mechanism test uninformative, not robust success.
Audit every wrong case from saved geometry, measurements, witnesses and solver
receipts. For all outputs independently validate witness replay and metrics.

One run, max2340 public histories x2presets x2classes=9360MILP calls, dedup may
reduce this. Original1second/10000node per-class budget, no retries or tuning.
No new inference family, drift-aware rescue, noise-amplitude sweep, path search
or successor after scoring. Finish diagnostics, focused tests, report/current,
supported registration/inheritance attempts and scoped commit/push. CPU SciPy/
HiGHS TASK_NOT_GPU_SUITABLE; no paid or persistent resources. Outputs under
artifacts.local/work/ba-bias-drift-20260922/run-v1; refuse overwrite.
