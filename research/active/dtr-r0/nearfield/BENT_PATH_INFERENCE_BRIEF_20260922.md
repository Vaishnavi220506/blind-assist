# Fixed bent-path continuous inference brief

Date: 2026-09-22. Scope: a new disclosed synthetic Development path-geometry
comparison, prepared before its scored execution. This module does not generate
the cohort or run the experiment. The root protocol owns source and stress arms.

## Public input and sole change

`bent_path_inference.infer(observations)` accepts one to thirteen dictionaries
containing only `camera: [x,z]` and `bins: [eight positive integers]`. No scene ID,
source geometry, actual pose, stress label or truth is accepted. Exact nominal
pose membership is required in the union of these paths:

- `straight_x`: origin, then X=1 through 12 cm at Z=0.
- `x_then_z`: origin, X=1 through 6 cm at Z=0, then Z=1 through 6 cm at X=6 cm.
- `z_then_x`: origin, Z=1 through 6 cm at X=0, then X=1 through 6 cm at Z=6 cm.

Each full path has thirteen views and 12 cm travel. Their union has thirty
poses. This inference API admits any one to thirteen records from that union;
the experiment runner must verify actual path ordering, uniqueness and cost.
No other pose, noise tolerance or parameter-domain extension is admitted.

## Reused model and derivation boundary

The original single axis-aligned rectangle model is unchanged: center X in
[-.9,.9] m, width [.03,.60] m, center Z [.60,3.40] m, thickness .04 m, wall Z4.2 m,
original eight rays, .10 m radial quantization, full-extent reference query and
inherited EPS=1e-12. All new cameras have Z in [0,.06] m. Hence every allowed
rectangle front is at least .58 m and remains ahead of each camera; every box
back is at most 3.42 m and remains before the wall. The original signed lateral
slab, front-entry maximum, wall miss and bin disjunction derivation still applies.
The existing EPS-expanded hit-overlap envelope and IN query bounds are reused.
The slack-zero system is an outer closure, not an exact half-open-bin feasibility
test or a formal certificate. Numerical solver behavior remains material.

The wrapper executes the unchanged `path_constraint_inference._solve` function
body with a private globals dictionary whose sole substituted binding is the
instrumented solver callable. It does not change a shared module global, input
validator, old source file, objective, constraints or solver options. It invokes
the original IN and OUT solves once each, with `time_limit=1.0`,
`node_limit=10000`, `mip_rel_gap=0.0`, and `presolve=True`. No additional feasibility
solve or retry is allowed. Raw solver vectors and original receipts are retained.

## Frozen candidate recovery and readout

Keep every originally validated witness. Only a requested IN candidate with
solver status 0 and outcome `NUMERICAL_CANDIDATE_REJECTED` is eligible for the
already frozen `candidate_diagnostics.propose_projected_in` operation. It clips
the decoded center to the intersection of the model center domain and exact
closed query-membership intervals, retaining width and thickness. The original
scalar forward model must validate domain, label and every supplied bin after
projection. An invalid projection is discarded. There is no cache arm, candidate
sweep or new rounding/tolerance choice.

An `IN_MODEL_CONDITIONAL` or `OUT_MODEL_CONDITIONAL` decision requires a validated
witness for that label, its own original solver status 0, no opposite witness,
and opposite status 2 with the original exclusion-supported receipt. A witness
that conflicts with its class exclusion causes UNKNOWN. Limits, rejected
candidates and incomplete searches do not justify exclusion. Two opposing
witnesses mean constructive model ambiguity. Original receipts remain unchanged
even if candidate recovery succeeds; `original_witnesses` and `attempts` preserve
the distinction. No extra hypothetical four-action sweep is needed for this
path comparison.

## Stress interpretation

The caller may evaluate predeclared range bias or actual-pose offset while
passing nominal poses, as specified in the root protocol. Such arms are
`MIS_SPECIFIED_STRESS`: the nominal exact model does not gain a robust noise
envelope or a certificate. The module always reports
`NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY`, `robust_to_pose_or_range_error=False`,
`universal_uniqueness_proven=False`, and `sensor_clearance_certified=False`.
No perturbed result may be described as hardware safety, free space, robust
uniqueness or independent confirmation.

## Focused validation before scoring

Five tests passed in 0.013 seconds using deterministic mocked solver results:
path lengths/counts and exact public whitelist; original-fixture decision and
witness parity plus untouched shared globals and original budget; new off-axis
candidate replay; membership projection with mandatory label/bin validation;
and limits/exclusion-conflict fail-closed behavior. These tests execute original
constraint construction and candidate validation, but are not a scientific
cohort evaluation or real optimizer performance test. No pilot cohort has been
scored by this module's implementation agent. Existing model files are unchanged.
