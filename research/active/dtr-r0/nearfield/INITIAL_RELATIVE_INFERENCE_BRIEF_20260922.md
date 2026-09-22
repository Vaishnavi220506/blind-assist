# Initial-camera-relative inference: frozen method

2026-09-22. Prepared before the root's sole scored replay. This bounded synthetic
pilot changes the query reference frame in the frozen full shared-bias model.
It does not change the observation law, world geometry domain, nuisance bounds,
quantizer, solver budget, source scenes, or historical fixed-world labels.

## Public interface and reference frame

`initial_relative_inference.infer(observations)` accepts only thirteen ordered
`{'camera':[nominal_x,nominal_z], 'bins':[eight integers]}` records on the existing
`x_then_z` path: origin, six 1 cm X steps, then six 1 cm Z steps. The mode is
always `range_pose`. Source geometry, actual offsets, condition, IDs and truth
are not accepted. Other paths, incomplete histories and extra fields fail input
validation.

The query is fixed for the entire history in the actual **initial** camera frame,
with yaw zero. It is not the final or current camera query. For a candidate with
shared offsets `(px,pz)`, its actual initial camera is
`Q=(round(px,12),round(pz,12))`, matching the original injector at nominal origin.
Exact label validation subtracts Q from the full box extent before applying the
original corridor test, `abs(X)<=.30`, `.30<=Z<=3.00`, with its inherited EPS.

The world parameterization remains `[L,R,F,slack,b,px,pz]`, followed by binary
variables. The original center X [-.9,.9], width [.03,.60], center Z [.60,3.40],
thickness .04, wall Z4.2, radial bias +/- .002 and pose offsets +/- .001 m remain.
In particular, neither the world domain nor anchored wall is translated away.
This is not full gauge elimination, offset calibration, or an assumption that
px=pz=0. Co-translating a box and the reference camera preserves the relative
extent label when both resulting model instances remain admissible.

## Query rows and preservation of the observation model

The implementation constructs the original full-model constraints and replaces
only their query prefix. It asserts the original domain/query prefix and binary
index structure, retains every observation row and full variable bound, and
regenerates OUT big-M values for the added nuisance coefficients. No shared
module globals are patched or frozen source files changed.

Let `e=1e-12` be the existing predeclared round12 outward envelope and EPS be the
original scalar contact tolerance. The IN outer rows are:

```text
L - px <= .30 + EPS + e
-R + px <= .30 + EPS + e
F - pz <= 3.00 + EPS + e
```

The OUT outer disjunction is:

```text
R - px + slack <= -.30 + e
 OR -L + px + slack <= -.30 + e
 OR -F + pz + slack <= -3.00 + e
```

At slack zero these contain the corresponding strict-negative labels. The OUT
closure deliberately remains broader than the exact EPS label complement.
The near face is redundant because the minimum box back is .62 m while the
largest shifted query near boundary is approximately .301 m. The added e covers
Q differing from the continuous offsets; it is derived rounding alignment, not
a fitted measurement tolerance. Inclusive constraint closure does not replace
the exact original half-open bin replay or exact relative-label check.

## Candidates, readout and budget

Each query makes exactly two original-budget MILPs, IN then OUT, with
`time_limit=1.0`, `node_limit=10000`, `mip_rel_gap=0.0`, `presolve=True`, and the
original slack objective. There are no retries, nuisance grids or additional
feasibility solves. Original raw vectors, unrounded/decoded candidates,
validation details and solver receipts are retained.

Every candidate retains the old structure, world-domain, nuisance-domain and
exact biased forward-bin checks; only its query label check uses Q. An existing
validated candidate is retained. For a rejected IN candidate with own status 0,
the one inherited deterministic recovery projects its center, with fixed width,
thickness and bias, onto these closed intervals:

```text
X: [-.9,.9] intersect [Qx-.30-width/2, Qx+.30+width/2]
Z: [.60,3.40] intersect [Qz+.30-thickness/2, Qz+3.00+thickness/2]
```

The proposal must then pass all exact bins, domain bounds and relative labels.
A rejected proposal stays rejected. This projection supplies a possible witness;
it does not change exclusion constraints or establish missing-side absence.

`IN_MODEL_CONDITIONAL` or `OUT_MODEL_CONDITIONAL` requires the selected label's
validated witness with own solver status 0 and the opposite outer relaxation
reported infeasible with status 2 and no opposing witness. Other outcomes,
including limit statuses, candidate rejection, exceptions and conflicts, remain
UNKNOWN. Authority is numerical single-rectangle/shared-bias/initial-query model
only: there is no formal infeasibility certificate, universal uniqueness,
hardware error-bound calibration, or sensor-clearance claim.

## Focused validation and experiment boundary

Six tests passed before scoring, using hand fixtures, analytic row assignments
and mocked solver results only. They cover the exact public path, co-translation
label/bin invariance, unchanged observation rows and budgets, lateral/far query
rows, round12 alignment, shifted projection with bin rejection, preserved world
bounds, and conditional/UNKNOWN status handling. No source-cohort solve was run
while implementing this module.

The root's prospective replay uses the existing 375 public histories and 750
calls, sealing predictions before evaluator access. New truth is relative to
each history's actual initial camera; old fixed-world truth remains available
under a distinct field. Nominal comparisons share the same truth. Applying old
fixed-world decisions to new relative truth is a semantic diagnostic, not a fair
algorithm gain. Constant offsets are in model scope; varying pose/range drift
does not gain a soundness guarantee from this query correction. This initial,
yaw-zero pilot does not implement the broader current-camera contract.
