# Shared-bias continuous inference: frozen method

2026-09-22. Prepared before the root's scored replay. This is a new, bounded
synthetic Development measurement-model comparison. The two nuisance bounds
come only from the previously declared bent-path injector: radial bias
`b in [-.002,.002]` m and global translation `px,pz in [-.001,.001]` m. They are
not fitted to the 749 query results, hardware calibration or guaranteed sensor
limits. No source geometry, condition, actual camera, actual bias or truth enters
inference. Existing modules and sealed runs remain unchanged.

## Public API and budget

`shared_bias_inference.infer(observations, mode='range_pose')` accepts only one to
thirteen `{'camera':[nominal_x,nominal_z], 'bins':[eight integers]}` records from
the existing three-path, thirty-pose whitelist. Modes are exactly:

- `range_pose` (primary): shared b, px and pz within the bounds above.
- `range_only` (comparator): same b bound, with px=pz=0.

One shared nuisance triple applies to every ray and view in a history. There is
no per-ray correction, nuisance grid or per-view fitting. Each mode makes exactly
two label solves, IN then OUT, with the inherited `time_limit=1.0`,
`node_limit=10000`, `mip_rel_gap=0.0`, `presolve=True`, common slack in [0,.0001]
and objective maximizing that slack. No retry or extra feasibility solve follows.

The continuous variable order is `[L,R,F,slack,b,px,pz]`, followed by binary
disjunction indicators. The original single-rectangle geometry remains unchanged:
center X [-.9,.9], width [.03,.60], center Z [.60,3.40], thickness .04, wall Z4.2,
eight frozen ray angles, .10 m radial bins, and full-extent reference query.

## Constraint derivation

Let a public nominal camera be (cx,cz), ray direction `(ux,uz)` with uz>0, and
`a=ux/uz`. The ideal biased camera is `(cx+px,cz+pz)`. The actual replay camera is
`round(nominal+offset,12)` coordinatewise, exactly matching the prior injector.
An outward coordinate envelope `e=1e-12` m (one decimal unit, conservative relative
to the half-unit rounding plus binary representation error) is fixed before
scoring. This is a rounding-alignment allowance, not another sensor-error fit.

For an observed integer k use the inclusive outer closure
`lo=(k-.5)*.10`, `hi=(k+.5)*.10`. The true quantizer remains half-open and must be
checked by scalar replay; equality at hi is not by itself a valid witness.

For a target hit, let near_face=L for ux>0 and R for ux<0. Define
`tZ=(F-cz-pz)/uz`, `tX=(near_face-cx-px)/ux`. All allowed cameras, including
offsets and rounding, remain strictly behind the minimum target front .58 m,
so entry is `max(tZ,tX)`; the nonnegative-entry branch needs no new disjunction.
The following rows implement the measured-range closure with shared b:

```text
tZ + b + slack <= hi + e/uz
tX + b + slack <= hi + e/abs(ux)
(-tZ - b + slack <= -lo + e/uz)
 OR
(-tX - b + slack <= -lo + e/abs(ux))
```

For ux>0, overlap requires R >= cameraX+a*(F-cameraZ) and
L <= cameraX+a*(F+.04-cameraZ). Miss is the complementary strict OR; its closure
is R <= the front-ray X OR L >= the back-ray X. For ux<0 the roles are L at
front and R at back. Each ideal overlap bound is expanded by
`abs(ux)*EPS + e*(1+abs(a))`, where EPS=1e-12 is inherited from the scalar ray
contact comparison. Each miss bound is expanded only by `e*(1+abs(a))`; actual
misses already have separation beyond the inherited contact EPS. The code
substitutes px and pz with their proper signed coefficients in every row.

The wall branch uses `w=(4.2-cz-pz)/uz` and both inclusive bounds:

```text
w + b + slack <= hi + e/uz
-w - b + slack <= -lo + e/uz
```

It also requires the signed miss disjunction. Wall bins are never classified
using nominal fixed-wall-bin equality alone.

Branch selection follows the entire declared model envelope, not true return
attribution. Write P=.001 for range_pose and P=0 for range_only, B=.002:

```text
target_max = (3.42-cz+P+e)/uz + B + EPS
wall_min   = (4.20-cz-P-e)/uz - B
wall_max   = (4.20-cz+P+e)/uz + B
```

If [lo,hi] intersects the wall envelope it is a WALL branch. If lo<=target_max
it is a TARGET branch. A bin satisfying neither is explicitly inconsistent.
Their separation exceeds a whole .10 m bin for every permitted camera/ray, and
the implementation asserts that they cannot overlap. The target upper envelope
uses the farthest possible box back plus the inherited EPS contact extension.
Wall and box remain physically separated even under the declared nuisances.

IN uses the inherited expanded query-face bounds L<=.3+EPS, R>=-.3-EPS and
F<=3+EPS. OUT uses the inherited broader closure of R<-.3, L>.3 or F>3; it is a
safe outer superset of the strict EPS-aware negative class. The near-depth OUT
alternative remains impossible in the fixed geometry domain. All branch
indicators use bounded big-M disjunctions, with seven continuous variables
properly excluded from integrality. These are mathematical outer-closure
arguments, not a proof about numerical MILP termination or floating-point
certification.

## Witnesses, rejection and readout

A witness is `{'scene':<one-box scene>, 'bias':{'range_m':b,
'pose_x_m':px, 'pose_z_m':pz}}`. Geometry decoding retains the original normalized
12-decimal decoder. Nuisances are likewise rounded to twelve decimals and
clipped to their declared mode bounds. Full raw solver vectors, unrounded
scene/bias candidates, decoded candidates and validation diagnostics are saved;
normalization is never used as evidence that a candidate is valid.

Validation checks finite structure, geometry and nuisance domains, reference
query label, and every supplied bin. It calls the original scalar box-hit law at
the rounded actual camera, adds b to the resulting radial range, and then applies
the original quantizer. Diagnostics distinguish structure/nonfinite values,
geometry domain, bias domain, query-label and observation-bin failures and retain
per-view raw/biased ranges and mismatches.

An originally valid witness stays unchanged. Only requested IN, own status 0,
forward-rejected decoded geometry is eligible for the previously frozen exact
closed-query center projection. Width, thickness and all nuisance coordinates
remain unchanged. The proposal must pass the same full forward validation;
otherwise it remains rejected. No nuisance projection beyond initial domain
normalization, candidate sweep or extra solver call is introduced.

A conditional decision requires a valid own-label witness with original own
status 0 and opposite outer relaxation status 2/no candidate. Both-side
infeasibility, missing candidates, invalid candidates, limits and witness/exclusion
conflicts retain UNKNOWN. Every result carries
`NUMERICAL_SINGLE_RECTANGLE_SHARED_BIAS_MODEL_ONLY`; infeasibility is solver-reported
and is not a formal certificate, measured clearance or hardware guarantee.

## Interpretation and containment

The zero-bias exact model is contained in both nuisance models. Range-only is
contained in range-pose. More nuisance freedom cannot mathematically create
additional exclusions of the same label, although bounded numerical searches may
produce different candidates or failures. Saved zero-bias and range-only valid
witnesses must be replayed inside the larger models during postscore audit without
another MILP. Any numerical status contradicting a valid contained witness must
be reported, not promoted to additional certainty.

Global px can trade against L/R translation, while b and pz can be weakly
identified. A bias witness explains observations within this model; it is not
recovery of the true injector setting or sensor calibration. The full model also
allows simultaneous biases that were not combined in the prior injector arms.

## Focused tests and prescore state

Seven focused tests cover public-input rejection and shared nuisance domains;
analytic true-class row feasibility at slack zero for sixteen fixed geometry/
bias fixtures including query boundaries, extreme biases and round12-sensitive
offsets; range-only containment and seven-variable integrality; the changed wall
bin and impossible gap; half-open bin rejection despite closed feasibility;
exactly-two-call status gating and nuisance-preserving projection; and one new
two-view fixture executed through the real solver with exact witness replay.

Six tests initially passed. The half-open fixture used literal 1.45 whose binary
value fell below the intended bin boundary; its construction was corrected to
the quantizer's `(14+.5)*.1` expression. The affected test then passed. No model
formula, tolerance, bound or budget changed. The real fixture used exactly two
MILP calls and was not rerun for this test-only correction. All seven test cases
now have passing executions. No consumed 749-query scoring was performed by the
implementation agent; the root owns the sole frozen replay and its seals.
