# Multiple explicit shape hypotheses: bounded synthetic feasibility

EXPLORE, 2026-09-21. Newly authorized mechanism experiment; no old frozen fit,
whole-target oracle, consumed camera-forward cohort, or Android behavior changes.
Question: can a finite set of plausible shapes retain genuine decision ambiguity
that a single best fit hides, and where does an incomplete prior still fail?

## Fixed source and assumptions, before scored execution

Camera coordinates are X-right/Y-down/Z-forward, identity extrinsics, pinhole
horizontal FOV 45 degrees, 64x48 square-pixel image. Corridor is the closed box
X[-.30,.30], Y[-.30,.60], Z[.30,3.00] metres. Y is an explicit synthetic profile,
not a measured mounting profile. Primitive solids are axis-aligned boxes. Labels
test complete solid intersection, including contact, not box centres or pixels.

The observation contains a perfect binary foreground silhouette sampled at image
pixel centres and 8x8 ToF zones. Each zone casts 3x3 deterministic subrays, uses
the nearest surface on each ray, and reports the mean radial range of hits up to
4m. Empty zones are missing. This is a deliberately ideal, noiseless hypothetical
sensor and perfect-segmentation input, not existing RGB extraction, physical ToF
firmware, or an oracle mask claimed available in deployment. No hit identity,
per-ray range, geometry, family, or true membership enters the predictor.

The fixed prior bank has 150 boxes: X centres {-0.60,-0.42,0,0.42,0.60}, widths
{.08,.24,.48}, front Z {.25,.60,1.50,2.50,3.10}, thickness {.04,.30}; each has
Y[-.15,.15]. Enumeration is the listed order, thin before thick. Bank metadata
and its rendered observation signatures are legitimate explicit prior data.
The closed-world source contains each of these 150 geometries once. This is
prior-inclusion feasibility, not holdout generalization or statistical sampling.

Out-of-prior source adds exactly 24 cases, fixed before scoring:

- 12 appendage cases: X centre +/-.60, width .48, front Z {.60,1.50,2.50},
  thickness .04; an inward arm spans signed X [.28,.40], centred at the front
  depth with Z halfwidth .01, and Y halfwidth {.002,.025}. These explicitly
  challenge subpixel/unobserved extent, rather than omit it from labels.
- 12 off-grid cases: X centre {0,+.60}, width {.18,.32}, front Z equals
  {.60,1.50,2.50} times 1.07, thickness .11, Y[-.12,.18]. These challenge
  absence of the true shape from the finite bank.

No scene or shape is selected after inspecting scores. Total 174 independent
static synthetic cases, no temporal claim. If the fixed observation model yields
no decision-disagreeing feasible sets, mark ambiguity NOT_EVALUABLE rather than
silently redesigning the source.

## Frozen inference and comparators

A bank shape is consistent only if its 64x48 mask is identical and every observed
valid ToF zone has a modeled hit with radial mean residual <=.01m. Missing observed
zones impose no absence evidence. No parameter fitting, learned model, threshold
sweep, noise sweep, or post-result bank expansion. Range tolerance is a declared
synthetic consistency threshold, not calibrated uncertainty.

1. Set consensus: no observed range or no consistent shape -> UNKNOWN; disagreeing corridor membership
   -> UNKNOWN; unanimous membership -> IN or OUT_MODEL_PRIOR.
2. Single best: same consistency filter; choose minimum sum of squared range
   residuals over observed zones, then fixed bank order. No observed range or consistent fit ->
   UNKNOWN. This isolates set-valued versus one-explanation commitment.
3. Point-only: project each observed zonal mean on its public zone-centre ray;
   any representative point in the corridor -> IN, otherwise UNKNOWN. These are
   public aggregate proxies, **not actual returned contributor coordinates**.

OUT_MODEL_PRIOR is agreement within the finite prior, never sensor-certified free
space. IN is also a model prediction. Raw observations remain intact. A missing
return does not become an OUT decision.

## Evidence, metric and outcome decisions

Write protocol/source/config hashes and input observations before inference;
persist all per-case predictions and their SHA-256 seal before evaluator geometry
is joined. Predictor accepts only observation fields and the fixed bank. Keep
truth geometry in a separate source file, opened for scoring after the seal.

Report by closed-world (also near-.30m-window versus other), appendage thickness and off-grid strata: TP, FP, explicit
false OUT, correct OUT, UNKNOWN-positive/negative, coverage, precision, FPR,
alert FN (includes abstentions on positives), feasible-set sizes, and membership
disagreement. Count wrong OUT separately from UNKNOWN. A scene is one static
case; no event/timing, real-use rate, Calibration-A improvement, or hardware claim.

Keep a diagnostic component only if at least one genuinely indistinguishable
opposite-label pair is retained as UNKNOWN and consensus reduces wrong explicit
commitments versus single best, while reporting its coverage cost. Any out-of-prior
wrong OUT falsifies unqualified reliance on consensus as a clearance certificate.
If consensus merely abstains everywhere, no useful capability is established.
The result cannot establish practical segmentation, geometric calibration, prior
coverage, natural generalization, or edge-device timing.

Budget: one bank render, one 174-case render/inference/scoring execution, focused
mechanical unit tests, and saved-output arithmetic/hash verification. Pure CPU
NumPy small analytic geometry: TASK_NOT_GPU_SUITABLE. No CUDA/UE/network, old
cohort read, new learning, successor experiment, or fit/tolerance retry. Mechanical
failures retain receipts; outputs are not overwritten. Store all payloads under
artifacts.local/work/ba-shape-hypotheses-20260921/. No persistent process is needed.
