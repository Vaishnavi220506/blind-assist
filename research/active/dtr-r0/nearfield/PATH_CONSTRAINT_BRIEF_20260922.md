# Continuous constraints along a saved path: fixed conditional readout

2026-09-22, EXPLORE. New separately authorized interpretation of the consumed
180-scene / 9540-view observation-mechanisms evidence. No scene generation,
rendering, training, bank expansion, observation selector, or source adaptation.
Keep continuous_boundary_witness.py unchanged. Root seals all predictions before
joining saved source truth. Identical public histories may share one computation;
report every original scene and the number of unique solved histories separately.

## Public inputs and continuous model

The new infer(observations) accepts 1..13 records with exactly camera and eight
integer bins. Cameras must belong to the original53 saved diagnostic positions:
origin,1cm samples to12cm on the four axis spokes, and four (+/-.06,+/-.06) corners.
These include the13 original two-step tree poses. Histories need not be ordered
because this static model conjuncts them; no scene ID, true geometry, truth,
raw ranges or family enters the solver. All cameras have Z<=.12m, ahead of the
model's nearest rectangle front .58m, so the same ray-entry derivation applies.

Reuse the original one-rectangle domain, fixed .04m thickness, wall4.20m,
eight rays, .10m bins and original camera-frame query. IN/OUT below are only
conditional on this exact restricted analytic model; zero, multiple or moving
objects, noise, unmodeled shapes and physical sensors are outside its authority.

## Feasibility review and the necessary inherited-EPS envelope

Write L,R,F for lateral faces and front; a ray has direction (u,v), v>0, and
x_front=cx+(u/v)*(F-cz), x_back=x_front+(u/v)*.04. The ray hits iff its lateral
and depth intervals overlap. Its entry is max((F-cz)/v,(near_side-cx)/u).
Each target bin bounds this maximum in [(k-.5)*.1,(k+.5)*.1). The MILP uses
<= at the upper boundary, so it includes the half-open upper face as an outer
relaxation. Wall observations require a miss; relaxing strict miss inequalities
to inclusive ones adds tangent cases and cannot remove a genuinely missing ray.
A modeled hit cannot quantize to the wall bin: the farthest rectangle back3.42m
is separated from wall4.20m by .78m axial depth throughout this camera domain.

The common interior slack is allowed to equal zero. Maximizing it does NOT
change feasibility: every feasible slack0 assignment remains admitted. Thus no
additional zero-objective feasibility solve is necessary. OUT alternatives are
R<=-.30, L>=.30 or F>=3.00 in the slack0 closure. They include every strict OUT;
contact-only false OUT witnesses remain subject to forward label validation.
Big-M constants come from fixed variable bounds and only disable the unchosen
disjunct; continuous L/R/F are not a sampled or densified hypothesis bank.

The original scalar simulator deliberately uses EPS=1e-12 for interval contacts
and query membership. Literally unchanged ideal IN/hit inequalities can omit
an infinitesimal strip that this implemented model accepts. Before scoring,
the new wrapper makes only these analytically derived outward changes:

- IN bounds L<=.30+EPS, R>=-.30-EPS, F<=3.00+EPS.
- For each observed target-hit ray, add abs(u)*EPS to both lateral-overlap
  inequality bounds. The slab rule entry<=exit+EPS permits a lateral residual
  of at most abs(u)*EPS; this is the exact coordinate conversion, not a fitted
  uncertainty margin. No-hit, range-bin, OUT, domain and other bounds stay fixed.

At slack0 these inequalities are an outer relaxation of the ideal real-arithmetic
model with those inherited contact rules. Floating coefficients, double-precision
arithmetic, HiGHS numerical tolerances and finite solver reports are NOT a formal
soundness proof. Raw data or query tolerances are never enlarged after seeing
results. The original module, witness settings and old evidence remain unchanged.

## Frozen calls and readout

Exactly two MILP calls per unique public history, one IN and one OUT. Retain
original time_limit1.0s, node_limit10000, mip_rel_gap0, presolveTrue and slack
[0,.0001]. No retry, extra feasibility solve, objective change or parameter sweep.
Candidates undergo the same deterministic domain projection/12-decimal rounding,
then validate against the original scalar observer and EPS-aware full-extent
query on all supplied samples. Invalid candidates supply no positive evidence.

IN_MODEL_CONDITIONAL or OUT_MODEL_CONDITIONAL requires a validated witness of
that label with its own solver status0 AND solver status2 with no candidate on
the opposite OUTER relaxation. A valid incumbent found at status1 is preserved
as an existence witness but remains UNKNOWN under this fixed conservative rule.
Otherwise return UNKNOWN. In particular, status1 time/node limits, solver errors,
status0 without a candidate, numerical candidate rejection, an inconsistent
status/candidate combination, or both sides infeasible do not authorize an OUT
or IN. Record distinct solver outcomes and unmodified solver messages/timings.

Every output has authority NUMERICAL_SINGLE_RECTANGLE_MODEL_ONLY. Conditional
exclusion is a numerical solver report under the declared model, not a formal
certificate, measured free space, hardware evidence or a universal obstacle claim.
If both witnesses exist, preserve opposing-pair action separability and the
pair-only alias qualification. A chosen pair is not all possible ambiguities.

## Validation and fixed evaluation scope

Focused non-cohort fixtures check inherited query-EPS and ray-corner strips,
half-open-bin outer closure, near .30m contact versus actual OUT, forward replay,
pose/input whitelisting, and every numerical/solver outcome's decision mapping.
No fresh/consumed full-cohort scoring is performed by the component author.
Root compares saved initial/endpoints/path13 and existing3-view histories once,
with prediction seals, full denominators and separate conditional TP/FP/falseOUT/
UNKNOWN counts. The same consumed source is Development, never confirmation.
Artifacts belong under artifacts.local/work/ba-observation-deepening-20260922/,
the root's canonical consumed-mechanism successor evidence tree. No persistent worker.
