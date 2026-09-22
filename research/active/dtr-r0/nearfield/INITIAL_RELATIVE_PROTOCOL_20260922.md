# Initial-camera-relative query with unchanged shared-bias observations

2026-09-22 EXPLORE. User authorizes continuing the
[boundary reference diagnostic](BOUNDARY_OBSERVABILITY_RESULTS_20260922.md).
Choose the ACTUAL INITIAL camera as the query origin for this bounded multiview
pilot. All13 observations serve one corridor at that initial pose/yaw0, not a
query that changes as the camera moves. This is a specific experimental reference
contract, not a silent replacement for the adopted current-camera runtime task.
No historical truth, source file, model, report or score is overwritten.

## Mechanism and assumptions

Keep the previous global scene parameterization, box domain, wall4.2m, eight
directions, .10m bins, fixed x_then_z nominal13view12cm path, and full range_pose
shared constant observation error(range±2mm,X/Z±1mm). Actual initial camera is
Q=(round(px,12),round(pz,12)), following the unchanged injector. Change ONLY the
queried corridor to [-.3+Qx,.3+Qx] × [.3+Qz,3+Qz]. Continuous query constraints
must include the existing analytically derived1e-12m rounding outer envelope,
including inclusive OUT closures; exact validation/projection uses Q. No fitted
tolerance or new error amplitude is introduced.
Candidate geometry/query labels, IN/OUT constraints and IN projection must all
use this same translated reference. Preserve full-box intersection and old EPS,
state gating, solver limits and exact biased forward-bin validation.

Joint object/camera global X translation then preserves query truth. Do not set
px to zero by pretending it is calibrated, silently tighten uncertainty, drop
global geometry priors, or move the fixed world wall. Keeping nuisance variables
in the global parameterization avoids changing these inherited constraints.
World-domain and wall assumptions remain model restrictions; this is not a
complete quotient-state rewrite or cancellation of relative pose/extrinsic error.
No yaw/resolution change, current-camera query, new error process or learner.

## Fixed input and truth authority

Reuse ALL375 unique13view public histories from ba-bias-drift-20260922/run-v1,
mapping to the same180consumed scenes ×13conditions. No new observed/simulated
measurements. Inference receives only nominal camera+bins, not IDs/strata,
actual poses, condition tags, truth or evaluator bias. Use one model preset,
full range_pose, two original-budget MILPs per query. No retry or scored repair.

Seal protocol, code, dependency hashes and copied public inputs before solving.
Seal all predictions before evaluator join. Then define NEW truth from saved
source full box relative to that condition's saved actual initial camera. Save
old global truth alongside it, including exactly which labels change and class
denominators per condition/stratum. Do not force the new labels to be90/90.
All old baseline results remain unchanged under their original semantics.

The nominal condition has actual initial camera at origin, so old and new truth
are identical: old global range_pose versus new relative model is a paired
capability comparison there. In nonnominal conditions, count each algorithm
against ITS OWN query truth; compare old global outputs against relative truth
only as explicitly named semantic-mismatch diagnostics, not fair performance
gain or an error in the old model. Same recorded observations do not make
different query targets equivalent. Report all13conditions, not only favorable.

Model-scope controls are nominal and all6constant-error conditions. All6drifts
remain outside the shared constant model. Initial pose subtraction does not
eliminate changing inter-view motion error. Under pose drift, reference is still
the first actual pose and remains fixed for the entire13view history.

## Outcomes, validation and stopping

Report TP, FP, falseOUT, correctnegative, UNKNOWN, precision, recall/FPR with
actual denominators for all/general/boundary and each boundary face. Separate
true model ambiguity(two verified witnesses), numerical/incomplete search and
both classes reported infeasible. Nominal gains/losses use exactcase identities;
cross-condition retention must be truth-aware and separately show labels that
changed from nominal, not require retaining now-wrong labels.

Main falsifier: nominal correctness/coverage and whether any boundary conditional
decisions appear under the fixed relative query, without wrong commitments in
the model-scope controls. Zero gain means correct coordinate semantics did not
supply enough observation information; do not add yaw/finerbins after readout.
Any model-scope wrong commitment requires defect diagnosis, not a scored repair.
Drift errors are misspecification evidence; never claim safe abstention by design.
The old40 global-gauge counterexamples must preserve new relative labels, which
is a semantic regression check, not40recovered classifications.

Independent review validates shifted constraints, label/projection consistency,
forward witnesses, and all1260truegeometry+constantbias assignments after seals.
Audit full metrics/label changes, oldfile integrity and resource use. Exactly
one375query run,max750MILPs, original1s/10000node/class, no follow-up search.
Complete focused tests, audits, report/current, supported registration/local
inheritance and scoped commit/push. CPU SciPy/HiGHS TASK_NOT_GPU_SUITABLE.
No paid/persistent resources. New output must not exist:
artifacts.local/work/ba-initial-relative-20260922/run-v1.
