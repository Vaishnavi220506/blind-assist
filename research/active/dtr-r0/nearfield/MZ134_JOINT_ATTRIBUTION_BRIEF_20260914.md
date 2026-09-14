# MZ134: joint source feasibility with an explicit unresolved explanation

EXPLORE, consumed controlled simulation, one fixed method. This brief precedes
prediction and scoring. Inputs are the MZ132 sealed anonymous visible masks and
the original 288 measured ToF packets; MZ129 is the comparator. No renderer IDs,
depth truth, actor names, categories or evaluator labels enter prediction.

Question: do positive multi-zone signals require any visible surface explanation
when background, nonvisual and unresolved contributors are admitted? Test the
small nonnegative linear proxy y = A alpha + u. y is signal times squared range;
A contains every visible mask's zone area fraction, a diffuse-background column,
and an outside-RGB coverage column. u is an unrestricted nonnegative per-return
unresolved contribution. This is an explicit weak inverse model, not the exact
sensor forward model, a calibrated likelihood, or evidence that every feasible
coefficient corresponds to a physically realizable scene.

Use the inherited .25 m complete-link forward-depth cohorts for SIM_VALID.
SIM_MERGED retains unknown constituent depths and is not assigned a false point
depth. For each A column, alpha has exact interval [0, min(y_i/A_ij)] over
positive entries; u supplies the remaining signal. Save concrete alternative
solutions and residuals. All-zero alpha and u=y is always feasible. Any geometric
contraction therefore requires additional constraints absent from this method.
Do not use a normalization or regularizer to turn a feasible solution into a
source probability. Visible masks remain anonymous, including visible floor
and background. The diffuse-background column is a nuisance hypothesis, never
a semantic identification of a mask.

Readout retains native zone and range intervals for every unresolved return,
with inherited central-four-column weights, threshold1 and certain-coarse rule.
Compute ToF directly from these returns and full system with the frozen MZ129
Radar-only branch. Never OR old full-system alerts into new ones. Report the
matched MZ129 ToF-only and full-system comparator, TP/FP/FN/UNKNOWN, families,
first-alert events, false segments, execution time and native contributor
retention. Full native footprint preservation is a support property, not a
claim that interval localization is physically accurate.

Falsifier: a declared algebraic two-source example has one visible off-corridor
source and one hidden in-corridor source giving the same aggregate positive
signal at the same range. Its hidden explanation must remain feasible. The
example tests the inverse representation, not rendered pixel equivalence or
physical sensor equivalence. After prediction sealing, independently inspect
actual native hit projections for whole-zone retention, with evaluator truth
kept outside the predictor.

Keep only if mandatory attribution is identifiable and task effects improve
without hidden contributor loss. No mandatory source and no nuisance gain means
NOT_EVALUABLE for surface-origin attribution under this weak observation model;
stop without training, threshold sweeps, method rescue or baseline promotion.
One run, plus mechanical corrections only. No worker or paid allocation needed:
small scalar/NumPy work uses CPU TASK_NOT_GPU_SUITABLE. Root owns registration,
current decisions and delivery; this subtask writes only MZ134 code and reports.
