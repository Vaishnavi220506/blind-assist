# Public positive readout: pooled crossfit calibration and negative peak loss

EXPLORE, user-authorized continued development of the existing 1,537-parameter
public ToF head. Preserve the preceding sealed result. Its learned signal,
calibration opportunity imbalance and false maximum activations are separate
findings. Test those two latter mechanisms without changing model architecture,
input fields, per-return supervision, max aggregation, frozen A or the OR rule.

## Data and split

Reuse authenticated prior anchor192 + old288 + changed288. Add complete consumed
MZ146288 and MZ158288 captures using their authenticated public feature/A caches
and native sampled-return labels. They add background/negative coverage, not new
clear supported A misses: the available inventory found none in either source.
Do not select frames by errors. Original MZ136 test48 and dev48 remain excluded.
All 1,344 frames are Development. Shared simulator/family templates and prior
outcome access preclude fresh or independent-domain confirmation claims.

Retain six outer scene-index folds, k=0..5, jointly holding scene k from all
four 288 cohorts (192 report frames). Original anchor192 is always fitting
material; 960 other cohort frames form the outer development set. No outer
report frame enters its model, normalization or threshold selection.

Within each outer development set, sort its five scene indices and assign them
round-robin to three inner folds (two/two/one scene indices). Fit each inner
head on anchors plus the other development scenes. Every development frame gets
one score from a head that excluded its entire scene index across all cohorts.
Pool these 960 out-of-fit predictions for one outer threshold, maximizing final
OR clear-task F1, then fewer FP, then higher threshold, as previously. Include
finite nextafter(max calibration score) candidate; it is inactive on calibration
only, not a global disable. Fit the final head on all outer development plus
anchors, then apply the chosen threshold to the outer report frames. Transferring
a threshold from smaller inner fits to the larger refit is itself evaluated,
not assumed calibrated. Never tune on outer scores or pick the best outer fold.

Complete paired trajectories and all family members of each scene index remain
together. Report all cohorts and per-scene opportunity counts, including the
absence of independent supported clear misses when outer scene0 is held out.
Adding negative scenes or crossfitting cannot manufacture positive opportunities.

## Matched comparisons

Stage 1: unchanged balanced return BCE, with (a) prespecified fixed logit0 and
(b) pooled-inner-calibrated threshold. Fixed zero is a formal comparator in this
new Development experiment, not a retrospectively relabelled old result. The
class-balanced loss does not make sigmoid0.5 a physical event probability.

If Stage 1 fixed-zero outputs add any clear or boundary FP, or its calibrated
outputs leave supported clear A misses, execute one matched Stage 2: add negative
frame peak loss with fixed coefficient 1. No coefficient search. The criterion
decides whether to run this already specified comparison, not which result to
report. Otherwise stop fitting after Stage 1 and report the resolved comparison.

Keep equal total known-positive/known-negative return BCE mass. For fitting
frames only, define eligible negatives as A=false, strict truth=false, 5cm clear,
and at least one usable ToF return. Anchors have no A field and are ineligible.
Additional loss is the mean softplus(max valid return logit) over those eligible
fitting frames. Its dataset-total weight is one, matched to total return BCE
weight one; use unbiased minibatch weights. It neither copies positive frame
labels to returns nor demands multiple returns. Invalid/UNKNOWN and A-alerted
frames get no peak loss. Per-return loss remains unchanged for sparse positives.

Both arms use 30->32->16->1 ReLU, fit-only standardization/std floor .01, AdamW
lr .001, weight decay .0001, batch16, 120 epochs, final checkpoint. Seed for outer
k/inner j is 184017+10*k+j; final refit uses j=3. Same seeds, batches and schedules
for both arms. At most 24 fits per arm; no wall-clock cutoff, seed, epoch, model
or threshold-metric search. Finite calibration candidates are permitted by the
specified selection rule. Reuse measured CPU placement for Stage 1; measure
equivalent CPU/CUDA forward/backward with the peak term for Stage 2.

## Evidence, limits and completion

Seal protocol/source/data and final predictions before outer result reduction.
Compare A, BCE zero/calibrated, peak zero/calibrated; retain sampled-positive OR
as an existing reference only. Report clear P/R/F1 and coverage, strict/boundary,
rescued A misses/new FP by family and whole episode, calibration opportunity,
first observed alert, negative burden, A retention and zero-return fallback.
Report incremental effect of the peak loss separately from calibration/data
changes. Previous-to-current results are not a data-matched loss ablation.
Return scores and fitting losses do not replace final-alert results. No claim
of advance warning from left-censored episodes or removal of A false alerts.

Check group exclusions, normalization and fit-only loss indices; verify peak
gradients reach only the selected valid negative maximum, including single-return
and zero-return cases. Independently reconstruct saved models/calibration choices
and final counts. Keep the six-fold nature explicit: this is a reusable trained
component study, not one final all-data deployment checkpoint or App promotion.
Retain useful tradeoffs with their costs; null/negative arms remain in the report.
Finish evidence, current pages, scoped delivery and resource release. No capture,
new sensors, RGB/depth backbone, veto, successor sweep or ledger bypass.
