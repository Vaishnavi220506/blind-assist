# One frozen-32D last-layer readout pilot

EXPLORE, consumed same-simulator Development. User authorized one pilot after
the Exa cross-layout study. No new capture, encoder/trunk training, protected
test access, runtime promotion, loss sweep or automatic successor.

## Question and contrast

Can the retained 32D spatial representation support a common low-FP readout
across layouts? Freeze the original encoder and head through its penultimate
ReLU. Fit only Linear(32,1), 33 parameters. Uniform is the mechanism control;
layout-by-binary-label balanced is the prespecified challenger, never chosen
after evaluation. A is the frozen strong geometry baseline; C is A OR original
B>=7.6612162590026855. Both new heads also supplement A by OR. Original 0.2s
nonrecursive one-frame hold and UNKNOWN are unchanged.

## Data and fixed roles

Readout FIT: all eight original BCE dev layouts (576 frames), not its 24 original
representation-training groups. SELECTION: transfer layouts ending g00/g01
(eight groups,576 frames). EVALUATION: transfer g02/g03 (eight groups,576 frames).
Each role contains two groups per family. These ID-based roles are fixed before
new fit/results; all are already consumed Development, not fresh confirmation.
Every layout contains all three lateral clips and all24 times on the same side.
No per-layout ID/bias/threshold enters inference. Original reserved test logits,
labels and model execution remain unactivated. Cached original test features
are not indexed or passed to the model; hashing a source cache is not inference.

The preparation stage may partition the existing consumed evaluator-label file
into role files. Subsequent fit/selection programs read only their role labels;
evaluation labels and native support are joined after predictions are sealed.
IDs are namespaced by source. Original source hashes and published logits must
reproduce (absolute logit tolerance1e-4); stop on mismatch.

DEV has near-boundary OUTSIDE, unlike the old15k distant crossbar inventory.
All DEV groups have72frames,20-24 positives: equal layout-by-label mass mostly
upweights positives, not hard-negative mining. Interpret the contrast accordingly.

## Exactly two deterministic convex fits

Standardize each of32hidden features using FIT-only population mean/std with
std floor1e-6, shared across arms. Both start from zero weights and bias.
Full-batch weighted BCE plus .001/2 times squared weight norm; bias unpenalized.
Uniform row weight=1/N. Balanced row weight=1/(2G*n_group_label), giving equal
mass to each of16nonempty layout/label cells. No sampling randomness.
Float64 torch LBFGS, lr1, max_iter500,max_eval1000,tolerance_grad1e-9,
tolerance_change1e-12,strong_wolfe. Save final iterate only, no retry or tuning.
Measure equivalent CPU/GPU probes on disposable copies; use research_backend
to choose placement. Frozen feature extraction uses the measured backend too.
Record convergence diagnostics and actual workload time, not device latency.

## Selection and sealed evaluation

For each new head select once the lowest cutoff that adds no FP to A on BOTH
current and held policies, across every selection negative including UNKNOWN.
Forbidden trigger positions are (a) negative with A_current=false or (b) their
previous same-clip frame when A_hold at that negative is false. Cutoff is the
next float64 above the maximum forbidden score. Atomic >= ties; no zero floor.
This OR policy preserves A positives/onsets and FP segments when no FP is added.
A silent supplement is admissible but has no useful rescue signal.

Freeze both cutoffs and predictions before joining EVALUATION labels. Evaluate
both prespecified heads even if selection rescue is zero: this is a consumed
Development falsifier, never activation of the original protected test.

Primary useful repair requires, on selection AND evaluation: no extra Core or
Boundary FP/false segments versus A for current AND hold; all A flags retained;
Boundary current recall gain>=.10, gains in>=4/8groups; >=90% retention of the
specific C-rescued Boundary current frames; no loss of C-detected Boundary events.
Report every criterion and any partial tradeoff. Balanced-vs-uniform contribution
is separate from absolute usefulness; balanced is never relabelled by outcomes.

Report raw TP/FP/FN, precision/FPR, groups, UNKNOWN, events, first in-event
delay, internal silence and exit costs; recovered/lost IDs and native contributors.
xAUC compares Boundary positives to OUTSIDE negatives at positive-depth matched
times, separately per role with equal-layout off-diagonal mean and full matrix.
Matched lateral pair ordering is a different statistic. Negative-depth FP stay
in task totals and get a separate diagnostic count. No masks means IoU=N/A.

## Stop and delivery

One pair of fits, one fixed partition and one selection per head. Failure closes
this 32D/readout/weighting recipe, not raw-sensor information or all future models.
No feature, loss, seed, regularization, split or threshold rescue. Preserve source
and model receipts, predictions, checkpoints, metrics and audit. Attempt supported
registration/inheritance; document pre-existing global metadata failures without
editing/bypassing the ledger. Complete scoped tests, independent saved-result audit
and normal Git delivery; retain the existing demo and baseline.
