# Inherited spatial mechanisms: one raw/local HGB contrast

2026-09-22. Authorized EXPLORE on already consumed controlled Development.
Question: can a small stable tree learner exploit query-conditioned local RGB
support summaries better than raw regional pooling, at a useful alert cost?

The frozen single-frame reference is ToF A, threshold 0.4071309640537889 with
its original definite-support bypass and UNKNOWN. The matched learned arms are
RAW and LOCAL. Both use the identical public 320x180 RGB, 64 ToF rows and six
camera-frame BODY/HEAD/lateral query boxes. Both retain 64 row-major descriptors
of valid/range/original interval endpoints plus RGB mean/std, x/y gradients and
gray quantiles, complete-image visual context, and query coordinates.
LOCAL additionally fills 51 zero-padded feature slots with RGB statistics and
original-range summaries in three disjoint query-support bands: the complete
regional interval fits, only some of it could fit, or none fits. RAW zeros those
same slots. Missing ranges do not enter local bands; their RGB and explicit
missingness remain in RAW. The final feature width is 961 in both arms, with
different effective information. Both consume the same pixels and observations;
this tests the added representation package, not parameter count or isolated
geometry causality. Regional range hypotheses are not pixel depth or ownership.

Prior exact negatives remain frozen: MZ147's body-frame four-sensor HGB with
3275 additional pixel features and OOF/lifecycle readout; the 2026-09-21 R
per-zone MLP/max and U/G convolutional packages; and the 2026-09-22 paired
MobileNet first-hit/mask recipe. This experiment changes the learner/objective
to one shared query classifier, keeps a strong common raw spatial carrier, and
tests a matched deterministic local augmentation on the new six-query source.
It is neither a new name for those models nor a threshold/loss repair of them.

Source: existing `ba-query-occupancy-20260922-prepared`, generated from 48
procedural base layouts. Keep all original indices: 864 train frames/24 layouts,
288 dev/8, 576 evaluation/16. Train query labels only are used to fit; dev
labels select one inclusive threshold per fixed fit. Evaluation is consumed,
not fresh, but its labels are not parsed by this implementation until both
prediction files, source hashes, models and selections are sealed. No capture,
evaluator geometry, object IDs, native depth, family/split/clip/time metadata or
baseline prediction enters the feature extractor. Metadata is only for splits,
the fixed reference, selection costs and evaluation strata.

One fit per arm: sklearn HistGradientBoostingClassifier, 200 iterations,
learning rate .05, 15 leaves, minimum leaf samples 20, L2 1, 64 bins,
early_stopping=False, seed 202609224, 4 CPU threads. All valid binary query
labels (occupied class <6) are used with equal weight; no sweep, fit retry or
post-outcome feature change. Final models only. CPU placement is
TASK_NOT_GPU_SUITABLE for small tabular tree fitting and ragged image statistics;
record actual sklearn/OpenCV/NumPy backend and real feature-probe time using
the shared research_backend helper. This is not a CPU/CUDA speed comparison.

Frame score is the maximum of the two central BODY/HEAD query probabilities.
The declared candidate output is A OR score>=threshold (no temporal change).
Choose among atomic dev score ties plus the disabled cutoff: maximum added TP
subject to at most 1 added FP and 1 added false segment versus A; tie-break by
fewer segments, fewer FP, then the larger cutoff. Save every dev breakpoint.
Standalone scores at that same cutoff and unchanged A one-frame hold are
descriptive controls, not selected alternatives. No held threshold selection.

Usefulness on 576 eval frames: at least +10 percentage points frame recall
versus A, at most +2 FP and +1 false segment, no loss of A-positive frames or
events, and no A onset delay. A OR ensures retention mechanically, so added
coverage and added costs determine the useful claim. LOCAL may instead be
retained only as a representation component if it improves the matched RAW
union by at least +5 points recall, adds at most 2 FP/1 false segment, improves
at least four of 16 layout groups, and its paired whole-layout 95% percentile
bootstrap lower bound is above zero. No BODY/HEAD recall may fall >5 points.
Otherwise record the exact arm/package as a negative control. A valid lack of
feature/label opportunity is NOT_EVALUABLE rather than an invented model loss.

Report per-query confusion, all frame counts including UNKNOWN, false segments
and sampled duration, all event identities/onsets, frame-retention costs,
BODY/HEAD/family/INSIDE/BOUNDARY/OUTSIDE/base-layout strata, 1000 whole-layout
bootstrap resamples with seed 202609224, input/model/prediction hashes and
feature/fit/inference cost. Six fixed query boxes do not establish arbitrary
query generalization; scalar classifiers establish neither mask IoU nor depth.
No endpoint hardware, natural distribution, user-benefit or safety claim.

Run once via `tools/ba.ps1 run research-ue -RunSpec <saved specification>`;
observations/configuration/evaluator inputs retain their existing scoped roles.
Outputs use `artifacts.local/evidence/ba-inherit-spatial-20260922-run` and are
sealed consumed Development. Mechanical repair may resume only completed
model/feature checkpoints with old failure/source receipts; scientific retries
are prohibited. No shared CURRENT, README, ledger, inheritance, commit or push
is owned by this subtask; the root agent handles consolidated delivery.
