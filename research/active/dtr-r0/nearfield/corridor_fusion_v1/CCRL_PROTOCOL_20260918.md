# Minimal CCRL Development experiment

Question: can lateral counterfactual supervision recover A* misses while retaining
its low false-alert burden? This is one fixed existing-data EXPLORE run, not a
fresh confirmation or novelty claim. Keep the saved A* and all earlier evidence.

Inputs: the authenticated 1,344 x 2,485 A* features, strict native frame labels,
and complete source scene pairs. Original MZ136 dev/test are excluded. The later
288-frame single-version confirmation is now explicitly consumed Development
report data; it is never used for fitting, normalization, checkpoint or threshold
selection. No capture, new encoder, hardware or App changes.

HGB cannot directly optimize a pair loss. Use a zero-output-initialized
2485->16->1 residual added to A* log odds. Compare identical BCE and BCE+ranking
residuals; add BCE+ranking+invariance only if exact background intervention pairs
exist. This adapter is a disclosed architecture change relative to A*, and only
the matched residual contrast isolates the objective. A* is an independent
strong baseline, never described as already using CCRL.

Fixed recipe: seed187018, 120 full-batch AdamW steps, learning rate0.001,
weight decay0.01, last checkpoint only, margin1 logit, ranking coefficient0.25,
invariance coefficient0.1. Fit-only mean/std (scale floor0.001), standardized
feature clip[-8,8]. Strict labels for BCE; whole-pair rank orientation from native
truth. Latent source fields only authenticate training pairs and never enter the
prediction feature vector. Same RNG start does not guarantee identical later
geometry-dependent sensor draws. Exact invariance requires unchanged target,
camera, body, time, sensor seeds and native label with changed context. Zero such
pairs means that term is NOT_EVALUABLE; do not substitute arbitrary same-label
frames or synthesize feature edits.

Six outer folds hold the same scene index across all four non-anchor cohorts.
For each outer fit, five inner whole-scene HGB fits produce out-of-fit base scores
for the960 residual training rows. Every HGB fit includes the192 anchor rows;
the residual never trains on anchor in-fit predictions. Fit/report scene groups,
normalization and pair edges remain disjoint. Reuse saved outer HGBs on192 held
rows. Fit the final residual on the1,152 saved A* OOF scores, then attach it to the
unchanged final A*. This crossfit/refit transfer is a limitation to evaluate.

Each residual chooses one threshold from its1,152 outer-held scores using the
same clear-F1/fewer-FP/higher-threshold ordering as A*. Pooled OOF numbers are
selection statistics, not unbiased validation. Seal final weights, thresholds
and all288 public predictions before joining report labels. Preserve exact A*
score parity. No report threshold sweep or successor experiment.

Report clear5cm coverage, strict and boundary TP/FP/FN, precision/recall/F1,
family strata, paired score ordering, core/strict events and onset changes,
rescued/lost A* true frames, zero-return/returned-support strata and measured
compute backend/time. Model no-alert and missing returns never certify free space.
The candidate does not impose a missingness veto or claim native retention by
construction; measure retained native-positive true alerts explicitly.

Keep a challenger only if the consumed288 report improves clear F1 and recall
with no more clear FP, no lost A* clear TP, no core/strict event loss and no delayed
retained event. Attribute an objective gain only when ranking also improves on
matched BCE. Otherwise retain A*, record the failed fixed recipe, and stop.
One recipe, at most18 residual fits and30 nested HGB fits; no automatic capture,
sweep, fusion successor or new final-method claim. Mechanical fixes may resume
the same unmodified fitting recipe with failure logs. Retain weights, pairing,
predictions, source/model hashes and diagnostics; processes exit on completion.
