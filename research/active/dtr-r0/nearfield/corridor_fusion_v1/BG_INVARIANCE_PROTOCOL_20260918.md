# Texture counterfactual supervision pilot

One bounded EXPLORE experiment, authorized by the user's counterfactual training
proposal. Does background-texture consistency reduce false alerts without losing
the raw2224 representation's recall? Prior ranking-only CCRL failed; this changes
the representation and supplies authenticated background interventions. It is not
a reopening/tuning of that fixed recipe, or proof of causal factor disentanglement.

Keep frozen raw HGB and A* as independent strong references. Fit three identical
2224->16->1 zero-output-initialized residuals on raw HGB log odds:
B0 ordinary BCE; B1 BCE+.25*hinge intrusion (margin1 logit);
B2 BCE+.25*hinge+.1*mean absolute logit difference on texture pairs.
Same initialization seed189018,120 full-batch AdamW steps,lr.001,weight decay.01,
fit-only mean/std floor.001,clip[-8,8],last checkpoint. The residual is a disclosed
architecture change versus HGB; only B0/B1/B2 isolate objectives. No latent scene /
intrusion decomposition is implemented or claimed in this minimal loss test.

Existing data: original1344 rows, with192 anchor HGB-only; use1152 saved raw-HGB
OOF scores for final residual fitting. Add192 new texture-factorial train rows
to every residual arm equally. Underlying raw HGB weights/data stay fixed so
data expansion cannot be confused with a loss-specific difference. Existing
672 lateral pairs are authenticated; residual uses576 old plus96 new pairs.

New source seed189018:12 physical groups (3 per4 families),2 lateral members,
2 context appearances,6 frames each=288. Index0/1 train192;index2 held96. Change
only context texture enable/grid/seed; target/camera/native geometry/reflectance/
sensor seeds remain identical. Plain versus procedural grayscale tiles are not
portal/recessed/clutter geometry or arbitrary background changes. All context is
behind target and outside the corridor. Require native geometry/labels agreement,
identical public non-RGB sensor features for background pairs and actual RGB
change. If these fail, preserve source diagnostics and stop before invariance fit;
do not silently drop difficult pairs or force confidence under lost observability.

Calibration: six whole-scene outer folds, matching old A* splits. For old rows,
five inner raw-HGB fits per outer generate out-of-fit training logits (30fits,
fixed original HGB params). New train physical index0/1 is mapped to outer folds
0/1 and completely removed from that fold's residual fit; all texture and lateral
members stay together. New training logits come from corresponding outer HGB;
new groups never enter any HGB fit. Final residual trains on all old OOF+new192
logits from frozen full raw HGB for new rows. Crossfit-to-final transfer is explicit.
Each arm chooses one threshold using pooled1152old+192new held-fold scores and
the same clear-F1/fewer-FP/higher-threshold rule. No checkpoint or loss sweep.

Report unchanged consumed288 cohort and new held96 texture-factorial frames
separately. New96 is a small same-generator Development transfer pilot, not a
protected final or natural distribution. Freeze recipe/source before acquisition,
seal model/threshold and public predictions before report label joining. No new
report frames enter fit, normalization or threshold selection. Original dev/test
remain excluded. Existing HGB references must replay exact saved old288 scores.

Report clear coverage,strict/boundary/rod TP/FP/FN,P/R/F1,false-positive episodes,
intrusion score margin/order,texture-pair absolute logit/probability drift and alert
flips,core/strict events/onsets,native-supported lost/rescued TP and zero-return
strata. B2 attribution requires improvement over matched B0 AND B1, not only A*.
The hoped-for99TP/<=3FP on consumed clear is a target, not a selected checkpoint.
Keep candidate only if clearTP>=raw99,FP<=3, no raw clear TP lost, no core/strict
baseline event loss/delay, and new held invariance improves without recall/native
retention collapse. Otherwise retain baseline and close the exact recipe. Failure
does not show HGB has exhausted all possible signal or authorize switching projects.

Budget: one288-frame capture,21 residual fits,30 inner HGB fits,one fixed recipe.
No automatic tuning, second capture, factors model or successor. Mechanical
failures may resume exact input/config state with receipts; no scientific redraw.
Use worker UE with current capacity, owned process release and canonical artifacts.
Inherited no-total-wall-clock-cutoff capture adapter keeps readiness timeout300s;
monitor actual progress and diagnose stalls, do not reset by duplicate dispatch.
GPU-first real-step timing chooses residual backend; sklearn HGB CPU reason is
GPU_BACKEND_UNAVAILABLE. Preserve raw data, models,seals,logs,source/eval/audit
receipts; remove only owned reproducible DDC after completed release verification.
