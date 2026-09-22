# Frozen-checkpoint train/dev diagnostic

2026-09-22, user-authorized EXPLORE follow-up to the closed query-occupancy
prototype. One inference-only diagnostic, no fitting or threshold selection.
Use only existing train864/dev288 observations and labels with the selected
classifier epoch16 and occupancy epoch8. No held evaluation arrays are opened.
These are consumed Development observations, not fresh confirmation. The original
negative-control disposition, checkpoints, source seals and outputs stay fixed.

Before outcomes, examine three competing explanations: poor training-sample
fit of the selected checkpoint, loss of generalization, and subthreshold spatial
ranking concealed by the fixed0.5 mask cutoff. Report both original fixed alert
cutoffs and threshold-free AUROC/AP; never propose a new operating point here.
Inspect seven-class/positive-bin accuracy, binary query BCE, per-query probability
spread, and occupied-query-over-empty-query ordering within each same image.
Separate same-height lateral pairs from BODY/HEAD pairs. Pair INSIDE/BOUNDARY
with OUTSIDE at the same base layout and posed frame to test corridor distinction.

Mask metrics use native occupied and valid area in45x80 pooled cells, with exact
score ties and no invalid pixels treated as background. Report macro foreground
versus background AUROC/AP over visible-positive queries, prevalence (constant
score AP), foreground/background mean probability, original-cutoff IoU, and
peak-cell occupied fraction. Wrong-query maps on the same image are a descriptive
control: compare each positive query against each valid empty query's map.
No ground-truth-area top-k mask, posthoc cutoff, loss or checkpoint is selected.

Save train/dev predictions before metric analysis, exact input/code hashes,
per-query/per-pair results and a compact figure. Verify reproduction of original
dev fixed-cutoff counts; independently check ranking ties and main conclusions.
Use measured actual inference backend; no optimizer, backward, or BN-state update.
Only selected checkpoints survive, so conclusions concern those states, not the
last epoch or whether a longer/changed optimization could fit. Loss imbalance or
query-decoder explanations remain hypotheses unless directly distinguished.

Finish a separate report, scoped current update and diagnostic COMPONENT
registration if interpretation changes. Stop after delivery; no training,
calibration rescue, video, new capture, or App integration follows automatically.
