# Ordered spatial features with interval-relative geometry: matched contrast

EXPLORE, one separately authorized two-arm comparison, simulation only.
The previous interval-band statistics package is closed. Its HEAD-horizontal
0/43 versus old Spatial-BCE37/43 motivates preserving ordered image features;
it does not prove pooling caused the miss. This experiment tests whether explicit
interval-relative geometry helps an otherwise identical spatial head. It does
NOT isolate the causal effect of restoring spatial order versus the old package.

Both arms reuse the sealed frozen MobileNetV3-small features[:4] RGB cache:
24 feature channels x nine ordered samples in each 8x8 zone grid, losslessly
rearranged to [24,24,24] (feature, zone-row/subrow, zone-col/subcol). No encoder
fit, RGB resize change, global band statistics, native depth or evaluator masks.
The original cache contains admitted test observations, but only train/dev
indices are selected; whole-file provenance hashing does not activate that test.
No test images, labels, logits or metrics are opened or produced.

Four common channels: original range/8, valid, sample-ray horizontal/vertical
angles normalized by45 degrees. Original full interval radius=.1+3*(.01+.02*r),
lo=max(.1,r-radius), hi=r+radius. Eight further channels encode signed distances
to X=-.3,+.3 and Y=-.2,+.9 at BOTH interval endpoints on each sample ray:
[a*z+.3,.3-a*z,b*z+.2,.9-b*z], clipped[-4,4]/4. These are hypothetical geometric
query coordinates, not measured target locations or reduced support.
U (unconditioned control) sets these eight channels to zero; G retains them.
Identical 36-channel tensors otherwise, identical model dimensions/initial
weights/batch schedule/optimizer/device. U's zero slots are an explicit input
ablation, not a claim of equal effective feature capacity. Invalid ranges get
zero range and zero conditional channels, retaining validity0 and RGB; either
model may classify from RGB but cannot turn UNKNOWN into measured support.

Head for BOTH: Conv3x3(36,32,pad1),ReLU; Conv3x3(32,32,stride2,pad1),ReLU;
Conv3x3(32,32,stride2,pad1),ReLU; Flatten(32x6x6),Linear(1152,32),ReLU,Linear(32,1).
One fit PER ARM: seed20260921,1200 AdamW updates, batch64, lr.001,wd.0001,
unweighted BCE, identical replacement sampling, last checkpoint, no augmentation.
One same-workload CPU/GPU training probe on G chooses the common backend for
both arms using research_backend. Inference likewise uses one common probe.
No architecture, loss, seed, fit, feature or threshold retry after outcomes.

Use original24train/1728frames and8dev/576; evaluate once on all16 consumed
transfer layouts/1152frames. Complete INSIDE/BOUNDARY/OUTSIDE clips retained;
train/dev/transfer group IDs disjoint. These data informed the hypothesis, so
group separation does not make this fresh confirmation. Protected test closed.

A is frozen strong and nonrecursive .2s hold. U/G decisions each OR their own
current score with A, then apply the same hold. UNKNOWN is always A's value.
Include frozen old B and previous interval-band R as controls, without altering
their checkpoints/thresholds. OR supplements cannot remove A's existing FP.
For EACH new arm independently, select the lowest inclusive dev score satisfying
the existing budget: Core and Boundary separately, extra current FP<=floor(.01*Nneg),
extra hold FP<=floor(.02*Nneg), extra false segments<=floor(.125*Nclips) for each
readout. Include nextafter(max,+inf) and compare in float64. No transfer selection.

Separate two conclusions:
1. Usability, evaluated identically for U and G: all transfer cost caps pass;
   all A positive alerts retained; Boundary hold recall>=.50 and positive gains
   on>=8/16 layouts; HEAD-horizontal Boundary hold recall>=.50, >=3/4 events
   detected, and every detected horizontal event first-alert delay<=.4s.
2. Added-geometry hypothesis: G is usable, has at least5 more horizontal true
   frames than U, improves horizontal TP on>=2/4 layouts, and does not reduce
   total Boundary hold TP versus U. This tests the full trained decision system
   at the same budget contract, not scores at an identical numeric threshold.
If G adds no benefit but U is usable, retain U as a research challenger and
reject the geometry addition. If neither is usable, close both exact recipes.
If G is usable but its marginal contribution fails, retain only its scoped
capability, not a geometry-improvement claim. No default/demo promotion.

Seal train/dev logits and selected cutoffs, then transfer predictions BEFORE
evaluator joins. Report current/hold TP/FP/FN, precision/FPR, event delay,
initial/internal/terminal silence, exit cost, full layout/family/BODY/HEAD and
background groups. After sealing, audit new TP lacking native returned corridor
contributors; this remains learned classification, not metric-depth evidence.
No IoU claim (scalar alert output). Repeatability checks are not evidence of
hardware latency, natural-distribution efficacy or safety. No real sensor exists.

Finish scoped checks, local structured inheritance, results/current notes and
normal commit/push. Preserve global index303 registration failures and local
receipts; do not edit the old ledger. Stop after this contrast; no automatic
successor or protected-test opening, regardless of result.
