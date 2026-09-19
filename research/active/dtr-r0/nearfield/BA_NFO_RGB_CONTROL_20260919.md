# Matched RGB-only NFO diagnostic

## Completed result: recall recovery is mostly broad overprediction

The one RGB-only fit completed12epochs in142.82s on RTX5060 Laptop GPU.
Final loss0.32879 versus original NFO fusion0.09537; this is a limited
from-scratch training recipe, not a statement about the information ceiling
of RGB. Validation selected cutoff0.055 (recall97.56%,IoU42.28%) using the
unchanged grid/rule. Test thresholds were never scanned.

At2m the RGB-only mixed-domain IoU is41.11%, recall97.47%, FP754025,
versus fusion NFO51.04%,94.70%,470344. Full-image IoU is10.23% and
FPR94.27%, showing how little spatial specificity remains at the required
high-recall operating point. This is not a useful recovery result.

### <=20% near-area groups, frozen validation operating points

| Group | Depth fusion recall / IoU | NFO fusion recall / IoU | RGB-only recall / IoU |
|---|---:|---:|---:|
| Near return |88.26% /15.49%|92.42% /15.43%|97.92% /11.40%|
| Far return |89.64% /8.71%|69.61% /12.09%|97.89% /6.35%|
| Missing |64.35% /9.22%|72.97% /12.71%|98.50% /8.15%|

Far-return subgroup RGB-only gives13571TP/199710FP/293FN, compared with NFO
fusion9651/65965/4213. Its FPR is96.26% versus31.79%; precision6.36%
versus12.76%. Predicting every known pixel near would give6.26% IoU in
this subgroup: RGB-only's6.35% is barely above that trivial reference.
Thus recall97.89% does not demonstrate that RGB reliably localizes the
unreturned foreground. The declared Non-Veto trigger fails on IoU.

| Near area | Depth recall / IoU | NFO recall / IoU | RGB-only recall / IoU |
|---|---:|---:|---:|
|0–5%|76.00% /3.15%|66.32% /4.01%|96.92% /2.16%|
|5–10%|84.10% /9.64%|75.50% /11.70%|97.58% /7.76%|
|10–20%|90.20% /18.53%|87.44% /20.34%|98.17% /15.31%|
|20–50%|89.60% /36.67%|91.48% /38.21%|97.40% /33.76%|
|50–100%|94.85% /74.48%|96.97% /76.30%|97.45% /74.07%|

The independent RGB-only weight reload passed finite/nested output checks and
ToF-input independence. Preview examples use the same six predetermined images
as the initial matched report: greenTP, redFP, blueFN, purpleUNKNOWN. Visual
inspection confirms broad red near-prediction regions, consistent with counts.

Decision: **NEGATIVE_CONTROL** for the hypothesis that this matched RGB-only
recipe supplies spatially discriminative far-return foreground recovery at the
fixed operating rule. No Non-Veto fusion, completeness head, latent-surface
network or additional fit was started. The observations do not prove RGB lacks
information, nor that ToF causally suppresses an otherwise accurate RGB estimate.
The original matched NFO gain and its small-area limitation remain retained.

Deliverables: `rgb-only.pt`, `rgb-only-preview.jpg`, `results.json` (four
distances/all bins), `three-arm-table.json`, `inference-check.json`, protocol
and training logs in the evidence directory below. Cached test scores are
reproducible intermediates and are removed after delivery checks; actual weights,
metrics, examples and receipts remain. Global registration is still blocked by
the verified existing ledger303 input-fingerprint mismatch; no bypass.

This run adds exactly one trained RGB-only NFO to the retained matched depth
and NFO fusion comparison. The question is whether RGB supports useful recovery
of small near foreground when the simulated ToF publicly returns far background.
An observational subgroup difference alone does not prove a causal far-return
veto. Missing returns in a zone also do not eliminate neighboring ToF context.

Use the same 3000/500/500 Hypersim split, 256x192 images, seed190921,
12epochs, batch24, AdamW0.002/weight decay0.0001/cosine to10%, BCE plus0.2
ordinal loss, final epoch only. Initialize the entire original NFO, then delete
the ToF encoder and its16 input columns from the first fusion convolution.
All surviving initial parameter values are checked bit-identical to B.
RGB-only has246176 parameters versus258656 for fusion. No new backbone,
training data, loss, augmentation, fit budget or threshold grid is introduced.
This is a separately trained missing-modality architecture, not test-time zeroing.

The forward method cannot access zone values. A direct check verifies identical
output with absent versus arbitrary supplied zone arguments, and no ToF weights
remain. Geometry-derived area and public-return strata are evaluation only.
All known-depth pixels participate in the unchanged loss; UNKNOWN stays excluded.

Calibrate once on the same validation2m mixed domain at recall>=95%, maximizing
IoU on the original0.001..0.999 grid. Freeze one cutoff across all distances.
Report full/mixed/small domains and original five area bins, split by observed
near/far/missing return. Test data is consumed Development; no fresh claim.

Before fitting, define a useful-signal trigger for one subsequent Non-Veto
candidate: far-return <=20% foreground recall>=80% **and** subgroup IoU no lower
than original fusion NFO at its validation-selected cutoff. Recall alone could
be raised by excessive near prediction. If the trigger fails, stop this round;
do not train a fusion successor or tune test thresholds. A poor RGB-only result
does not prove RGB has no recoverable information at this resolution.

Script: `ba_nfo_rgb_control.py`. Evidence and actual checkpoint:
`artifacts.local/work/ba-nfo-rgb-control-20260919/`. Original evidence is unchanged.
The same CUDA encoder/decoder workload uses the retained backend-placement
measurement; record actual device and timing in this run. No A/A*, Radar,
BODY/HEAD or default application change is included.
