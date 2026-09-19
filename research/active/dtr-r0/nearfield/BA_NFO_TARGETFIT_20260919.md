# Same32cases, target-zone supervision: localization gain with full-image regression

One paired TRAIN-only loss-support contrast is complete. Restricting the
unchanged loss to the predetermined zones substantially improves their spatial
fit: IoU25.83%->60.00%, FP1,110->240, passing zones1/16->11/16, recall96%.
The exact fit still misses the sealed65%IoU and14/16zone gates. Full-image
performance regresses severely. Retain the diagnostic contribution, preserve
original NFO, and stop this fit without tuning or model promotion.

| Same32TRAIN frames,2m at0.081 | Original NFO | Full-image-loss fit | Target-zone-loss fit |
|---|---:|---:|---:|
| Target-positive recall | 71.25% | 97.50% | 96.00% |
| Target-positive IoU | 13.62% | 25.83% | 60.00% |
| Target-positive precision | 14.41% | 26.00% | 61.54% |
| Target-positive TP / FP / FN | 285 /1693 /115 | 390 /1110 /10 | 384 /240 /16 |
| Positive zones meeting recall90% and IoU50% | 2/16 | 1/16 | 11/16 |
| Target-negative FP / known pixels | 0 /7689 | 13 /7689 | 16 /7689 |
| Target-negative FPR | 0% | 0.169% | 0.208% |
| Full-image recall | 97.07% | 99.40% | 58.80% |
| Full-image IoU | 61.81% | 80.93% | 29.34% |
| All mixed-zone recall / IoU | 95.19% /52.75% | 98.57% /67.98% | 48.96% /32.54% |
| All pure-far FP | 21,214 | 4,937 | 70,192 |

Target-positive denominator remains400near/7,531far pixels. Against the
full-image-loss fit, target supervision removes1,020FP and adds150, rescues10TP
and loses16. Negative controls remove8FP and add11. Raw paired changes and
all16positive-zone results remain in verification.json/results.json.
The two passes are target recall>=95% and target-negative FPR<=1%; the two
failures are target IoU>=65% and at least14/16positive zones meeting the local
criteria. The predeclared targets were not lowered after results were seen.

## Controlled change

Reference: [previous hard-case fit](BA_NFO_HARDFIT_20260919.md).
Reuse its exact32TRAIN identities,16positive/16negative zones and32distinct
scenes, selected from the original manifest with disclosed depth/texture rules.
The same target-zone selection bytes are copied and hash-verified; no new
selection, calibration, validation or test set is accessed.

Restart from the **original trained NFO**, not the previous hardfit weights.
Before training, all32four-head outputs reproduce the previous initialization
bit-for-bit. Preserve full256x192RGB, actual public8x8ToF, architecture,
four-head BCE+.2ordinal formula, AdamW.002/.0001, clip5, seed190921,
batch8, exact512sampled batches,512updates, constant LR and final-step-only
checkpoint. Both arms use cutoff0.081 and identical diagnostic gates.

The sole experimental change is the loss's supervised pixel domain. Set depth
labels outside each selected complete zone to NaN only in the loss call, then
call the original loss function. Its known-pixel mean now spans15,620pixels
across the32zones, including all near and far pixels and all16negative controls,
instead of1,558,122known full-image pixels. UNKNOWN stays excluded.
The network never receives the zone mask or depth truth; its full RGB/ToF
inputs and all-image output remain unchanged. There are no learned zone
weights, class weights, auxiliary losses or inference gates.

## What this changes in the diagnosis

On this exact seen set and budget, supervision domain strongly affects small
support localization. Eleven zones now meet the local criterion, including
one5-pixel target recovered exactly. This is evidence against treating the
previous diffuse masks as proof of an unavoidable input/architecture ceiling.
It is not evidence that all16cases can already be fitted: three failing zones
retain excess FP, one6-pixel zone retains only2TP, and one4-pixel zone is lost.

The intervention changes gradient allocation and the supervised label
distribution together. It supports a supervision-domain contribution under
the matched recipe, not a universal proof that pixel count alone caused the
earlier failure. It also does not establish any new optimal loss weighting.

Removing full-image supervision sacrifices performance elsewhere, including
on these same seen images. The target-only checkpoint is unsuitable as a
replacement for retained NFO: full recall58.80%, mixed recall48.96%, and
70,192pure-far FP are material costs. Gains on the chosen zones cannot hide
that collapse. No validation/test inference was performed; there is no
generalization, hardware, alert or safety claim.

Retain as COMPONENT diagnostic evidence only. End this exact contrast without
extra steps, threshold sweeps, loss mixtures or resolution variants. The result
does not authorize an automatic successor or change original NFO/App defaults.

## Verification and delivery

[Implementation](ba_nfo_targetfit.py) reuses the previous selection/evaluator.
[Two focused tests](test_ba_nfo_targetfit.py) verify full-support loss/gradient
identity, zero output-loss gradients outside target/at UNKNOWN, and invariance
to outside depth-label changes. Both pass. These zero-gradient assertions
concern output supervision, not a claim that outside RGB context has no influence.

An independent process reloads the actual final weights and feeds only RGB
and public ToF. All32four-head outputs match bit-for-bit, all frame/domain
metrics reproduce, selected-target confusion counts are independently recomputed,
loss-support arrays and original512batch order match exactly, outputs remain
nested, and the original checkpoint hash is unchanged. The fit took16.68s on
CUDA NVIDIA GeForce RTX5060 Laptop GPU; this is host training time.

Evidence: `artifacts.local/work/ba-nfo-targetfit32-20260919/` retains protocol,
selection, batch plan, loss-support masks, baseline, real final checkpoint,
512-step log, scores, per-frame/per-zone metrics, previews, verification,
disposition and command receipts. Original scores/weights are referenced at
their existing evidence roots. The fit and verification processes exited;
no worker or next fit remains running.

Registration reports the existing ledger303 input-fingerprint mismatch;
global inheritance remains pending. Local structured disposition and command
receipts are retained without repairing or bypassing that ledger.
