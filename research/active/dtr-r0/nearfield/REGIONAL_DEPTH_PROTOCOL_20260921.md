# Regional measurement learning: one local-depth algorithm comparison

EXPLORE, user authorized on 2026-09-21 after whole-object oracle closure.
Question: can a regional ToF measurement operator improve Boundary rescue at
the retained false-alert budget, without Core event or first-alert loss?
This is a runnable algorithm experiment, not another coverage/oracle diagnosis.

## Mechanism and contrast

Current public ToF is an optical-Z simulation proxy, not radial range. Its
generator chooses the dominant inverse-square-energy 10cm depth bin, reports
its noisy mean and may drop a zone. A return is not a point at the zone center.
MZ140/142 disclosed a regional-to-center approximation in a different source;
their failures do not prove that approximation caused their negative results.
DELTAR motivates regional fusion (https://zju3dv.github.io/deltar/), but uses
richer measured distributions. This candidate receives only original64 ranges,
validity, boxes and RGB; it invents no measured histogram or perfect association.

New identical POINT/REGION encoder-decoders take raw RGB detail, repeated public
range/8 and validity plus exact X/Z,Y/Z rays. Eight by eight point samples per
zone form64x64 pixels, from the existing192x256 lattice and exact native640x360
pixel centers; no interpolation of measured depth or native-label input at test.
The predicted field covers the nominal ToF footprint, not the whole image or
occluded object. RGB continuity and training-learned surface patterns are model
priors; all predicted depth and intrusion probabilities remain predictions.

Each pixel has81 categorical depth outputs:80 bins covering[0,8)m at.1m width
and a finite>=8m overflow. Missing native depth remains -100/UNKNOWN, not far.
The common network has24-channel full-resolution features,32-channel half and
48-channel quarter branches with dilation2, bilinear feature upsampling, and
one48-channel decoder followed by81 logits. No pretrained model or scalar head.

Both arms use dense visible native-depth cross-entropy plus .25 measurement
loss and .25 frame-alert BCE. REGION averages predicted categorical distributions
over64 samples in each zone; POINT uses sample[4,4] only (a central sample,
not the exact continuous zone center). Both weight eligible bins1..79 by
1/max(z,.3)^2, normalize energies, then softmax at temperature.1. Measurement
loss is negative log probability that this soft winner lies in the unchanged
public interval r +/- (.1+3*(.01+.02*r)). Missing returns contribute zero loss.
This is a disclosed differentiable surrogate for dominant-bin sensing, not an
exact simulator likelihood, observed histogram or deterministic physical bound.
The two arms differ ONLY in regional versus center aggregation in that loss;
their input tensors, network, supervision, loss weights and decoder match.
No winner IDs, native inverse-depth weights or target masks enter inference.

Alert score is logit(mean of the16 highest predicted corridor probabilities).
For each pixel, sum probability over bin centers satisfying .3<=Z<=3,
abs(X)<=.3, -.2<=Y<=.9. This .1m quantized decoder is an explicit approximation;
it does not certify continuous or whole-object extent. Native depth supervises
training; full original obstacle truth governs frame/event evaluation.

## Fixed data, budget and acceptance

Use original24 train layouts/1728 frames,8 dev layouts/576 frames, and all16
consumed transfer layouts/1152 frames. Their complete groups are disjoint.
Transfer has already shaped research choices: this is reused Development,
never fresh confirmation. Do not access original protected-test pixel payloads,
labels, logits or metrics. No new capture, hardware input or protected test.
Training native payloads are selected by train indices before loading. Native
dev/transfer payloads are only loaded for local metrics after prediction sealing.

One fit per arm: seed20260921,1200 AdamW updates, batch16, lr.001,wd.0001,
uniform replacement, identical initial parameters and batches, final checkpoint.
No augmentation, tuning, checkpoint/seed/weight retry or automatic successor.
Synthetic engineering checks and disposable CPU/GPU workload probes do not
select scientific parameters; record and discard their weights. Benchmark the
complete REGION training loss and use one common selected device for both arms.

A is frozen Core; preserve original A/B/R/U/G saved outputs as controls. Each
new current decision is A OR its score above its dev cutoff; apply the unchanged
nonrecursive one-frame .2s hold. Preserve A's UNKNOWN for both readouts. This
cannot remove A's existing false alerts and must retain all A alerts/onsets.

Select each cutoff once on dev: lowest inclusive float64 score satisfying Core
and Boundary separately, added currentFP<=floor(.01*Nnegative), added heldFP
<=floor(.02*Nnegative), added false segments<=floor(.125*Nclips) per readout.
Include nextafter(max,+inf) (disabled supplement). Do not select on transfer.

Usability: all transfer cost caps pass, all A current/held positive alerts and
Core events/first-alert times are retained; Boundary held recall>=.50, gains
over A in>=8/16layouts. Report all four families, including HEAD-horizontal,
without adding a special posthoc family gate or hiding their failures.
Regional contribution: REGION is usable, adds>=5 Boundary held TP over POINT
at these same budget rules, and improves Boundary TP in>=2 transfer layouts.
Same budget rules do not mean identical realized FP counts; report those costs.
If either arm is useful, retain its scoped Development capability. Only the
contribution criteria support the new regional operator. If neither is useful,
close both exact recipes with their geometry evidence; no rescue sweep follows.

## Evidence and delivery

Seal source/protocol/code hashes, observations, train labels, training receipts,
checkpoints, dev scores and cutoffs. Seal transfer scores, per-pixel corridor
probabilities, depth predictions and all alerts BEFORE evaluation joins.
Report current/held TP/FP/FN, precision/recall/FPR, event timing and missed
events, false-alert segments/durations, layout/family/layer/background groups,
UNKNOWN, native-contributor status of new TP, and host costs. Local geometry
metrics use valid sampled rays: precision/recall/IoU at probability.5, known
depth-bin accuracy and finite<8m depth MAE, separate from final alert evidence.
Missing geometry is masked, never a negative. No local-label coverage admission
gate and no claim that pixel metrics demonstrate obstacle or safety benefit.

Provide observation-only inference from checkpoint+RGB+64ranges+boxes; it must
reproduce saved scores on a focused replay. Preserve original negative terminals,
global ledger303/unknown-terminal receipts and local structured inheritance if
supported registration remains blocked. Finish the result, owning current and
scoped normal commit/push; release task-owned resources. No App/demo promotion.
