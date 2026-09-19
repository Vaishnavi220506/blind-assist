# Native support audit: sampling is partial; zone-local size is not object size

The 16 sealed positive TRAIN cases were inspected at native 1024x768 and input
256x192 resolution, using their existing selected zones and saved joint-fit
predictions. No training, new inference, validation, test or sensor regeneration
was performed. Existing metrics and all failures remain unchanged.

The result does **not** support input averaging as the main explanation for the
remaining joint-fit errors. It does establish an evaluation-scope distinction:
zone-local small near support is not sufficient evidence of an independent small
obstacle. Preserve this cohort as a zone-conditioned near-support diagnostic; do not
use its 14/16 gate as a universal verdict on small-obstacle recovery.
The user's task of localizing a small near patch inside a mixed ToF cell remains
valid even when that patch crosses the cell boundary. Cross-zone support does
not excuse the 11/16 failure or authorize dropping any of the 16 examples.

## Measured findings

- All 16 native RGB/depth source hashes and prepared-array hashes match their
  sealed manifest. Reapplying the original conversion reproduces RGB and depth
  exactly. Each input pixel covers a 4x4 native block; RGB averages that block,
  while nearest-neighbor depth samples its top-left native position.
- Of 400 nearest-neighbor near labels, 38 (9.5%) have less than 50% native-near
  footprint coverage; 24 (6.0%) have at most 25%. Most positive labels therefore
  are not foreground-minority averages. This alone does not measure RGB contrast
  or prove that a network can recognize them.
- Of 192 saved joint-fit false-positive pixels, 125 (65.1%) have zero native-near
  coverage. The other 67 have some near coverage; 34 have at least 50%. Sampling
  semantics explain an ambiguity for a subset, not the majority of the FP.
- Four of five FN come from one missed target. Their mean native-near footprint
  coverage is 96.875%, all at least 50%, with no unknown depth. They were not
  averaged into background-majority RGB pixels. The remaining FN has 12.5% near
  coverage, consistent with strong mixing. Neither result proves RGB visibility.
- In all 16 cases every connected near component intersecting the selected box
  in the input-resolution mask also extends outside it. At native resolution
  every component extends outside in 15/16; the remaining case has a large
  crossing component plus tiny isolated 1-2-pixel components. Thus every case
  contains cross-zone support. Connectivity uses 8-connected depth<2m and is
  **not** a semantic instance annotation; touching surfaces can merge.

## All-case inventory

`Minority` counts near-labeled input pixels whose native footprint is less than
half near. `FP empty` counts saved joint FP with zero native near coverage.
Pass/fail is the unchanged individual recall/IoU criterion from the joint fit.

| Scene / camera / frame | Near | Minority | FP | FP empty | FN | Original zone gate |
|---|---:|---:|---:|---:|---:|---|
| ai_011_008 /00 /0006 | 88 | 9 | 6 | 6 | 0 | Pass |
| ai_050_003 /03 /0047 | 34 | 5 | 11 | 11 | 0 | Pass |
| ai_041_004 /00 /0053 | 35 | 4 | 18 | 14 | 0 | Pass |
| ai_039_002 /00 /0025 | 24 | 7 | 12 | 12 | 0 | Pass |
| ai_016_009 /03 /0073 | 23 | 0 | 4 | 0 | 0 | Pass |
| ai_003_004 /01 /0056 | 7 | 3 | 15 | 15 | 0 | Fail |
| ai_050_004 /04 /0065 | 9 | 2 | 23 | 11 | 0 | Fail |
| ai_050_001 /03 /0057 | 80 | 0 | 22 | 7 | 0 | Pass |
| ai_048_008 /03 /0098 | 5 | 0 | 5 | 0 | 0 | Pass |
| ai_003_010 /01 /0030 | 11 | 0 | 8 | 2 | 0 | Pass |
| ai_004_008 /00 /0087 | 26 | 6 | 8 | 8 | 0 | Pass |
| ai_054_008 /00 /0064 | 6 | 0 | 4 | 4 | 0 | Pass |
| ai_006_002 /00 /0014 | 8 | 0 | 6 | 0 | 0 | Pass |
| ai_048_010 /02 /0044 | 4 | 0 | 0 | 0 | 4 | Fail |
| ai_048_009 /00 /0037 | 5 | 2 | 0 | 0 | 1 | Fail |
| ai_015_009 /00 /0057 | 35 | 0 | 50 | 35 | 0 | Fail |

## Visual inspection and interpretation

All four contact sheets (all 16 cases) were inspected. Native context reveals
chair/sofa edges, furniture surfaces, frame-like edges, a lamp edge and bathtub
rim among these selected patches. These are visual descriptions, not source
instance labels or proof of physical obstacle size. The five failures contain:

- Easel/support-edge patch: all 15 FP are on zero-near footprints.
- TV/stand-region patch: 11/23 FP are zero-near, 5/23 majority-near.
- Camera-like object at the edge of a sofa-region patch: all four near labels
  are missed despite high native footprint coverage. Native near support extends
  well beyond the chosen zone; the low-resolution image still contains structure.
- Lamp-edge patch: only one FN, whose footprint is 2/16 native near pixels.
- Bathtub-rim patch: 35/50 FP are zero-near, 7/50 majority-near.

Sampling mismatch is a difference between point-ray labels and area-integrated
RGB. It is not automatically corrupt data, spatial misregistration or a reason
to replace the old labels with majority occupancy. Full-footprint and
known-only occupancy are recorded separately; unknown is never counted as far.
All near-labeled footprints in these selected zones have fully known depth.
The original ToF is a fixed simulated regional return and is unchanged; this
audit says nothing about physical sensor misses.

## Decision and next useful check

Keep original NFO, joint-supervision diagnostic evidence, the 11/16 failure,
and the closure of the fixed supervision-allocation sequence. Do not start a
high-resolution training run on the assumption that downsampling explains the
remaining errors: the present audit does not supply that premise. Nor does it
rule out a future benefit of high resolution or establish an architecture ceiling.

For the existing mixed-cell task, the next useful capability question concerns
spatial localization: can a changed spatial representation reduce empty-near
footprint FP while recovering the high-occupancy missed support, with the same
inputs, cases, loss and cutoff? This would require a separately bounded contrast;
the audit has not identified a validated architecture or authorized another fit.
Do not replace that question with a new easier cohort. If independent physical
small-object claims are later needed, add source-native instance identity and
whole-object projected extent as an extra evaluation stratum, selected without
model scores; do not rewrite the original task or prior denominators. No successor
dataset selection, training or evaluation was executed in this audit.

## Reproduction, checks and evidence

[Audit script](audit_ba_nfo_native_support.py) reads only the 16 selected TRAIN
source pairs and the already saved joint-fit predictions. It verifies file
hashes, exact RGB/depth regeneration, nearest-sample coordinates, independent
4x4 block averaging against OpenCV area reduction, and original near counts.
An independent reduction of saved audit arrays checks every confusion count
against joint-fit results and recomputes totals. All 16 visual comparisons were
inspected. CPU is used for file decoding, small exact-array reductions and
connected components; no neural compute or claim of accelerator speedup.

Evidence root: `artifacts.local/work/ba-nfo-native-support-audit-20260919/`.
Retain protocol, source hashes, original-footprint fractions and error arrays,
per-case metrics, full contact sheets, verification, disposition and delivery
receipts. No task worker or temporary runtime remains. Existing global ledger
line 303 registration failure is recorded separately; local evidence does not
bypass that metadata failure.
