# BA-NFO: direct near-field occupancy, preparation protocol

This is the final proposed public-data training route after closing DEPTHOR
repair, ZPA, SCDE and RPCC. It is **not trained yet** because the required
dense training assets are absent from the checkout.

## Question and fixed comparison

Can a small task-specific RGB plus simulated 8x8 ToF model directly predict
near-field occupancy better than a general dense-depth model? Freeze four arms:
RGB-only, ToF-only, RGB+ToF, and RGB+ToF+ordinal consistency. Compare all against
raw ToF expansion, RGB metric monocular depth, and frozen DEPTHOR. No DEPTHOR
refinement, Radar, IMU, temporal input or final alert logic enters the first fit.

Each output has four probabilities: depth below 1.0, 1.5, 2.0 and 3.0 m. The
ordinal penalty enforces nondecreasing probabilities. Labels come from dense
camera-axis depth; reference depth never enters model inputs. ToF simulation is
fixed before training and uses zone area, missingness, range noise, multi-target
and partial-view behavior with one identical generator for all labels.

## Data and split

Hypersim supplies exact indoor geometry; SANPO-Synthetic supplies first-person
outdoor variation; UE supplies thin rods, chair legs, door frames, signs,
suspended head bars, grazing corridor edges and foreground/background mixed
zones. Split by complete scene/sequence, never random frames. Target size is
10,000–30,000 frames after manifest freeze. ZJU-L5 remains a real 8x8 ToF
sanity check only: its reference/depth package has no BODY/HEAD corridor truth.

The existing MZ120/MZ122 assets are not interchangeable labels: they supervise
45-cell native AABB volume occupancy, not four dense pixelwise distance masks.
Reusing them would collapse the stated hypothesis into the already-consumed
occupancy route and hide the input/label change.

## Minimal model and loss

Use a frozen small RGB encoder, an 8x8 ToF zone encoder with range/status and
missingness, two or three fusion blocks, and a tiny decoder. First recipe:
weighted BCE/focal near-mask loss plus a small ordinal monotonicity penalty;
near false negatives receive a bounded higher weight. Do not sweep backbones,
losses, thresholds or augmentation. Record actual GPU/backend and end-to-end
model timing.

## Acceptance and stop

Primary mixed-zone 2m gate: recall at least frozen DEPTHOR's 95.69% and IoU at
least 78.40% (+3 points over 75.40%). Also report 1/1.5/3m, scene-heldout
results, unknown/coverage and threshold bands. UE task evidence must show no
loss of event recall with a meaningful FP reduction under the same fixed
readout; absent BODY/HEAD truth is NOT_EVALUABLE. One failed gate closes this
public-data direct occupancy route. No automatic successor or larger training
budget follows.

## Current preflight decision

`artifacts.local/work/ba-nfo-preflight-20260918/preflight.json` finds no local
Hypersim or SANPO-Synthetic roots and no dense UE RGB+depth+ToF manifest. It
records the exact frozen protocol and stops before download/training. No old
MZ120 labels, ZJU corridor assumptions, or synthetic pseudo-data are substituted.
