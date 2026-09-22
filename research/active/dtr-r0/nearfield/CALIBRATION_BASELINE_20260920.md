# Calibration adopted as the camera-forward branch baseline

2026-09-20. User-authorized branch decision; existing consumed evidence audited.

**Adopt Calibration as this camera-forward controlled-research branch's baseline.
Disable the exact RGB and matched noRGB veto heads in this branch's active method.**
Retain their frozen code, checkpoints, predictions and negative results as controls.
This decision does not alter the historical four-arm runner or Android defaults.

The same 36-clip / 432-frame evidence gives Calibration **144 TP / 146 FP / 0 FN**,
precision 49.66%, recall 100%, F1 66.36%, FPR 50.69% over 288 negatives;
OUTSIDE-layout FP is 87/144, all 24 events and their first in-event alerts survive.
Raw has 144/181/0, F1 61.41%: Calibration removes 19.34% of FP and 17.92% of
OUTSIDE-layout FP, increasing F1 by 4.95 percentage points. UNKNOWN stays 391;
TN stays zero. Pixel IoU is not defined for this scalar alert task.
See the immutable [transfer result](CORE_TRANSFER_RESULTS_20260920.md).

## Proposed soft footprint is already implemented

The premise that Calibration replaces a zone with a calibrated point does not
match the implementation. Even Raw tests the complete zone support for possible
and definite intersection. [Calibration](tof_corridor_calibration.py) already
integrates the fraction of the full two-dimensional ray-slope rectangle inside
the corridor over the original depth interval:

```
zone_score = depth_fraction * conditional_footprint_overlap
frame_score = max(zone_score among original possible zones)
alert = original definite support OR (original possible alert AND score >= threshold)
threshold = 0.007085703945147101
```

The corridor is the unchanged camera-frame volume X=[-0.3,0.3], Y=[-0.2,0.9],
Z=[0.3,3] metres. Its angular intersection varies with depth; a fixed +/-10-degree
cone would change the task. Calibration here selects a scalar operating point
on the old 96-frame Development set. It does not estimate sensor extrinsics,
move measured points or fit physical per-zone angle uncertainty. The uniform
ray-slope/depth measure is a geometric score, not a calibrated probability.

The [standard-library audit](audit_calibration_footprint.py) validates the original
observation/prediction/evaluation seals, 432 observation hashes and both original
readout code hashes. All 22,371 saved valid zone samples satisfy the factorization
(maximum floating-point discrepancy 5.56e-17); reconstructed decisions match
432/432 Calibration alerts. All original event onsets also match Raw.

Consequently the requested hard-to-soft replacement supplies no new predictor.
Multiplying the current score by overlap again would apply spatial downweighting
twice, a different hypothesis. No such weighting experiment, threshold search,
new model, capture, or temporal successor was run. This audit is not a negative
result for a newly tested soft method and cannot claim a new FP improvement.

## What 37 to 40 false segments actually contains

Calibration's alerts are a subset of Raw's on every frame. Mapping each of the
37 original FP segments to retained children gives:

| Original segment disposition | Count | Resulting segments |
| --- | ---: | ---: |
| Completely removed | 1 | 0 |
| One retained segment, possibly shortened | 32 | 32 |
| Split into two | 4 | 8 |
| Total | 37 | 40 |

Five internal withheld frames make four gaps. Three gaps occur before contact
in BOUNDARY-layout clips; one occurs in an OUTSIDE-layout clip. All are in
background_1 arrangements. Sampled FP duration still falls 36.2 to 29.2 seconds.
Neither these durations nor segment counts measure actual demo annoyance.

| Clip suffix (all core_b1_) | Gap frames | Score before / gap / after | Winning zone sequence |
| --- | --- | --- | --- |
| body_suspended_solid_boundary | f0376 | .256406 / .002124 / .358519 | 43 / 42 / 51 |
| body_suspended_solid_outside | f0389 | .008066 / .006619 / .030244 | 50 / 42 / 42 |
| head_hanging_plane_boundary | f0267-f0268 | .034338 / .001390,.002079 / .019983 | 36 / 37,29 / 37 |
| head_protruding_edge_boundary | f0304 | .175863 / .000976 / .024019 | 35 / 26 / 34 |

All five withheld frames still have possible support but their continuous score
falls below the frozen threshold. The winning zone changes entering every gap.
The possible-zone set changes across three windows; it stays {42,50} in the
OUTSIDE window. Conditional angular overlap in gap-winning zones is only
0.43%-1.86%, while their depth fractions are 22.21%-35.57%. These facts locate
threshold crossings in the existing soft readout. They do not isolate sensor
noise from moving geometry, dropout or winning-zone changes, and do not establish
an inside/outside hard-membership toggle or 1-2 degree calibration jitter.

## Disposition and evidence boundary

The versioned [baseline declaration](CALIBRATION_BASELINE_20260920.json) records
Calibration as `RETAINED_CORE`, exact RGB/noRGB vetoes as `NEGATIVE_CONTROL`, the
unchanged input/threshold bindings and the user-adopted active method. Closing
these heads does not reject all possible uses of RGB.

The existing global registry is blocked by ledger303/unknown-terminal errors;
the current inheritance attempt and its output are retained in the audit payload.
The branch decision is effective locally and in this versioned declaration;
global structured inheritance is not claimed complete or bypassed.

Evidence remains controlled simulator arrangement transfer. Reusing the 432
frames here is consumed-data diagnosis, not another fresh confirmation, hardware
calibration, physical uncertainty estimation, natural-scene or safety validation.
No double-threshold event layer is started: there has been no new soft-method
experiment satisfying the user's conditional next-step premise.

Durable audit: `artifacts.local/work/ba-calibration-footprint-audit-20260920/`.
It retains input hashes, every original-segment mapping, gap frame scores,
depth/angular decomposition, winning-zone intervals, event timing and receipts.
The audit uses CPU standard-library scalar/hash operations (`TASK_NOT_GPU_SUITABLE`).
No persistent task processes, paid allocation, models or disposable datasets were
created. Frozen source artifacts and negative controls remain intact.
