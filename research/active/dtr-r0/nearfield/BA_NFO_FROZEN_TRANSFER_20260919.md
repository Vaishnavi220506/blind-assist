# Frozen transfer check: both 32-frame adaptations fail the original task targets

One fixed comparison on all 500 original validation/Development frames is
complete. The joint-loss and late-fusion checkpoints improve selected training
targets but **do not transfer the requested joint benefit** at the fixed 0.081
operating point. Joint meets zero of four user targets; late fusion meets only
the IoU-retention target. Both reduce far-small and mixed recall in every one
of the six validation scene families. Preserve original NFO as the reference,
reject these two frozen adaptations as transfer upgrades, and end the 32-frame
adaptation sequence without checkpoint, threshold, loss or structure repair.

Original NFO itself still misses the user's 75% far-small recall target. Retaining
the reference does not mean the small-near task has been solved.

## Same-cohort comparison

These are **500-frame Development results**, not the 32-frame training metrics
or the original 500-frame test numbers. Far-small keeps the original definition:
public ToF>=2m, 0<near/known area<=20%, known truth depth<2m. The frozen four user
targets are recall>=75%, IoU>=same-cohort original NFO, mixed recall>=94.5%, and
pure-far FP<=same-cohort original NFO. No new calibration is performed.

| 2m metric at0.081 | Original NFO | Joint-loss fit | Late-fusion fit |
|---|---:|---:|---:|
| Far-small recall | 68.36% | 62.37% | 60.12% |
| Far-small IoU | 11.13% | 10.89% | 11.64% |
| Far-small TP / FP / FN | 13,134 / 98,788 / 6,079 | 11,984 / 90,809 / 7,229 | 11,550 / 79,997 / 7,663 |
| Mixed recall | 95.00% | 87.80% | 87.25% |
| Mixed IoU | 51.16% | 50.43% | 51.36% |
| Pure-far FP | 212,988 | 367,659 | 273,576 |
| Pure-far FPR | 2.106% | 3.636% | 2.705% |
| All small-support recall, any ToF return | 80.65% | 71.97% | 70.44% |
| All small-support IoU | 12.79% | 12.79% | 13.48% |
| Full-image recall | 96.98% | 87.46% | 89.89% |
| Full-image IoU | 61.62% | 57.33% | 58.60% |
| Outside ToF coverage recall | 96.02% | 81.35% | 86.54% |
| Four user targets met | Reference: 3/4 | 0/4 | 1/4 |

Denominators are unchanged across arms: far-small19,213near/273,531far,
mixed632,572near/871,862far, pure-far10,112,945known pixels, and
full-image3,123,846near/21,102,014far. All350,140unknown-depth pixels are excluded
from scoring, never converted to negatives. Full domain/frame/family records
are retained, including public-far and outside-coverage results.

Late fusion's far-small IoU improvement is accompanied by 1,584 extra FN and
60,588 extra pure-far FP versus original NFO. Its better precision on far-small
zones does not compensate for failed recall and pure-far rejection targets.
Joint fitting adds1,150far-small FN and154,671pure-far FP. Neither is a suitable
replacement at this fixed operating point.

## Paired changes and breadth

| Versus original NFO | Joint fit | Late fit |
|---|---:|---:|
| Previously detected far-small near pixels lost | 2,727 | 2,855 |
| Previous far-small misses recovered | 1,577 | 1,271 |
| Previous mixed near hits lost | 57,821 | 59,253 |
| Previous mixed misses recovered | 12,282 | 10,254 |
| New pure-far FP | 239,633 | 158,157 |
| Old pure-far FP removed | 84,962 | 97,569 |

The validation cohort has 52 scenes in six scene families. Both candidates
reduce far-small recall and mixed recall in **all six families**, not just one
outlier. Joint pure-far FP rises in all six; late pure-far FP rises in five and
falls in one. Pixels are correlated within scenes; their large count is not a
claim of independent sample size or statistical significance.

| Family | Original far-small recall | Joint | Late |
|---|---:|---:|---:|
| ai_017 | 55.42% | 49.84% | 46.27% |
| ai_023 | 90.00% | 83.28% | 80.98% |
| ai_033 | 71.19% | 64.88% | 62.69% |
| ai_035 | 77.62% | 65.96% | 60.84% |
| ai_047 | 83.93% | 80.39% | 77.25% |
| ai_051 | 64.79% | 61.93% | 63.02% |

Descriptive depth strata do not rescue the result. On the original separated
near/far + ToF>=2.2m stratum, recall is49.79% original,43.92% joint,41.73% late.
For far-small near pixels<=1.8m, recall is60.23%,55.29%,51.02%; for(1.8,2)m it
is70.83%,64.53%,62.88%. All original pixels remain in the main denominator;
these strata are not new gates or grounds to discard failures.

## Fixed protocol and evidence boundary

The user explicitly authorized this comparison after the research retrospective.
It is a separate transfer diagnostic; it does not retroactively pass or alter
the prior joint11/16 or late12/16 training gates or their skipped-validation
receipts. No old experiment was rerun, extended or recalibrated.

Use the original manifest's complete500`val` rows in original order. Assert zero
image-ID, scene and family overlap with both the original3,000training frames
and the32micro-fit frames. The source is still synthetic with related assets.
This cohort previously participated in original cutoff calibration and other
evaluations, so it is **consumed Development**, not fresh blind confirmation.
The original500`test` observations are not accessed in this task.

All three models and their file hashes are fixed before inference. Forward sees
only contiguous full256x192RGB and the unchanged public8x8ToF fields. Use the
original four-head sigmoid/cummax and the same0.081cutoff, eval/inference mode,
all parameter gradients disabled, batch24. No ranking/threshold sweep,
checkpoint selection, extra epoch, weight mixture or candidate omission.

Baseline IoU and FP targets use this cohort's actual original values11.1304%
and212,988. Applying the old test's12.09%/157,187numbers here would mix
denominators. The user's absolute75%far-small and94.5%mixed recall targets remain
unchanged. Outside/full/all-small costs are reported alongside the four targets.

## What this changes in the research decision

The earlier selected-target99.5%recall/70.07%IoU does not establish improved
cross-scene performance. This check directly shows that the two32-frame
adaptations fail the desired transfer comparison. Their local learning and
ablation evidence remains useful, but cannot support a promoted method.

Stop modifying these checkpoints to fix this Development outcome. Preserve
original NFO, all negative outcomes and the frozen candidates as references.
Any later candidate should return to representative training coverage and an
explicit cross-scene evaluation question, rather than optimize the same400
selected near pixels. No such new training is part of this delivery.

The result does not prove late ToF fusion, every loss weighting, or spatial
representation research generally ineffective. It also does not isolate
overfitting, forgetting and calibration as separate causes. It rejects these
specific frozen adaptations at the required operating point; no hypothetical
retuned result is substituted for their measured failure.

## Verification and artifacts

[Implementation](ba_nfo_frozen_transfer.py). All six original baseline domain
confusion matrices reproduce exactly against the prior saved validation result.
All checkpoints and input hashes remain unchanged. An independent evaluator
reconstructs all domains from source arrays and recounts the packed predictions
for all500frames/three arms and all six families; every confusion count matches.
It then reloads all three actual checkpoints on the predeclared first24 and
last20frames with the same batch layouts: all132output masks match bit-for-bit.
Independent read-only review found no material implementation or claim issue.

Actual inference uses CUDA on the host RTX5060 Laptop GPU, reusing the retained
architecture placement evidence. The three-arm pass took13.93s including I/O
and evaluation; this is not phone latency. No optimizer, training worker or
subsequent candidate experiment runs. All processes exited; `-B` avoids task
bytecode caches.

Evidence root: `artifacts.local/work/ba-nfo-frozen-transfer500-20260919/` retains
protocol, fixed manifest, compact lossless masks, per-frame/domain/family metrics,
paired changes, independent verification, dispositions and delivery receipts.
Source checkpoints are referenced without duplicating them. Global registration
and inheritance still fail at the existing ledger line303 fingerprint issue;
actual command receipts and local structured disposition are retained without
altering or bypassing that ledger.
