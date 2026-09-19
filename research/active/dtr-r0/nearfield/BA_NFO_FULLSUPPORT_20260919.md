# NFO full-training support supervision

Status: `REJECT_EXACT_FULLSUPPORT_RECIPE`. Both full-training fits and the
fixed500-frame Development evaluation are complete. Retain original NFO;
this loss recipe fails the recall requirements despite better small-area IoU
and fewer pure-far false positives. No repair or successor was started.

## Result on the same 500 Development frames

| Metric | Original retained NFO | Fresh matched control | Full-support loss |
|---|---:|---:|---:|
| Far-small recall | 68.360% | 68.271% | 62.562% |
| Far-small IoU | 11.130% | 11.247% | 12.301% |
| Mixed recall | 95.000% | 95.046% | 90.402% |
| Pure-far FP | 212,988 | 208,202 | 196,151 |
| All-small recall | 80.648% | 80.621% | 73.326% |
| All-small IoU | 12.788% | 12.853% | 14.848% |
| Full-image recall | 96.982% | 97.038% | 96.017% |
| Full-image IoU | 61.621% | 61.596% | 59.783% |
| Outside-ToF FP | 1,003,881 | 1,015,544 | 1,254,187 |
| User targets passed | Reference:3/4 | 3/4 | 2/4 |

The candidate misses both75% far-small recall and94.5% mixed recall. It also
fails matched-control recall guards. Same data, initialization and training
orders do not make GPU fits bit-identical: the fresh control has slightly
different final weights from the historical model, with far-small recall
only0.0885 percentage points lower. The candidate's5.7097-point loss relative
to this matched control is not credited as ordinary baseline improvement.
The single seed does not estimate across-seed uncertainty.

Far-small TP/FP/FN are13,117/97,417/6,096 for control and
12,020/78,501/7,193 for candidate. Candidate loses1,637 control near hits and
rescues540 misses:1,097 net additional missed near pixels. It removes28,807
old far-small FP but adds9,891 new ones. Pure-far removes79,501 FP and adds
67,450, net12,051 fewer. These are pixel counts, not obstacle/event counts.

| Development family | Control far-small recall | Candidate | Control mixed recall | Candidate |
|---|---:|---:|---:|---:|
| ai_017 | 55.436% | 50.468% | 94.411% | 91.045% |
| ai_023 | 89.180% | 78.361% | 94.878% | 91.234% |
| ai_033 | 71.015% | 65.939% | 94.787% | 89.656% |
| ai_035 | 77.337% | 74.150% | 96.478% | 93.753% |
| ai_047 | 82.360% | 77.079% | 96.847% | 93.744% |
| ai_051 | 65.290% | 57.398% | 94.216% | 87.089% |

Both recalls decline in all six families. Far-small recall also falls for
depth<=1.8m (60.452% to54.415%) and1.8–2m (70.645% to65.034%); retaining
boundary cases in training did not restore their fixed-cutoff recall.
Outside-ToF FP grows238,643 relative to control, so fewer pure-far FP is not
a claim of globally reduced false positives. Full-image FP grows95,901.

Compared with the prior frozen joint32/late32 negatives, this recipe preserves
more mixed recall (90.40% versus87.80%/87.25%) and reduces pure-far FP
(196,151 versus367,659/273,576), but still fails the original task. Different
initialization, losses and training budgets prohibit attributing this comparison
to expanded coverage alone. The result weakens the practical explanation that
simply moving this style of targeted supervision to full TRAIN would solve the
problem; it does not prove all losses or this architecture incapable.

Close this exact loss recipe at the frozen operating point, preserve its
precision/recall tradeoff as negative-control evidence, and retain original
NFO as reference. The75% target remains unmet. Do not escalate these weights
through another cutoff, coefficient, training-budget or branch sweep.

## Frozen protocol

Question: can supervision across the complete original training distribution
improve small-near support on unseen training families at the fixed original
cutoff? The 32-frame joint/late adaptations failed transfer and remain negative
controls; their local fitting gains are not generalization evidence.

One paired EXPLORE run uses all original 3,000 TRAIN frames (335 scenes,
39 families), the original 258,656-parameter NFO and exact original random
initialization. Both arms use the same seed190921, 12 epochs, batch24,
1,500 updates, AdamW lr0.002 with the original epoch cosine to0.0002,
weight decay0.0001, gradient clip5 and final checkpoint only. No warm start
from any 32-frame checkpoint or new spatial branch.

Control uses original full-known BCE across four near thresholds plus0.2
ordinal loss. Candidate uses0.5 full loss +0.25 small-zone loss +0.25 pure-far
loss, each independently normalized over its known pixels. All terms retain
the same four-head BCE/ordinal definition. Empty batch strata substitute the
full loss. Record effective pixel coefficients before training; these are
loss coefficients, not measured gradients.

Small-zone supervision includes all known pixels of every zone with
0<near2m/known<=20%, regardless of ToF return, including one-pixel supports,
near1.8–2m, and missing returns. Pure-far uses the unchanged evaluator domain:
finite public ToF>=2m and no known near2m pixels. UNKNOWN remains excluded.
All other pixels, including outside ToF coverage, retain full-image supervision.
Labels only form training loss masks; forward sees original RGB/public ToF.

Evaluate original retained NFO, fresh control and candidate once on all original
500 val frames, with cutoff0.081 and sigmoid/cummax. This is consumed synthetic
Development, disjoint from TRAIN by image, scene and family; no original test
observations are accessed. Baseline must reproduce the previous frozen-transfer
counts exactly. No intermediate validation or recalibration.

Candidate acceptance requires far-small recall>=75%, far-small IoU>=same-cohort
original, mixed recall>=94.5% and pure-far FP<=same-cohort original. In addition,
none of these four metrics may regress versus the freshly trained control,
and at least far-small recall or IoU must strictly improve. Report full/outside,
all-small, depth strata, family metrics and paired changes without redefining
the primary denominator. Failed acceptance stops this exact recipe with no
threshold, loss, budget, architecture sweep or automatic successor.

This contrast isolates the specified loss recipe under full TRAIN coverage.
It does not isolate coverage as the cause of previous failures: initialization,
budget and supervision distribution also differ from the prior32 adaptation.
One seed and consumed synthetic data cannot establish hardware, natural-scene,
phone, alert, or safety effectiveness. Even a pass needs separate confirmation.

Implementation: [ba_nfo_fullsupport.py](ba_nfo_fullsupport.py).
Evidence: `artifacts.local/work/ba-nfo-fullsupport-20260919/`.

## Coverage, verification and delivery

All3,000 TRAIN source hashes match.1,509 frames contain small-support zones,
covering3,058,424 known pixels and254,276 near pixels;133,697 near pixels are
in1.8–2m.2,800 frames contain pure-far zones with57,510,641 known pixels.
All39 training families and335 scenes remain represented.1,721,218 unknown
TRAIN pixels remain excluded. No special32-frame selection or depth-gap
filter enters this recipe.

Across the identical1,500 batches, extra strata are never empty. Small-zone
per-pixel coefficients relative to unfocused pixels have median25.282,
range10.484–93.885; pure-far median2.266,range1.931–2.792. The potential
empty-stratum reporting simplification identified by code review is therefore
inactive in this run. These ratios describe coefficients, not gradients.

Two focused tests pass: whole-zone admission with one-pixel/near2m/missing-ToF
and UNKNOWN cases; weighted loss and gradient equality against an explicit
per-pixel formulation, including empty fallback. Independent code review
finds no blocking training/fairness error. The independent verifier reconstructs
all11 evaluation domains from source, recounts all500 packed predictions for
three arms and all six families, and reproduces every frame/aggregate count.
All350,140 Development UNKNOWN pixels are excluded. Original baseline counts
match the earlier frozen-transfer result exactly. Reloading each checkpoint on
fixed first24 and last20 frames reproduces all132 masks bit-for-bit.
Original initialization, all1,500 batch orders, code/input/checkpoint hashes
are verified. The historical checkpoint remains unchanged.

Actual training uses CUDA on the host RTX5060 Laptop GPU; retained equivalent
work CPU/GPU placement evidence applies to this original architecture. Control
and candidate take146.13s and160.89s for their12 training epochs respectively;
these are host training measurements, not phone inference latency. Both fits
finish normally and the process exits. Only final weights, shared initial
weights, sealed protocols/orders, manifests, logs, loss/coverage receipts,
packed predictions and verification/disposition evidence are retained.

Global experiment registration and terminal inheritance remain pending at the
pre-existing ledger303 input-fingerprint error. Actual command receipts and a
local structured disposition are retained; no ledger repair or bypass.
