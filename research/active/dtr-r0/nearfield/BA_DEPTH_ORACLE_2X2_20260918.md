# BA-Depth 2x2 oracle: count versus placement

**Count correction has much more IoU headroom than placement alone under the
fixed per-zone count. It is not a recall-preserving fix.** Existence-only SCDE
and mono-guided ZPA are closed by user instruction. No training or new model.

## Equal-support comparison

Reuse160 sealed ZJU-L5 frames, existing reference-defined mixed zones and the
same reference-valid/raw-known pixel domain. Compute BOTH predicted count k
and reference count g on exactly that support inside each nonoverlapping zone.
All four arms share this privileged evaluation support. Baseline reconstructs
224725TP/63181FP/10125FN exactly, reproducing75.4032% IoU.
No invalid reference pixel is ranked or counted in any arm. This removes the
earlier full-footprint versus valid-footprint mass confound.

| Count / placement | Precision | Recall | IoU | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DEPTHOR / DEPTHOR | 78.05% | 95.69% | 75.40% | 224725 | 63181 | 10125 |
| GT / DEPTHOR rank | 93.06% | 93.06% | 87.02% | 218544 | 16306 | 16306 |
| DEPTHOR / oracle placement | 78.50% | 96.24% | 76.17% | 226018 | 61888 | 8832 |
| GT / GT | 100% | 100% | 100% | 234850 | 0 | 0 |

GT-count arm sorts DEPTHOR continuous depth, selects exactly g pixels. Oracle
placement selects reference-near pixels first, then reference-far if k>g,
retaining exactly k. Per-zone TP=min(k,g), unavoidable errors=abs(k-g).
Sum abs(k-g)=70,720, matching oracle-placement FP+FN=61,888+8,832.
GT/GT identity is a scoring ceiling, not an independently validated truth.

Count-only improves IoU11.612 points, while placement-only improves0.764.
At fixed k, perfect placement can recover only1,293 TP and remove1,293 FP.
Count correction removes46,875 net FP but loses6,181 net TP. The current
scored predicted near area287,906 exceeds reference234,850 by53,056pixels;
local undercounts still coexist with that global excess.

The factorial IoU interaction is12.220 points: 1-I_mass-I_place+I_current.
Thus24.597-point gap =11.612 +0.764 +12.220, an algebraic decomposition,
NOT three disjoint physical causes. Correcting mass changes which ranking
errors matter; threshold ambiguity overlaps both. Do not claim all depth errors
are mixture-weight errors, or use this oracle as proof a trainable head works.

## Threshold distance bands

Both-axis distance means abs(GT-2) AND abs(prediction-2). The retained JSON
includes nonoverlapping4x4 tables for <=.05, (.05,.10], (.10,.20], >.20m,
plus per-frame tables. Values exactly on decimal boundaries use1e-6m tolerance
for stored float32 roundoff; this does not change the binary2m predictions.

| Error set | Total | Both <=5cm | Both <=10cm | Both <=20cm | Both >20cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| All current FN | 10125 | 78 | 431 | 1948 | 4978 |
| All current FP | 63181 | 137 | 947 | 3156 | 37820 |
| New FN versus raw | 8468 | 67 | 309 | 1034 | 4870 |
| New FP versus raw | 15 | 0 | 0 | 0 | 7 |

Only12.21% of new FN have both values within20cm;57.51% have both farther
than20cm, implying >40cm depth error across the threshold. For all current FP,
only5.00% are both within20cm and59.86% both beyond20cm. A simple small
threshold calibration is not the dominant explanation. Remaining mixed-distance
cases are explicitly present in the joint tables, not forced into either group.
New FP is tiny because raw's conservative expansion already marked most false
near pixels; it must not be mistaken for all current FP.

## Decision, limits and reproducibility

Prioritize count/fraction estimation as a *candidate question* over fixed-mass
rearrangement if further research is authorized. Pair it with a recall constraint
and ranking diagnostics; even perfect counts lose recall under current ranking.
This is not evidence to train a mixture head automatically or to stop every
learned-depth approach. No mixture-network or recent-paper claims were needed.
Sensor return ownership, BODY/HEAD risk, natural transfer and precise thin-object
recovery remain unevaluated. RealSense reference is imperfect; mixed-zone P20
selection misses some very small foreground objects.

All160 input and prediction hashes verified. Two focused tests cover exact count,
optimal-placement overlap/count-error identity, and decimal band endpoints.
Initial diagnostic retained in ba-depth-oracle-2x2-20260918; v2 fixes only
float32 threshold-band endpoint handling. Four-arm results are unchanged.
Final evidence: artifacts.local/work/ba-depth-oracle-2x2-20260918-v2, including
protocol, summary, scene/frame tables, zone count audit and threshold matrices.
CPU saved-array work only; no new inference, downloads, training, background job
or paid resource. Existing ledger303 registration mismatch is not bypassed.
