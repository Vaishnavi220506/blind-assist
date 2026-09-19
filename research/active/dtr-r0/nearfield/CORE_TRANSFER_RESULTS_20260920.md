# Frozen Core Transfer Validation: calibration transfers; RGB guard fails

2026-09-20. COMPLETE. One36-scene/432-frame evaluation, zero training updates.

**The frozen calibrated readout transfers its frame-level FP reduction with no
TP, event or onset loss. The frozen RGB head fails the joint transfer condition.**
RGB removes additional false alerts but loses true frames, one contact event,
delays two contact events and suppresses native corridor contributors. Core
INSIDE event recall alone stays12/12 and would conceal these costs.

This is new controlled-arrangement/appearance evidence under the same simulator,
not natural-scene, new-world, physical-ToF, real-time or safety validation.
The [protocol](CORE_TRANSFER_PROTOCOL_20260920.md) was sealed before capture.

## Four-arm result

All432 frames are admitted:144 strict positives,288 negatives. Twelve INSIDE
clips,12 BOUNDARY clips,12 OUTSIDE clips; BODY/HEAD each216 frames. All24 strict
events enter at nominal1.2s; none are left-censored. Every frame remains scored,
including missing/ineligible RGB opportunities. All methods retain UNKNOWN391,
TN0; silent positive frames count FN.

| Metric | Raw | Calibrated | Calibrated + noRGB | Calibrated + RGB |
| --- | ---: | ---: | ---: | ---: |
| TP / FP / FN | 144 / 181 / 0 | 144 / 146 / 0 | 137 / 138 / 7 | 134 / 135 / 10 |
| Frame recall | 100% | 100% | 95.14% | 93.06% |
| Precision | 44.31% | 49.66% | 49.82% | 49.81% |
| FPR, all288 negatives | 62.85% | 50.69% | 47.92% | 46.88% |
| Core INSIDE events | 12/12 | 12/12 | 12/12 | 12/12 |
| BOUNDARY-contact events | 12/12 | 12/12 | 11/12 | 11/12 |
| All strict events | 24/24 | 24/24 | 23/24 | 23/24 |
| All false-alert segments | 37 | 40 | 39 | 39 |
| All false sampled duration | 36.2s | 29.2s | 27.6s | 27.0s |
| OUTSIDE-layout FP,144 negatives | 106 | 87 | 81 | 79 |
| OUTSIDE-layout false segments | 12 | 13 | 13 | 14 |
| OUTSIDE-layout false duration | 21.2s | 17.4s | 16.2s | 15.8s |

Calibration removes35 total FP and19 OUTSIDE-layout FP while retaining every144
TP and every first in-event alert. However, false segments increase37->40 overall
and12->13 on OUTSIDE layouts: frame/duration reduction is not a demonstrated
reduction in the number of interruptions.

RGB removes11 further total FP and8 OUTSIDE-layout FP versus calibration. Its
increment over noRGB is only3 total FP and2 OUTSIDE-layout FP, at lower recall.
Paired identities show7 FP removed and4 introduced versus noRGB;8 noRGB TP lost
and5 recovered. Their equal23/24 event totals hide different lost events.
OUTSIDE fragmentation also worsens to14 segments. This is not a successful
transfer of the old lossless RGB benefit.

## Event, height and geometry costs

RGB retains all12 Core INSIDE event detections and their first-alert times, but
loses one later INSIDE TP (`f0011`, horizontal HEAD obstacle) and9 contact TP.
Relative to raw/calibration:

- Entire `core_b0_head_horizontal_boundary` event is lost (six positive frames).
- `core_b0_head_protruding_edge_boundary` first alert moves1.2->1.4s (+0.2s).
- `core_b0_body_suspended_solid_boundary` moves1.2->1.6s (+0.4s).

noRGB loses the full `core_b0_body_suspended_solid_boundary` event and one
positive frame in `core_b1_body_large_solid_boundary`. Its detected-event onsets
are otherwise unchanged. Clip-first times, preexisting alerts and each event's
full membership are retained in the result JSON; they are not substituted for
first in-event alert timing. Durations are inclusive sampled frames x0.2s.

| Height,216 frames each | Raw TP/FP/FN | Calibrated | noRGB | RGB |
| --- | --- | --- | --- | --- |
| BODY | 72/93/0 | 72/78/0 | 65/70/7 | 70/74/2 |
| HEAD | 72/88/0 | 72/68/0 | 72/68/0 | 64/61/8 |

| Full3D relation | Frames | Raw TP/FP/FN | Calibrated | noRGB | RGB |
| --- | ---: | --- | --- | --- | --- |
| INSIDE | 72 | 72/0/0 | 72/0/0 | 72/0/0 | 71/0/1 |
| BOUNDARY | 80 | 72/8/0 | 72/6/0 | 65/6/7 | 63/5/9 |
| OUTSIDE | 280 | 0/173/0 | 0/140/0 | 0/132/0 | 0/130/0 |

BOUNDARY includes72 exact-contact positives plus8 near-depth negative samples
within the fixed2cm band. Full3D OUTSIDE also includes pre-entry depth negatives;
the separate144-frame OUTSIDE-layout denominator tests the new lateral physical
arrangements. Neither label uses obstacle centre offset.

All RGB changes in final frame decisions occur on background0 (Brick):
calibrated72/68/0 becomes RGB62/57/10. On background1 (Wood), both are72/78/0.
noRGB gives66/60/6 and71/78/1 respectively. Thus no uniform background transfer
is established. Per-type counts and complete strata are retained in JSON.

## Mechanism and native-support diagnosis

Exactly570 eligible zone samples occur, all pure target-owned under the inherited
native point/bounds ownership proxy:11 INSIDE,158 OUTSIDE,401 CROSSING. These
classifier labels retain their original full-X semantics and are not Core labels.

| Component diagnostic | noRGB | RGB |
| --- | ---: | ---: |
| Eligible pure subset accuracy | 145/570 (25.44%) | 79/570 (13.86%) |
| True OUTSIDE classified/suppressed OUTSIDE | 40/158 | 15/158 |
| Suppressed zone samples | 121 | 72 |
| Suppressed samples containing native corridor support | 35 | 30 |
| Native contributor sample incidences suppressed | 758 | 1,535 |

For RGB, native losses occur on24 frames;23 of those frames still alert through
other votes. For noRGB, all19 affected frames still alert. Aggregate TP/event
retention therefore cannot certify preservation of native support. Contribution
incidences are repeated sampled point/zone/frame counts, not unique surfaces or
physical photon evidence. All22,371 raw valid anchors and their intervals remain
byte-for-value equivalent in the four-arm audit; removing alert votes can still
discard their evidential role. Definite votes and UNKNOWN remain unchanged.

The sharp deterioration from the old in-sample fitting accuracy is diagnostic
evidence of failed attribution transfer for this exact head/recipe. It does not
prove that local RGB has no transferable information or prescribe a new model.

## Validation, disposition and stop

The immutable protocol SHA256 is
`a8fc94b7f0590c9d364a5b4bdbfb10aae4b0d01345c93bb664a73842a56f5592`.
One capture completed; every frame has authenticated live mesh/material/bounds,
camera/target trace, renderer readiness, native source/RGB hashes and admissible
visible support.36 frame6 thumbnails were visually checked; two native-size
frames were also inspected. Source construction reproduces every saved return
and winning-bin contributor list. Predictions were sealed before evaluation.

Twelve focused synthetic tests pass. Pre-freeze independent code review checked
source identity, cube geometry, tensors and metrics. Post-evaluation independent
standard-library recount passes on first attempt without model calls or imports
of the production metric functions: all four counts, subgroup/scene masks,
segments, event identities/times, native suppression, frozen model/code hashes
and success/failure gates agree. No scientific or mechanical retry occurred.

Capture wall time307.797s; the sealed prediction loop6.501s includes readout,
crops, transfers and IO.482 model calls cover241 eligible frames across two
heads,570 zone samples per head; synchronized GPU-call total0.534s. Actual
backend is Torch2.11.0+cu130 on RTX5060 Laptop, float32 frozen heads, CPU OpenCV
and scalar geometry. These single host measurements are not device latency.

Retain calibrated readout as a bounded controlled-transfer component/baseline,
with fragmentation cost stated. Close this exact frozen RGB veto as
`NEGATIVE_CONTROL` for the lossless transfer role; matched noRGB also fails TP
and native-retention guards. Preserve the original96-frame fitting result and
old Thin-object Challenge unchanged. No retraining, threshold/crop/architecture
changes, further capture, Android promotion or automatic successor followed.

Global registration remains blocked by pre-existing ledger303 fingerprint
mismatch; inheritance reports unknown terminal. Actual failed receipts and
local structured dispositions are retained, without claiming global registration.
All owned UE descendants exited and release is confirmed; inference/evaluation
processes exited. No paid allocation or persistent worker remains.

Durable payload: `artifacts.local/work/ba-core-transfer-20260920/`, including
protocol/spec, raw capture, public observations, lineage, sealed four-arm
predictions, per-frame/per-zone results, independent recount script/receipt,
source contact sheet and failure/release receipts. Model weights remain by
verified reference. Only task-created Python bytecode caches are disposable.

Implementation: [scene geometry](core_transfer_spec.py), [capture](core_transfer_capture.py),
[launcher](launch_core_transfer.py), [staged runner](run_core_transfer.py),
[metrics](core_transfer_metrics.py), [geometry tests](test_core_transfer_spec.py),
[metric tests](test_core_transfer_metrics.py).
