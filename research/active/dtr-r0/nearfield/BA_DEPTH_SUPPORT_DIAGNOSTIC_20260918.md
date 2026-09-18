# BA-Depth support-loss diagnosis

**Do not start the proposed existence-only SCDE training from this diagnostic.**
Every near-return zone containing a newly missed pixel still has predicted near
surface. Most already satisfy the proposed quantile constraint. This challenges
the specific motivating mechanism, not learned RGB-ToF geometry generally.

## Method and scope

Reuse all 160 sealed ZJU-L5 probe inputs and predictions, checking their hashes.
No inference, new data, training, new thresholds for predictions or alert change.
This is posthoc consumed Development diagnosis. The protocol fixes 2m near masks,
the existing mixed-common region, range tolerances .05/.1/.2m, boundary bands
5/10px, and a separate +/-10cm threshold-crossing flag before diagnostic scoring.
CPU is used for small saved-array operations (TASK_NOT_GPU_SUITABLE).

New FN = reference<2, raw<2, DEPTHOR>=2. Rescued FN = reference<2, raw>=2,
DEPTHOR<2. Shared FN = reference<2, raw>=2, DEPTHOR>=2. They are disjoint.
Each valid near-return projected rectangle covering a new FN is inspected in
full, using its visible footprint. Overlapping rectangles use existential union,
not repeated pixel counts. Reference is never used to relocate the sensor return.

## Accounting correction

| Quantity | Pixels |
| --- | ---: |
| New FN | 8,468 |
| Rescued FN | 4,627 |
| Shared FN | 1,657 |
| Raw FN = rescued + shared | 6,284 |
| DEPTHOR FN = new + shared | 10,125 |
| Net increase = new - rescued | 3,841 |

The previous report's 3,841 is a **net** increase, not the newly lost set.
The new set spans 25 frames and 52 valid near-return zones. Class B ('raw did
not provide near support') is impossible within this set by construction.
Raw-positive does not prove that its echo originated at that particular pixel.

## Existence versus localization

| Diagnostic | Result |
| --- | ---: |
| Affected zones with no predicted surface <2m | 0/52 |
| Affected zones already satisfying Q10<=d+0.1m | 48/52 |
| New FN pixels covered by an already-satisfied zone | 8,097/8,468 (95.61%) |
| New FN pixels with no covering-zone prediction within +/-0.1m of its return | 0/8,468 |
| Same, +/-0.05m diagnostic | 658/8,468 (7.77%) |
| Same, +/-0.2m diagnostic | 0/8,468 |

These are empirical compatibility diagnostics, not physical sensor uncertainty
certificates. 'At least one compatible pixel' does not mean a useful patch or
correct attribution; Q10 additionally requires roughly 10% area, which the
sensor did not measure and which can expand genuinely thin surfaces. Moreover,
Q10<=d+epsilon is one-sided: arbitrarily too-near predictions can satisfy it.
Thus existence conservation and preservation of the true obstacle surface are
different requirements. The proposed constraint is already satisfied around
95.61% of newly missed pixels and gives no violation signal there.

## Boundary and threshold diagnostics

New FN within 5px of a valid-reference >.5m discontinuity: 2,699 (31.87%).
Within 10px: 3,753 (44.32%). Both GT>=1.9m and prediction<=2.1m:
309 (3.65%). GT alone within .1m of threshold:777 (9.18%);
prediction alone within .1m:1,753 (20.70%).
These flags overlap and are not exclusive causal categories. They do not prove
reference error or boundary-head effectiveness. Fewer than half lie in the
10px band, so neither 'all boundary noise' nor 'all threshold ambiguity' fits.
No image or echo identity labels establish precisely which true surface produced
each return. The remaining positioning error is not evidence of missing ToF
existence. RealSense reference and mixed-zone thin-object limits still apply.

## Decision and delivery

The user's conditional training trigger (a clear predominance of near-support
erasure) is not established: no affected zone loses all predicted near surface,
and none of the new FN lacks compatible covering-zone support at +/-0.1m.
No Near-aware/SCDE training runs, no Hypersim download, no loss-weight search.
Preserve the earlier 12.22-point IoU gain and its recall cost; no alert promotion.
Different mono/fusion backbones still prevent attributing their entire difference
causally to ToF. No newer-paper claims were needed or verified in this diagnosis.

Two focused tests check gross/net FN accounting and a quantile-satisfied but
mislocalized surface counterexample (including one-sided too-near acceptance).
Counts reconcile exactly with the frozen probe. Retain protocol, summary,
per-frame and zone records under
`artifacts.local/work/ba-depth-support-diagnostic-20260918/`.
No owned background process or paid allocation remains.

This adds a diagnostic conclusion to the existing probe, not another trained
challenger. Structured experiment registration remains affected by the previously
verified ledger303 fingerprint error; no manual ledger changes. Prior MZ140/142
recipe-specific negatives remain intact.
