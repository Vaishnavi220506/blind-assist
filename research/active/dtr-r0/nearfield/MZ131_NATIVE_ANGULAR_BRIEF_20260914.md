# MZ131-A: ToF-native depth-coherent center-span hypothesis

EXPLORE; one fixed paired replay on the already consumed MZ123 288 frames.
Baseline is the retained MZ129 Radar arm, 139 TP / 93 FP / 5 FN, 15 events.
No new capture, training, threshold sweep, Radar change or MZ130 repair.

Question: can adjacent, depth-compatible independent ToF zones constrain an
echo's angular extent enough to affect the actual remaining alert supports?
The earlier depth graph proves useful directional grouping on its own artificial
panel; it does not establish within-zone localization. MZ130 is a negative
control: RGB coverage must never define the complete ToF/Radar spatial coverage.

## Fixed observable recipe, before scoring

- Use only raw ToF zone IDs, angular bounds and returned status/range. Each
  eligible zone has exactly one target, SIM_VALID, finite positive range and
  finite nonnegative sigma. Multi-target, MERGED, malformed and missing zones
  supply no narrowing. They retain their MZ129 support.
- Four-neighbor links use the prior strict absolute slant-depth gap <0.30 m.
  Require at least two independent zones. Reject an entire connected cohort
  when its global max-minus-min range is >=0.30 m, avoiding transitive drift.
- For each angular axis, let c be the equal-zone mean of angular zone centers.
  If at least two distinct center coordinates participate, use
  [c - max_i(abs(center_i-c)), c + max_i(abs(center_i-c))]. Otherwise preserve
  the full union of native zone bounds on that axis. No fixed single-zone
  center, signal weight, RGB box/mask, image clip or evaluator input defines it.
- This **center-span envelope is an uncalibrated localization hypothesis**.
  It is not the union of full physical zone footprints, nor a geometric proof
  that their outer halves contain no echo. The full-bound union would include
  every original zone and cannot narrow it. Native edge retention is therefore
  a falsifier of this extra assumption, not an optional aggregate metric.
- Intersect the native envelope with each existing MZ129 return ROI solely to
  preserve baseline restrictions. Empty intersections fall back unchanged.
  Existing RGB association remains inherited; no RGB input selects the cohort,
  centroid or angular width. Keep full original range, pose uncertainty and
  enclosing slant geometry. No interval-subdivision change. Preserve MERGED,
  certain-coarse override, MZ129 Radar/raw/carry/guards, weights and threshold.
- Retain one contribution per zone. Non-alert remains UNKNOWN.

Save both arms, every return/cohort and source/code hashes before scoring.
Report TP/FP/FN, all original TP IDs, event timing, false segments, and separate
BODY/HEAD/rod/boundary counts. For the baseline 93 FP, distinguish (1) any
narrowed return, (2) narrowed positive-weight possible return, (3) such return
losing its possible bit, (4) zone score changed, and (5) alert removed.
Audit every narrowed return's native contributors, identifying newly excluded
samples and newly excluded corridor-point contributors separately from actors
that intersect the corridor. Missing native lineage is NOT_EVALUABLE.

Retain only with FP below 93, all 139 original TP, all 15 events without later
alerts, and zero newly lost previously-contained native corridor contributors.
Interpret effect size and support reach together; touching only irrelevant
supports is no gain. Failure ends this formula: no post-score fallback tuning,
new arm or automatic successor. A failure is scoped to this envelope, not all
ToF-native angular inference. Outputs stay in artifacts.local/work/mz131-native-
angular-20260914. CPU scalar geometry: TASK_NOT_GPU_SUITABLE. No persistent
allocation. Registration uses supported knowledge commands; preserve ledger
fingerprint failures as pending metadata without rewriting the ledger.
