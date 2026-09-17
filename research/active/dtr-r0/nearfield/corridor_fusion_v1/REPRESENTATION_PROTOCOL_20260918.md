# Three-arm representation ablation

One EXPLORE experiment: does retaining several RGB associations per public ToF
return improve corridor decisions compared with no association or a single
observable association? This isolates representation under A*'s fixed recipe.

Use the authenticated 1344 A* training frames, strict labels, saved six whole-scene
folds, and identical HistGradientBoostingClassifier parameters/seed185017. Each
arm gets six independent fits and one full-data refit (21 fits total). Choose its
threshold using pooled1152 OOF clear-F1, then fewer FP, then higher threshold,
exactly as A*. No normalization, weighting, architecture or data change.

- Raw: all2217 native sensor geometry columns and seven unassociated RGB/sensor
  summary columns (2224). Exclude all259 explicit hypothesis columns AND both
  ToF-conditioned seed/distinct-box counts. This is a sensor-geometry baseline,
  not raw ADC and not an ablation of IMU or native support geometry.
- Single: same2485 interface and proposal frontend as multi. Each usable,
  nonmerged ToF return chooses at most one intersecting RGB descriptor by maximum
  pixel intersection-over-union, ties by lexicographic box coordinates. No label,
  range, corridor overlap or model score chooses the winner. Pool as before.
- Multi: unchanged2485-feature A* representation, independently retrained.

Use a task-owned copy of the sealed original frontend; verify multi against every
authenticated training cache row and original report extraction, single sensor
parity, and raw column membership. Native slots, missingness, merged and RGB-external
support remain encoded in every arm. No missingness-based veto is added; retained
native-supported true alerts still require outcome accounting.

The prior288-frame single-version confirmation is consumed Development report
data, excluded from fitting and threshold selection. Original MZ136 dev/test stay
excluded. Authenticate public input hashes; seal all public predictions before
joining report labels. Reproduce saved A* OOF scores, selected threshold and all
report scores exactly. A parity failure is an implementation diagnostic, not a
representation result. Mechanical repair may resume the same recipe with logs.

Report clear5cm coverage and precision/recall/F1, full strict and boundary counts,
BODY/HEAD/rod/shallow strata, paired score discrimination, events and onset costs,
native-supported and zero-return strata, lost/rescued A* true frames. Report OOF
selection separately. Also retain full report operating-point curves as explicitly
posthoc descriptive diagnostics: maximum clear TP with FP no greater than frozen
A*, and minimum clear FP while retaining at least A* core AND strict event counts.
These are matched comparisons, not thresholds selected for deployment or unbiased
performance estimates. Matched counts do not guarantee identical event retention;
the primary frozen-threshold event changes remain authoritative.

Evidence for the proposed contribution requires multi to improve the task tradeoff
over both controls without hiding family, native-support or timing costs. If raw
matches or beats it, do not credit explicit associations as the effect source.
If single matches/beats multi, multiple alternatives are not established as the
source. Mixed results retain a scoped Development mechanism hypothesis only.
Keep A* unchanged; no automatic MLP/loss run, threshold rescue, new capture or
App promotion. No statistical independence, natural-distribution or safety claim.

CPU reason GPU_BACKEND_UNAVAILABLE: inherited sklearn HGB and NumPy/OpenCV frontend
have no equivalent GPU backend. Record versions, real timings and backend receipt.
Preserve protocol, source/input hashes, feature caches, models, OOF scores, report
predictions and diagnostic receipts. No persistent resources are required.
