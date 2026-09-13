# MZ130: positive local support before spatial extent

EXPLORE, same 288 consumed MZ123 controlled frames. User authorized continuing
the range-extent diagnosis. Baseline is the complete frozen MZ129 Radar arm
(139 TP / 93 FP / 5 FN), including MZ125 full-context allocation, MZ128 binary
alert-use, raw Radar, complete original flow carry and resolution guards.

Hypothesis: positive image contours and spatially/range-compatible echoes can
constrain a local surface hypothesis more faithfully than an entire rectangle.
This is a correspondence hypothesis, not semantic identity or a completeness
guarantee. No fixed box shrink, zone center substitution, peripheral deletion,
merged-depth inference, missing-modality veto, training or parameter scan.

Frozen observable construction:

1. Recompute the original MZ125 pixel residual frontend from authenticated RGB.
   Require recovered boxes to reproduce. Match each original connected component
   exactly to its proposal. Dilate that component by the inherited two-pixel
   pad, clip to the padded box/image, and encode its exact union of pixel cells
   as nonoverlapping rectangles. No mask/ambiguous component means fallback.
2. ToF retains individual returns and the original .25 m complete-link valid
   depth cohorts. Replace the box-area signal template by mask-area overlap per
   zone, retaining the existing coverage .8, cosine .9, margin .1 and minimum
   two distinct positive zones. An already-associated return must keep the same
   proposal; conflicting correspondence falls back. A previously unresolved
   valid return may acquire this explicitly reported new positive association.
   Intersect accepted component cells with the original ROI/zone; an empty
   result falls back. For each tile retain the entire old range/pitch/yaw bounds
   and enclosing slant geometry. A return is possible if any tile intersects.
   No interval subdivision, MERGED narrowing or change to certain/raw overrides.
3. Radar uses the unchanged MZ129 current associations and filtered distances.
   It may use local support only if at least two distinct valid ToF zones have
   accepted the same image component and their slant-derived horizontal-range
   intervals overlap the current raw Radar range ±.15 m. Use the union of their
   local tiles within the inherited image-angle compatibility window ±12 degrees.
   This is a working proxy, not a calibrated hardware beam/CI. Empty/missing/
   ambiguous evidence falls back to the original current Radar support. Preserve
   the original frontoparallel plane anchor from the whole proposal; project
   tiles on that same plane so subdividing does not recenter or change distance.
   Raw Radar, complete original carry, old/new guards remain independent.

Four arms: frozen MZ129; ToF only; Radar only; both. Radar's ToF corroboration
uses local correspondence in both relevant arms, without changing the independent
ToF alert bit in the Radar-only arm. Thus its mechanism includes cross-sensor
correspondence, not just Radar postprocessing. Keep all absent/unresolved outputs
UNKNOWN; retained records do not imply unchanged warning influence.

Seal observable outputs before labels/native geometry enter scoring. Report
every changed frame, branch changes/fallback reasons, TP/FP/FN, precision,
families, individual event timing, false segments/bin duration and actual host
cost. Separately audit projected native hit retention within changed ToF unions
and within changed Radar hypotheses; distinguish native corridor points from
points on actors that intersect the corridor. Native data never selects tiles.

Retain a Development candidate only with fewer FP, all 139 original true frames,
all 15 original event times, and zero newly excluded native corridor-hit samples
previously retained by the affected hypothesis. Report other dropped contributors
and unresolved continuous-surface coverage even if these checks pass. Missing
native correspondence is NOT_EVALUABLE, not a zero-drop success.

MZ129's interval-refinement null and protected Radar gain remain intact; MZ124's
broader raw-null policy is a different comparator. One fixed replay plus its
native attribution ends this task; no tuning on failure or automatic new capture.
Any gain here remains consumed Development, requiring a separate frozen fresh
scene check before generalization claims. Artifacts stay under
artifacts.local/work/mz130-local-support-20260914. Scalar geometry uses CPU
(TASK_NOT_GPU_SUITABLE); the inherited pixel frontend uses its existing CPU
implementation (GPU_BACKEND_UNAVAILABLE). No persistent allocation is required.
