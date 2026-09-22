# One public RGB region / multi-zone anchoring feasibility check

2026-09-21. User authorized one distinct spatial-ownership attempt. EXPLORE on
consumed diagnostic cases only; no training, threshold sweep, alert modification,
conditional hold, new data or protected-test access. Stop after this check.

Prior Otsu minority-region and tiny RGB classifier negatives remain closed.
Different hypothesis: actual observations in multiple adjacent ToF zones may
anchor one connected RGB region and constrain ownership in a queried zone.
This is an appearance/association hypothesis, not verified surface identity.

Fixed diagnostic cohort: all seven original-high added lateral negatives with
contained-depth evidence in the completed axis audit, plus their seven matched
BOUNDARY positives at identical group/time. IDs are selection0702/0711 paired
0678/0687; evaluation0486/0487/0494/1063/1070 paired0462/0463/0470/1039/1046.
These fourteen frames cover three layouts; they are selected consumed failure
examples, not an independent validation cohort or a general performance estimate.

One implementation: grayscale native RGB in the complete actual nominal-ToF
image footprint; one Otsu threshold; eight-connected components from BOTH
polarities, no minority/target/nearest-component selection, learned weights,
contour tuning or morphology. Regions touching the footprint edge are clipped
and cannot anchor. No ground-truth ROI or rendered geometry enters extraction.

A region's anchor is a valid observed zone whose entire native-pixel footprint
belongs to that component. Require at least two four-neighbor anchors; all such
anchors must share a nonempty intersection of their original range intervals.
Intersection only tests consistency. For geometry always use their full interval
envelope, never the intersection. No copy of range into an unobserved zone.

Queried zones have original full intervals contained in [.3,3], possible corridor
intersection and ambiguous full horizontal support. This limits diagnosis to the
remaining mid-event uncertainty; it is NOT a new universal distance requirement.
A region is compatible with a query if it occupies >=4 original sampled-lattice
points in that zone (the existing sensor minimum sample count), and the query
interval intersects the common anchor overlap. Preserve all compatible regions.
Exactly one compatible region gives a tentative attribution; zero or multiple
gives UNKNOWN. Classify the complete region horizontal extent, expanded by one
native pixel, against [-.3,.3] using the envelope of ALL anchor and query
intervals. A proposed OUTSIDE relation remains tentative, never CLEAR or an alert
veto. Public processing never reads frame labels, target bounds or native traces.

Seal component metadata, queried-zone assignments and diagnostic masks before
joining evaluator information. Report anchor availability, candidates, UNKNOWN,
paired discrimination and native ownership where an assignment exists. An
assignment is not correct merely because its OUTSIDE label matches frame truth:
check actual selected-return contributor pixels belong to the proposed region,
using private lineage only after sealing. No assignment means ownership is
unresolved and must not be counted as correct. Do not claim missing RGB content
or absent full64 information from a failed strict anchoring recipe.

Useful local feasibility requires an explained OUTSIDE distinction for at least
one negative in EACH role, without an OUTSIDE assignment to a paired Boundary
positive or loss of observed contributor coverage in assigned zones. Otherwise
close this exact recipe, report the failure stage and do not loosen anchoring,
retune segmentation, train a head or automatically start a successor.

CPU OpenCV connected-component and small geometric reductions are
TASK_NOT_GPU_SUITABLE; no model or paid allocation. Output
`artifacts.local/work/ba-rgb-multizone-20260921/`. Preserve all original inputs,
scores, predictions, intervals, UNKNOWN and demo. Recount independently, save
registration/inheritance receipts and complete scoped normal Git delivery.
