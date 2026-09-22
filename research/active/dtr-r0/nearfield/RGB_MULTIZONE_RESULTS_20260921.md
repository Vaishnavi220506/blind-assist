# Multi-zone RGB anchoring: local positive coverage, no residual-FP distinction

2026-09-21. One frozen feasibility check completed on fourteen consumed
diagnostic frames. **No negative frame receives an explained OUTSIDE result.**
Only two matched Boundary-positive frames have usable multi-zone anchors; their
three queried zones retain every observed contributor and remain CROSSING.
Close this exact anchoring recipe as a NEGATIVE_CONTROL for resolving the known
lateral mistakes. Keep A/demo unchanged; no segmentation retry, weaker anchors,
training, new alert gate, conditional hold or automatic successor follows.

## Distinct hypothesis and scope

[Protocol](RGB_MULTIZONE_PROTOCOL_20260921.md),
[runner](rgb_multizone_probe.py), [focused tests](test_rgb_multizone_probe.py).
The previous [minority Otsu / local classifier](TOF_LATERAL_ATTRIBUTION_20260920.md)
did not enforce agreement with actual neighboring ToF observations. Its
[transfer failure](CORE_TRANSFER_RESULTS_20260920.md) remains intact.

This check keeps both Otsu polarities and every connected component over the
complete public ToF footprint. A region must contain at least two adjacent
whole-zone footprints with mutually overlapping original measured intervals.
The overlap tests consistency; full interval envelopes are retained for geometry.
One compatible region yields tentative query attribution; no or multiple regions
yield UNKNOWN. Complete horizontal region extent plus one pixel is classified,
never the object center. Components clipped by the public footprint cannot anchor.

The fourteen fixed frames comprise all seven remaining original-high lateral
negatives with contained-depth evidence, plus their same-layout/time Boundary
positives: two selection pairs and five evaluation pairs. Only three layouts
are represented. This is selected consumed failure diagnosis, not fresh transfer,
general accuracy, RGB information sufficiency or an alert-policy evaluation.
RGB source hashes and public outputs are sealed before evaluator joins. No
target ROI, masks, native depth, target identity or labels enter extraction.

## What happened

| Pair: negative / positive | Role | Maximum full-zone anchors: negative / positive | Query zones: negative / positive | Assigned query zones: negative / positive |
| --- | --- | --- | --- | --- |
| f0702 / f0678 | Selection | 0 / 2 | 2 / 2 | 0 / 2 |
| f0711 / f0687 | Selection | 0 / 2 | 1 / 1 | 0 / 1 |
| f0486 / f0462 | Evaluation | 0 / 0 | 2 / 2 | 0 / 0 |
| f0487 / f0463 | Evaluation | 0 / 0 | 1 / 1 | 0 / 0 |
| f0494 / f0470 | Evaluation | 0 / 0 | 2 / 2 | 0 / 0 |
| f1063 / f1039 | Evaluation | 1 / 1 | 0 / 0 | 0 / 0 |
| f1070 / f1046 | Evaluation | 1 / 1 | 0 / 0 | 0 / 0 |

There are16 queried zones in10 frames:3 assigned and13 UNKNOWN. Four frames
have no queried zone under the frozen contained-depth/possible/CROSSING scope;
they are unresolved, not successfully classified. Five queried negative frames
have eight UNKNOWN zones; the other two negatives have no qualifying query.
No OUTSIDE assignment occurs, so neither role meets the fixed distinction test.
Zero wrong-positive OUTSIDE assignments with zero negative distinctions is not
a successful specificity improvement.

The two attributable positives are body_protruding_plane_g01 f0678/f0687.
The assigned query zones are34/50 and42. After sealing, private winning-return
lineage maps55,87 and165 sampled contributor points into the proposed component:
307/307 incidences retained. These are observed-return contributor pixels, not
307 unique obstacles or proof of target identity in arbitrary scenes. Source
native corridor counts for the two frames are27 each. No alert vote was removed.

## Failure stage and visible evidence

The failure occurs at the strict multi-zone anchoring stage. Unclipped regions
in eight of fourteen frames contain no complete valid zone; four contain one
and two contain two. Only the two positive frames above reach the required adjacent pair. On
the HEAD hanging examples both positive and negative targets are visible in RGB,
but this segmentation/whole-zone rule supplies no anchor. Thus it is incorrect
to diagnose absence of an RGB cue or missing objects from the UNKNOWN result.

The seven-pair contact sheet was rendered and visually inspected, with query
zones orange, anchors blue, and anchored region extent cyan. It is retained at
`artifacts.local/work/ba-rgb-multizone-20260921/matched-evidence.png`.
The full-zone requirement is a deliberately conservative association assumption,
not a necessary condition for all possible RGB/ToF methods. Its observed lack
of coverage closes this recipe; it does not establish an8x8 information limit.

The two original-high negative frames with no compatible contained query were
already geometrically out of this probe's scope. Their zero query count is not
a new RGB benefit. Likewise, the positive f1046 has21 native corridor samples
yet no qualifying query, demonstrating why eligibility must not be equated with
absence of physical positive evidence.

## Verification and delivery

Three focused checks pass: no diagonal/row-wrap adjacency, full-interval lateral
classification, and clipped-region abstention. The [independent saved-output
audit](verify_rgb_multizone.py) passes: 72 hash entries, seven seals, all fourteen
reconstructed masks, anchors, queries, pairings and contributor counts agree.
The audit uses the prescribed OpenCV Otsu/labeling primitives and independently
recounts the geometry. OpenCV and small geometric reductions run on CPU under
TASK_NOT_GPU_SUITABLE, with no model load, training or paid allocation.

Evidence under `artifacts.local/work/ba-rgb-multizone-20260921/` includes all14
public region-label maps, component/anchor/candidate records, source hashes,
frozen protocol, results, native coverage and the contact sheet. Original
scores, intervals, alerts, UNKNOWN and protected-test status remain unchanged.
Local disposition is NEGATIVE_CONTROL for this exact multi-zone RGB anchoring
recipe's residual lateral-error role. Supported global registration failed at
the pre-existing `experiments/index.jsonl:303` input-fingerprint mismatch;
inheritance returned `unknown terminal id: ba-rgb-multizone-20260921`.
Both receipts and the structured local disposition are retained; global metadata
remains pending without a manual ledger bypass. Stop after scoped delivery.
