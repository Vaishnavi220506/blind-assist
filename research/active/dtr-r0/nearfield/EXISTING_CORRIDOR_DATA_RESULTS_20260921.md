# Existing data are readable, but do not supply the needed lateral hard negatives

2026-09-21. The bounded audit is complete. **The15000 BODY-query frames support
input reuse and controlled-object geometry adaptation, but not an immediate
training fix for the recent near-boundary false positives.** Preserve the adapter
and candidate inventory as COMPONENT evidence; do not start a fit or more capture.

[Protocol](EXISTING_CORRIDOR_DATA_PROTOCOL_20260921.md),
[audit code](audit_existing_corridor_data.py),
[geometry adapter](reuse_corridor_geometry.py),
[tests](test_reuse_corridor_geometry.py),
[disposition](EXISTING_CORRIDOR_DATA_DISPOSITION_20260921.json).

## What was actually verified

Both accepted indexes and summaries match their manifests. All15000 metadata
rows were joined to frozen source specifications and actual rendered-cube receipts;
all geometry was supported. RGB/native/label paths exist for every indexed row.
This is not a full hash/decode audit of45000 payload files. Actual payload checks
were restricted to10 selected pairs/20 frames, chosen before native inspection:
all60 RGB/native/old-label hashes matched, RGB verified640x360, native depth float32
360x640. Four actual RGB files in two selected pairs were visually inspected.

Native depth is axial camera-forward distance in metres. Images already use the
same640x360,100-degree horizontal-FOV contract as current sampling; camera yaw
varies and must enter the world-to-camera geometry transform. Current64-zone
single-return reconstruction succeeded on the20 samples (15-60 valid zones).
No classifier, checkpoint, optimizer, training or new acquisition was used.

## Counts expose the gap hidden by total dataset size

| Readout | BODY-query5k | BODY-query10k |
| --- | ---: | ---: |
| Indexed/geometry-verified frames |5000|10000|
| Target intersects current corridor |3000|6000|
| Nearby laterally excluded target; all controlled components outside |250|500|
| Candidate families |crossbar only|crossbar only|
| Candidate sites / regions |250/10|500/10|
| Candidate original fixture groups |12|12|
| Same-group positive counterpart available |250|500|
| Candidate TRAIN / DEV / EVAL roles |125/50/75|250/100/150|

The750 candidate site names are distinct across the two indexes, but both reuse
the SAME12 original crossbar fixture groups. They are not750 distinct object
geometries. All candidates carry the old LATERAL_OUT condition, but eligibility
was recomputed from actual geometry, not accepted from that condition name.
The old groups' positive members can also change height/dimensions; a same-group
pair is not automatically a pure lateral intervention.

Candidates' nearest lateral edges are1.6359-1.7936m outside the current corridor.
The17 OUTSIDE false-positive frames from the preceding frozen transfer have
only0.06598-0.11911m clearance, from three geometries. These are very different
spatial distinctions. More background variation at a1.7m lateral gap cannot be
assumed to teach the missing6.6-11.9cm boundary distinction across target types.

The audit's `near` candidate field requires separate Y and Z envelope overlap,
not a general proof of joint Y/Z finite-slab intersection. Lateral exclusion is
strict, and final full-corridor intersection uses exact15-axis OBB SAT. This
candidate filter is an inventory aid, not a new ground-truth class.

## Small payload check: only three visibly supported pairs

Only two dataset/family strata contained candidates, so the prespecified maximum
of40 pairs became10 pairs (five regions per dataset), not a request to manufacture
missing family coverage. Eight selected pairs have TRAIN roles and two DEV;
no EVAL pair was sampled and no protected BCE test data was opened.

All10 positive endpoints have current-corridor target-matching native support.
Only3 negative endpoints have at least3 native points matching the target OBB;
all three are BODY-query10k. The other7 have zero such target matches, although
supporting fixture parts may remain in view. This is not proof that the entire
object/fixture is invisible, nor a segmentation/actor-ownership certificate.
Four RGB spot checks agree with the limited interpretation: side fixtures can
appear at the edge while the designated cross-member is absent or only partial.

All10 negative endpoints have zero visible native points in the current corridor.
Eight nevertheless contain invalid rays whose paths intersect the query volume;
even the other two cannot exclude hidden/out-of-FOV scene geometry. **Zero full-
scene negatives were certified.** Do not convert this absence of visible support
into a binary no-alert label or use it as an FPR denominator.

## Why old labels cannot simply be renamed

Old BODY/HEAD labels are visible-surface queries with different height/lateral/
forward limits. With the old1.7m camera height, the current camera-volume maps to
0.8-1.9m above the declared floor; the old BODY and HEAD boxes use0.65-1.4m and
1.4-1.85m and different widths/ranges. Their query ownership arrays represent
geometric region membership, not target-instance identities.

Actual controlled objects are authenticated centred Engine Cube meshes, including
rotated oblique rods. Their receipt pose/scale reconstructs precise solid OBBs;
intersect each component, not the compound bounding-box envelope. This can prove
controlled-object positives and controlled-fixture exclusion. It cannot certify
absence of background obstacles. The old background inventory was not a complete
per-site scene-volume certificate. UNKNOWN must remain explicit.

A separate existing distance5k summary was also checked read-only:5000 frames,
2500 accepted pairs,500 sites, all intended endpoints HEAD-positive. It is useful
for distance-condition diagnostics, but supplies no HEAD negative denominator.
It was not added to this15000-row audit or its payload sample.

## Decision and verification

Retain accessible RGB/depth/background assets and the tested geometry adapter.
Do not feed all15000 frames into the current binary classifier as if they supplied
reliable, diverse near-boundary negatives. This audit rules out that shortcut;
it does not rule out using these assets for background/representation work or
separately scoped target-relative supervision after further contract work.
The particular coverage gap is visible, near-boundary lateral exclusion across
multiple target types; the full-scene negative-label gap is separate.

Seven synthetic tests pass: Unreal rotation parity, camera invariance, closed
contact/full extent, oblique AABB false positives, a separating cross-axis case,
compound empty gaps and invalid mesh/transform rejection. Independent saved-record
recount confirms counts, role/pair bookkeeping and output/code seals. Geometry
and metadata use CPU TASK_NOT_GPU_SUITABLE;20 dense native checks used CUDA.
Original data were unchanged. No existing scene/model/timing protocol was tuned.

Evidence is retained under
`artifacts.local/work/ba-existing-corridor-data-audit-20260921/`, including all
geometry records, selected IDs before payload checks,60 payload hash comparisons,
native summaries, visual notes, previous FP clearance comparison and provenance.
All task processes finished. Supported global metadata failures are recorded in
the local disposition; no ledger edits or manual bypass occurred.
Stop this audit without training, new capture, original-test access or successor.
