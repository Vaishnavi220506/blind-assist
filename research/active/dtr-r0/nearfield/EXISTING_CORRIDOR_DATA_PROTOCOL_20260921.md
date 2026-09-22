# Existing-data corridor supervision feasibility check

User authorization2026-09-21: check existing data before any new training.
One read-only source audit; no fit, model inference, acquisition, threshold
selection, retained BCE test access or automatic successor. Outputs alone are new.

Sources are the accepted BODY-query5k dataset-v1 and10k final-dataset-v2 indexes.
Inspect all15000 indexed metadata rows and referenced controlled-cube receipts,
retaining original role/site/group/family/condition identities. Check input hashes,
payload existence and camera/receipt alignment. Missing/mismatched evidence stays
explicit; no silent exclusion or new authoritative labels in original datasets.

Question: do these assets contain diverse nearby lateral negatives with reliable
current camera-forward geometry and accessible RGB/native depth? Current query
X=[-.3,.3],Y=[-.2,.9],Z=[.3,3]m. Reconstruct each verified rendered cube as an OBB
using its actor pose, rotation, scale and one-metre cube bounds; exact separating
axis intersection, including contact. Test unions of individual solid components,
not a compound object's bounding-box envelope or its centre alone.

Report separately: target-part intersection, any controlled-component intersection,
target extent outside laterally but within near Y/Z extent, old condition labels,
and current native visible-scene corridor support. Empty/no target is not a useful
nearby lateral negative. A target outside does not imply the whole scene is clear.
Unseen/invalid/background geometry cannot be certified negative by missing depth.
No existing BODY/HEAD bit or LATERAL_OUT name is accepted as the new task label.

All geometry-count distributions are descriptive, not model outcomes. Determine
same-group positive counterparts from geometric target intersection, prioritizing
existing BODY_ONLY then HEAD_ONLY then BOTH; preserve same-site/group pairing.
Report available pairs, sites, regions, families, roles, clearance and near-depth
distributions and repeated original fixture groups.

Bound actual payload checks to at most40 pairs/80 distinct frames: up to5 pairs
per dataset/family, chosen deterministically round-robin over sorted regions and
sites. Prefer existing LATERAL_OUT when equivalent; do not select on model scores
or native success. Include both outside and positive members. Verify RGB/native/
old-label hashes, image size, native dtype, finite/unknown coverage. Original
sampled depth and sensor simulator may reconstruct64 returns for compatibility,
but no classifier is instantiated. Dense native point checks run on available GPU;
scalar metadata/SAT/counting are TASK_NOT_GPU_SUITABLE.

For sampled frames count all visible surfaces in the current corridor and native
points matching controlled target OBBs within original2cm geometry tolerance.
Matching is geometric visible support, not instance segmentation ownership. Count
invalid native rays intersecting the corridor separately. No visible support
cannot prove the absence of occluded surfaces or out-of-FOV obstacle volume.

Decide whether the inspected assets support target-relative geometric supervision,
whether accessible sampled pairs support a cautious visible-scene diagnostic,
and what remains missing for full-scene binary alert training. Diversity is
reported by exact counts, never assumed from total frame count. No number of
sampled passes promotes the remaining payloads to file-by-file verified status.
Stop after the audit, focused geometry tests, result/current/disposition and scoped
delivery. Prioritize reuse only where the actual contract and evidence support it.

Payload root: artifacts.local/work/ba-existing-corridor-data-audit-20260921.
Preserve provenance, per-frame statuses, selected IDs, results and limitations.
Existing global metadata failures are recorded, never manually bypassed.
