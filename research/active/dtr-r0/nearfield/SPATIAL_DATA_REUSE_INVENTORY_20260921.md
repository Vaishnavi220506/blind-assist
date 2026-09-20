# Existing data relevant to the camera-forward RGB-ToF task

2026-09-21. Bounded inventory prompted by the user's reminder during the frozen
supplement transfer. Counts below were checked against existing specifications,
prepared manifests and admission summaries, not a new file-by-file payload audit.
No models were run, protected labels opened, or successor experiments authorized.

| Existing source | Verified size | Reuse and boundary |
| --- | --- | --- |
| Core transfer |432 frames/36 clips|Matching camera task and single return;12-frame approach clips, consumed Development. Regression/diagnosis.|
| Core workpoint transfer |432/36|Same task;12-frame approach clips, consumed Development.|
| Core hold validation |432/36|Same task;12-frame approach clips, consumed Development.|
| Full event transfer |576/24|Matching task;24-frame approach/dwell/retreat, consumed. Complete-event diagnosis.|
| Spatial BCE |2880/40 base groups|Direct input/label compatibility;1728 train and576 dev consumed;576 test/8 groups still unactivated and excluded from new training.|
| BODY-query5k |5000/1000 groups/250 sites|Native RGB/depth/camera and ownership retained; adapt current64-zone input and full-volume labels. Static conditions, not event sequences.|
| BODY-query10k |10000/2000 groups/500 sites|Same potential; original5000/2000/3000 roles already used Development. Known background/visibility limitations retained.|
| Hypersim NFO subset |10000;8000/1000/1000|Native RGB/depth and near-region labels. Existing central80% proxy zones differ from current45-degree geometry; full obstacle/event labels absent.|
| SANPO-Synthetic NFO subset |3000/60 sessions|Native RGB/depth; up to50 sampled frames/session, not full event sequences. Existing256x192 inputs need original-source alignment to current16:9 recipe.|
| ZJU-L5 consumed subset |160/8 scenes|Real RGB+8x8 ToF, different projection/calibration contract. Near-surface sanity only; current corridor calibration and complete event labels not established.|

The five closely matched sources total4752 frames, including the576 unactivated
test frames that remain reserved. The15000 BODY-query frames are a substantial
candidate for later representation/background training and failure diagnosis
without automatically collecting another source. Their old BODY/HEAD labels
cannot simply be renamed to the current camera-forward full-extent contract.
An explicit geometry/input adapter and source admission check would be required;
whether every necessary original payload remains readable was not rechecked.

NFO's additional500 UE frames derive from the5k BODY-query source and are NOT500
additional independent frames. A large image count does not establish event
diversity or a clean held-out verification set. Existing consumed data remain
useful for training/diagnosis but cannot be relabelled as fresh validation.

## Inspected evidence

All payload paths below are relative to the canonical ignored artifact root:

- `work/ba-core-transfer-20260920/spec.json`
- `work/ba-core-workpoint-transfer-20260920/spec.json`
- `work/ba-core-hold-validation-20260920/spec.json`
- `work/ba-full-event-transfer-20260920/spec.json`
- `work/ba-spatial-bce-20260920/spec.json`
- `work/body-query-5000-20260909/dataset-v1/summary.json`
- `work/body-query-10000-20260909/final-dataset-v2/summary.json`
- `work/ba-nfo-20260919/prepared-manifest.json`
- `work/ba-depth-probe-20260918/protocol.json`

Contract/context files inspected:
[BCE results](SPATIAL_BCE_RESULTS_20260920.md),
[5k results](BODY_QUERY_5000_RESULTS_20260909.md),
[real-depth probe limitations](BA_DEPTH_PROBE_20260918.md).
Two additional local working files were inspected for context only:
`BODY_QUERY_10000_COLLECTION_20260909.md` and `ba_nfo_data.py`. They are pre-existing
untracked work and are not promoted to route authority by this inventory; their
payload counts above come from the retained summaries/manifests.

This inventory supports prioritizing existing-data reuse before further capture;
it does not start an adapter, new fit, label revision or new experiment. The
already running frozen16-group supplementary-policy verification remains bounded.
