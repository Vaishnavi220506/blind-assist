# Return-Lineage / Surface-Occupancy Audit

2026-09-20. Pure consumed-data diagnosis; no new algorithm or observation arm.
Retain Calibration and its threshold/output; keep RGB/noRGB and aggregation
closures. Do not run strongest/closest/top2 or recover negative gap alerts.

Audit all Core432 and old Thin96 nominal45-degree stored vectors, reporting
cohorts separately. Include all original144/36 positive frames, including absent
sampled support and existing FN. Inspect the outgoing pre-gap winner across each
of four full gap windows:13 records, with five gap records. A two-frame gap is
one interruption; temporal transitions use immediate previous frame, not a flank.

Use frozen native optical-Z, calibrated camera projection, authenticated target
and backdrop bounds/actor metadata, stored quantized boxes, original seeds and
winning-bin lineage. Replay the original single-return constructor exactly only
to verify values/traces. No alternative return is emitted. Reconstruct all
eligible10cm bins before selection, sample counts and sum(1/max(Z,.3)^2), plus
target/backdrop/other contributor counts. These weights are a simulator proxy,
not measured photon signal, reflectance, waveform peaks or VL53 confidence.

Target identity uses the prior +/-2cm native-depth/bounds ownership proxy;
actor/source IDs authenticate the bounds but are not a per-pixel segmentation
buffer. Preserve unknown/other ownership. Compute occupancy at BOTH native pixel
centres in each full quantized footprint AND the actual point-sampled sensor
lattice. Report counts/fractions, valid/invalid, foreground/background, strict
range<=3m and target-inside-corridor support. Never treat unobserved as clear.
Also project the target AABB into the image: an empty intersection certifies no
projected bounding support; a nonempty rectangle alone does not prove visibility.

Foreground means the scene target, which can exceed3m in negative pre-entry
frames. Separately report strict near/corridor samples; do not mislabel a3m+
foreground/background switch as disappearance of a <=3m obstacle.

For an observed pure non-target winning bin with greater depth than every
eligible target sample, and with no sampled native corridor contributor in the
winning bin, classify retained target samples as B_RETURN_WINNER_FLIP_CAPABLE.
Report >=1 target sample and nested >=4 views. The original4-hit minimum applies
to TOTAL zone hits, not target hits or each bin. Record target-bin rank, maximum
target-bin weight, winning-bin weight, distance separation and sensor reason.
If the target's projected AABB, native footprint and sampled footprint all have
zero overlap while background is observed, classify A_GEOMETRY_TRANSPORT only
when following previously present foreground (static rows say A_ABSENT).
Native0 but projected overlap, native>0 but sampled0, mixed winner, unobserved
return, insufficient hits and noise/out-of-range remain C_MIXED_OR_UNRESOLVED
with explicit subreason. A retained target-winning return is RETAINED_TARGET,
not forced into a failure class. Save actual source role of winning samples.

All-zone denominators: Core27,648 and Thin6,144 frame-zones. Immediate same-zone
transition denominators: Core25,344 and Thin5,632. Report full positive frames;
positive zones with sampled target inside corridor; B zone/frame counts,
>=4 subset, same-zone target-owned->far transitions (including a separate prior
<=3m subset), and actual baseline FN/low-score co-occurrence. Keep persistent
foreground suppression separate from a temporal flip. Boundary crossing is the
change in native target-occupied zone set between immediate frames; separate
this image-grid crossing from the original corridor boundary label.

Core HEAD/BODY strata use frozen tags. Thin strata use the eight original source
clips, without manufacturing HEAD labels. Report all existing FN identities and
whether their target-near zones lose to background, are missing, or never have
eligible target samples. Correlation with FN is not a counterfactual recovery.

No new method, threshold, capture, hardware call, training or automatic successor.
Preserve input/code/result hashes, candidate bins and detailed records under
`artifacts.local/work/ba-return-lineage-20260920/`. CPU source/array/metadata
replay is TASK_NOT_GPU_SUITABLE. Stop after the one fixed diagnostic and delivery.

Hardware reference: ST UM3109 Rev12, target-order and multi-target sections,
https://www.st.com/content/st_com/en/technical-documents/UM3109.html .
The manual supports up to4 reported targets/zone, defaults one and Strongest,
Closest ordering, and specifies600mm minimum target separation. The toy proxy
does not implement this detector, reflectance or physical target separability.
