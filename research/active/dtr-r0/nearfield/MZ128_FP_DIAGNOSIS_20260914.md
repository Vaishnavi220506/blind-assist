# MZ128: attribution of all 103 remaining false-positive frames

## Finding

The dominant observed failure is **real off-corridor surfaces acquiring an
overlapping predicted spatial extent**, not the angular connected-component
background bridge demonstrated by the separate ToF direction experiment.
101/103 FP frames have current real-object support sufficient for an alert;
the other two require inherited historical range support. This statement is
about the saved simulation and native evaluator geometry, not hardware accuracy.

The frozen `binary_four_mz125` arm remains139 TP /103 FP /5 FN /41 TN. No
predictor, threshold, proposal, sensor model or default has been changed.
This is supplemental evaluator-side diagnosis of the existing MZ128 terminal,
not a new trained candidate or independent validation experiment.

## Exact population and branch accounting

All288 IDs, saved binary outputs and native-object truth are reproduced before
selecting the103 FP IDs. A frame is FP only when its frozen alert is true and
the native-bounds corridor label is false. The corridor is forward0.2–3.6 m,
lateral-0.3–0.3 m and height0.4–2.05 m. It is not the2 m notice criterion from
the later artificial ToF direction panel.

| Individually sufficient alert paths | FP frames |
| --- | ---: |
| ToF only | 67 |
| Radar only | 14 |
| Both ToF and Radar | 22 |
| **Total** | **103** |

ToF is sufficient on89 frames; Radar on36. Guard events and certain-coarse
overrides are both zero on these103. Fixing one branch does not remove a frame
where the other branch remains sufficient.

The family counts are HEAD13, rod+farwall25, BODY30 and boundary-stress35.
These are scene families, not causal labels:68 FP occur outside the boundary
stress family. The full103-ID listing is retained in `combined-audit.json` and
the per-frame ToF and Radar reports.

## ToF: finite footprint and unresolved range

| Sufficient ToF support status | Frames | ToF-only / also Radar |
| --- | ---: | ---: |
| Valid returns only | 62 | 44 /18 |
| Merged returns only | 22 | 19 /3 |
| Both statuses | 5 | 4 /1 |

These89 frames contain422 positive-weight supporting return records:340 VALID
and82 MERGED. Of them270 retain whole-zone extents and152 use an RGB association
proxy. All422 are matched to known native actors, with **zero native contributor
hits and zero contributing actor bounds inside the specified corridor**. The
predicted intervals nevertheless intersect it. This is verified using native
object bounds as well as sampled rays; missing samples alone are not used to
certify absence of a hazard.

Far-wall support participates in24 ToF FP frames:10 have only far-wall sources
and14 also have near-rod sources. Non-far-wall actor support participates in79;
these counts overlap. This identifies real background participation, not an
angular bridge. There are415 single-actor returns and seven multi-actor returns
across four frames. In particular, **75/82 MERGED returns still have only one
source actor**: SIM_MERGED is not synonymous with two different near/far objects.

The implementation treats a MERGED return as range0.02–4 m and retains the full
zone. That uncertainty can reach the corridor even when the actual contributors
are clustered near3 m at its side. It also cannot be discarded simply because
its reported average looks far.

Two fixed evaluator-only checks retain Radar/guards and all other logic:

| Diagnostic control | TP / FP / FN | Meaning |
| --- | ---: | --- |
| Frozen binary arm | 139 /103 /5 | Reference |
| Remove all MERGED ToF supports | 139 /84 /5 | 19 FP require this support in the saved panel |
| Replace only MERGED range bounds with native contributor min/max | 139 /92 /5 | 11 FP depend on the broad unknown-range envelope under this oracle |

These are causal diagnostics, **not deployable corrections**. The native min/max
is evaluator-only; deleting all merged support has no general near-obstacle
retention guarantee even though this consumed panel loses no TP. The remaining
errors are not all repairable by knowing the merged depth: angular extent,
interval enclosure, RGB localization and other sufficient branches remain.

All422 ToF supports have evidence_age_s=0. Of340 VALID returns, only two exceed
the model's3-sigma range-noise bound (largest absolute error0.1531 m). The error
is measured against the native pre-noise representative range, not assumed
object depth. This does not calibrate reliability or prove range noise irrelevant;
it argues against interpreting the entire population as spurious ToF returns.

## Radar: offroute association dominates current false support

Of36 Radar-sufficient FP,33 are reproduced by the inherited current Radar
geometry; three require the inherited flow/range-carry addition. Current
geometry is reconstructed with the **original cached RGB proposals**, since
MZ125 changed ToF allocation while preserving the old common-Radar branch.
No RGB pixels, optical flow or learned model is rerun.

Of the33 current supports:

- 30 come from real native actors; all30 native hit points and full actor lateral
 bounds lie outside the +/-0.3 m corridor. All30 use `ASSOCIATED_PROXY`. The raw
 Radar center point itself does not trigger on any of these30, whereas its
 associated visual spatial extent does.
- 3 are authored transient random returns; all three frames also have sufficient
 ToF support. Removing these transient returns alone removes zero whole-frame FP.
 There are no persistent-ghost supports in this population.

Real native horizontal ranges are2.373–3.314 m, with maximum observed range error
0.1496 m. Keeping the same visual box but substituting exact native range still
supports28/30 real returns. The two range-sensitive cases also have ToF support,
so this substitution removes zero whole-frame FP. Extent/association remains
the larger measured issue for this branch.

Three carry-dependent Radar FP are `mz123_suspended_head_pair0_out_01`,
`mz123_suspended_head_pair0_out_09` and
`mz123_shallow_boundary_stress_pair2_enter_04`. The first two are Radar-only;
the third also has ToF. Their Radar packets arrive with no valid current slots.
The inherited formula and frozen maximum-age0.5 s constrain this attribution;
the exact optical-flow ancestry/age was not replayed and is not claimed.

## Concrete examples

1. `mz123_near_rod_farwall_pair1_out_09`, ToF zone50: the measured range is
   3.04 m. The real rod's lateral extent is[-0.536,-0.497] m, entirely outside
   the corridor. The predicted return interval is[-0.600,-0.168] m, extending
   across the-0.3 m boundary and triggering the alert.
2. `mz123_suspended_head_pair2_out_02`, MERGED ToF zone21: actual contributor
   ranges3.375–3.492 m and actor lateral extent[0.521,0.808] m. The reported
   3.44 m tuple is represented as0.02–4 m, producing a lateral interval starting
   near0.002 m. Native same-actor surface spread, not a near pole at0.02 m,
   generated this uncertainty in the simulation.
3. `mz123_suspended_head_pair0_out_00`, Radar slot0: observed range2.45 m,
   native2.466 m; real actor lateral extent[0.473,0.826] m. A cached RGB box
   spanning pixels[67,36,574,360] supplies a large associated extent that reaches
   the corridor, although the raw Radar centerline does not.

## Temporal accounting and depth-graph applicability

The103 FP form26 consecutive segments across15 episodes: eight singleton
segments and95 FP frames belonging to multi-frame segments. False-bin duration
is25.75 s at0.25 s per frame. Boundary stress includes18/18 pre-entry negatives
and17/18 post-exit negatives; it is not only a one-frame transition artifact.

Four FP occur during actual ToF packet loss. Radar packet loss and invalid IMU
are zero among FP. Eight FP have no valid Radar slot, which is distinct from
packet loss. Thus packet absence and short-lived flicker are present but do not
explain the bulk of the observed errors.

MZ128 has **no ToF angular connected-component/centroid alert stage**. It already
uses complete-link forward-depth cohorts with a0.25 m span for association,
then per-return possible spatial intersection plus independent Radar and guards.
The RGB frontend does use2D connected components; that is a different operation.
Consequently, zero of these103 can be assigned to the specific absent ToF
angular-centroid bridge mechanism. A future integration of the0.30 m direction
component has no measured correction count on this population; its22/22 result
from another60-packet panel cannot be transferred here.

The next decision should focus on **how trustworthy range evidence acquires an
overbroad or incorrect corridor extent**, including whole-zone uncertainty and
the inherited Radar visual boxes. This diagnosis does not authorize a parameter
sweep, suppression policy, integration experiment or source expansion.

## Reproduction and integrity

[Main diagnostic](mz128_false_positive_diagnosis.py) uses the standard library
and existing scalar evaluator geometry. It independently reconstructs every
one of288 frozen binary readouts, verifies native labels, matches source and
prediction seals, audits supporting return lineage and preserves input hashes.
An independent first-level audit verifies103 IDs/branch bits, reconstructs
current Radar geometry, and checks temporal segments and source provenance.

```powershell
python research/active/dtr-r0/nearfield/mz128_false_positive_diagnosis.py --output artifacts.local/work/mz128-fp-diagnosis-20260914/tof-native-audit-v2
```

Use a new output directory on replay. Evidence root:
`artifacts.local/work/mz128-fp-diagnosis-20260914/`. Canonical ToF output is
`tof-native-audit-v2/`; v1 preserves the initial diagnosis before noise/role
annotations were added. `radar-time-audit/` retains scripts, reconstructed
current Radar records, temporal/source reports and hashes; `combined-audit.json`
cross-checks the joint source accounting. The existing project scientific Python
runtime was used for the Radar geometry; no package was installed. All commands
ended and no worker, GPU, capture, download or persistent allocation remains.

The existing MZ128 registration/inheritance still has its documented ledger303
fingerprint blocker. This supplemental audit does not rewrite that ledger or
claim a newly registered terminal. Algorithms and frozen decisions are unchanged.
