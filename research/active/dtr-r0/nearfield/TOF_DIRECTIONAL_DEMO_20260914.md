# ToF directional readout engineering demo

The user's intended weighting preserves obstacle direction, rather than reducing
side-zone contributions to one collision-alert score. This additive prototype
reads the four consumed Willow frames captured for the color/ToF visual review.
It does not replace MZ116, relabel MZ123 FP, or establish improved alert precision.

## Fixed readout and scope

Each zone retains its horizontal/vertical angular midpoint. The eight horizontal
position coefficients are [-7,-5,-3,-1,1,3,5,7], multiplied by 2.8125 degrees;
vertical coefficients reverse that sequence from top to bottom. Positive-return
zones form four-connected angular support regions. Each region gets its own
quality-weighted bearing/elevation, so separated left/right support stays separate.
These are angular regions, not guaranteed object identities: touching objects or
different ranges in one zone can remain unresolved.

One zone contributes once, using 1 for SIM_VALID or 0.5 for merged-only support.
The weights are illustrative fixed values, not calibrated confidence. A central
bearing within +/-5.625 degrees is labeled CENTER; the other bearings are LEFT
or RIGHT. Outputs are relative to the camera. Vertical bearing is not body/head
height. Valid slant-range summaries exclude merged returns; they are not an
estimate of the exact object distance. Merged-only regions retain direction and
UNKNOWN metric distance. Missing packets/returns remain UNKNOWN.

MZ116's collision output remains intact. The new output suggests a position
notice; it does not infer collision risk, suppress an alert, infer a safe route,
or drive device speech. No new capture, training, threshold fitting or deployment.

## Observed result

| Consumed frame | Incumbent | Weighted direction | Valid slant median |
| --- | --- | --- | --- |
| willow-clear | No alert | UNKNOWN_NO_RETURNS | unavailable |
| willow-left | Alert | LEFT, -13.3875 degrees | 3.08 m |
| willow-right | Alert | RIGHT, +13.3875 degrees | 3.07 m |
| willow-center | Alert | CENTER, 0 degrees | 3.00 m |

Equal weights give the same categorical results. This demonstrates preservation
of directional information on the existing examples; it does not demonstrate a
quality-weighting advantage or general direction accuracy. An explicitly
artificial union of disjoint left/right observed zones yields two regions LEFT
and RIGHT, rather than a spurious CENTER. That check has no RGB assigned and is
not a simultaneous capture or a fifth empirical sample.

Five focused tests cover bilateral separation, merged-only direction with unknown
range, missing packet/no-return UNKNOWN, duplicate-target non-amplification and
single-zone preservation/vertical orientation. All pass. The four-frame replay
and artificial bilateral check pass. Stop this engineering demonstration here;
the registered MZ116 baseline and existing experiment ledger remain unchanged.

## Reproduction and retained outputs

```powershell
python research/active/dtr-r0/nearfield/test_tof_directional_readout.py
python research/active/dtr-r0/nearfield/run_tof_directional_demo.py --source artifacts.local/work/color-tof-20260914/returned-v6/capture-v6 --baseline artifacts.local/work/color-tof-20260914/baseline-predictions.json --output artifacts.local/work/tof-directional-demo-20260914/replay-v1
```

Use a fresh output directory. The ignored artifact directory retains results,
all 64 zone weights, source/code hashes and the four-frame color figure. No
persistent processes or worker resources were started for this demonstration.
