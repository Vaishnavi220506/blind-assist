# Existing ToF near-component availability audit

2026-09-20. **Input audit complete; real thin-foreground weak-return existence
is NOT_EVALUABLE with the inspected inputs.** The user confirmed no owned 8x8
ToF board or measured captures. Existing downloaded ZJU-L5 real data were also
checked; their recorded representation does not expose independent weak peaks.
The previous suggestion to inspect discarded raw echoes omitted this prerequisite.
Do not treat this outcome as a negative sensor or algorithm result.

## What was checked

The read-only [audit](audit_tof_near_component_availability.py) inspected 1,044
existing records in four fixed cohorts. Every inspected NPZ/HDF5 payload matches
its existing manifest hash; the 288-frame public JSONL matches the retained A*
input seal. No model, simulator, evaluator JSONL, native depth array, device
transport or new capture was executed/read. Hashing complete containers does not
use their reference-depth arrays for inference. This is a scoped inventory, not
a claim to have searched every file on the machine.

| Existing input | Records | Actually saved ToF representation | Can inspect a separate physical near peak? |
| --- | ---: | --- | --- |
| Camera-corridor observations | 96 | 64 scalar simulated ranges; 4,528 finite / 6,144 zones | No waveform or second target |
| Fixed BA-NFO test split | 500 | 64 scalar simulated ranges; 24,116 finite / 32,000 zones | No waveform or second target |
| Previously selected real ZJU-L5 | 160 | `hist_data[64,2]` plus projected rectangles and mask; 6,818 valid / 10,240 zones | Gaussian parameters, not independent peaks |
| Retained A* public source | 288 | Up to two simulated targets; 403 double-return / 18,432 zones | Both reported slots already public; no measured waveform |

The corridor NPZ fields are exactly `rgb`, `boxes`, `values`. BA-NFO adds
`depth`, `mixed`, `raw`: the first two are reference-derived, while `raw` is the
zone value expanded over its rectangle. They are not additional sensor returns.
The [sensor constructor](ba_nfo_data.py) creates a depth-derived weighted histogram
internally and saves only the strongest bin's noisy scalar. It does not save a
time-resolved photon measurement. Regenerating a histogram from reference depth
would create a new simulated input, not recover a lost physical observation.

The retained two-return source has 12,178 empty, 5,851 single-return and 403
double-return zones, totaling 6,657 targets. Its statuses are 4,840 `SIM_VALID`
and 1,817 `SIM_MERGED`. The public targets retain distance, synthetic noise sigma,
signal proxy and status. [The generator](mz115_zonal_tof.py) explicitly labels
this a hypothetical finite-footprint model. Its 81-bin ray/reflectance histogram
and hit lineage remain private evaluator fields in [the wrapper](mz115_zonal_sensors.py).
Those private files were not opened for this audit. Missing stored bins do not
establish missing photons, and `SIM_MERGED` is not proof of two real surfaces.

## ZJU-L5 field names do not imply a recorded photon histogram

All 160 HDF5 inputs have exactly `rgb`, `depth`, `fr`, `hist_data`, `mask`, with
`hist_data` shape 64x2. The author's source at revision
`f3c517562bcb2dfb9e4bbf74965f2ee76a2dd096` passes the two columns as location and
scale to a Gaussian distribution, or samples location +/-3 scale; it does not
decode two detected targets or independent time-bin amplitudes.
See [DELTAR's sampling implementation](https://github.com/zju3dv/deltar/blob/f3c517562bcb2dfb9e4bbf74965f2ee76a2dd096/src/utils/dataloader.py#L65).

Our prior [raw expansion](ba_depth_probe.py) consumes the first column. The second
column is still present in the retained HDF5 files, so it is not irretrievably
discarded. It could support a separately defined uncertainty study, but sampling
from that Gaussian would not reveal an observed weak foreground peak or its
pixel ownership. This audit does not equate that scale parameter with physical
foreground/background spread or claim hardware `range_sigma` semantics.
Both author source files and URL/revision/SHA receipts were saved locally.

## Actual adapter versus research packet interface

The [implemented hardware route](../../../../docs/GLASSES_HARDWARE_ROUTE.md) is
AtomS3R-M12 + Unit ToF4M with one scalar distance, validity, timestamp and age.
The Android parser has no multi-target or CNH fields. It filters unrecognized
headers and nulls an invalid scalar, but there is no evidence that its firmware
ever sent additional near-target slots. Installed chip/firmware identity was not
verified; the old firmware source path in the hardware note was absent.

The [VL53L8CX research packet](vl53l8cx_measurement_packet.py) accepts 16/64 zones
and 1..4 already-reported slots, preserving counts, status and optional quality
fields, including sigma. It contains no histogram/CNH field and is not an
integrated device driver. The [model adapter](tof_model_adapter.py) preserves
reported slots and their ordering; its frozen range/valid bridge rejects modes
beyond 64 zones with one/two slots rather than silently truncating them. Enabling
more fields in that Python schema would not itself acquire new measurements.
Exact source paths, line ranges and hashes are in `driver-interface-audit.json`.

## Decision and evidence boundary

The availability check is complete. The empirical question of whether a thin
foreground produces a stable usable near component remains unanswered. There is
no newly discovered measured near-return stream in these inspected records.
The previous [CPU contour failure](BA_CONTOUR_PARALLAX_20260919.md) stays scoped
to its tested recipe; this audit supplies no additional negative accuracy result.

A valid physical test needs measured multi-target output or genuine recorded CNH
with sensor/firmware/mode, timing and field semantics preserved, plus fixed
thin-object-present/absent observations against the same wall. Report detected
target count/status and valid-frame coverage, not only successful near ranges.
Existing scalar files cannot supply this experiment retrospectively. A new
simulation could test a declared information ceiling, but could not answer the
physical weak-peak question; no such successor was launched here.

This input/adapter inventory changes no algorithm, threshold, model, firmware or
runtime default and introduces no new algorithm-evaluation terminal. It does not
attempt to repair the unrelated global experiment registry.

Payload root: `artifacts.local/work/ba-tof-near-component-audit-20260920/`.
Retain `availability.json`, `audit.log`, `driver-interface-audit.json`, author
source snapshots/receipts and delivery verification. The scalar/file audit took
6.704 s on CPU; this is bookkeeping time, not an inference benchmark. No new
models/datasets were downloaded, apart from two small author source files. All
commands finished, no task-owned device connection or background allocation
remains, and no files were deleted.
