"""Read-only audit of existing public ToF payloads; no sensor/model/native inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
DEFAULT_OUT = WORK/'ba-tof-near-component-audit-20260920'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def npz_cohort(root, rows, expected_fields, hash_key):
    valid, identities = 0, []
    for row in rows:
        path = root/row['prepared']
        digest = sha(path)
        assert digest == row[hash_key], ('Source hash mismatch', path)
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == expected_fields, ('Unexpected fields', path, data.files)
            values = data['values']
            assert values.shape == (64,) and np.issubdtype(values.dtype, np.floating)
            assert not np.isinf(values).any()
            valid += int(np.isfinite(values).sum())
        identities.append(dict(id=row['id'],file=str(path.relative_to(ROOT)),sha256=digest))
    return dict(frames=len(rows),zones=len(rows)*64,finite_single_returns=valid,
        missing_single_returns=len(rows)*64-valid,fields=sorted(expected_fields),
        values_shape=[64],maximum_stored_returns_per_zone=1,measured_photon_bins=False,
        authority='SIMULATED_SINGLE_RETURN',identities=identities)


def audit(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'availability.json').exists(), 'Use a new audit output; preserve completed evidence'
    corridor = WORK/'ba-camera-corridor-20260919'
    observations = read(corridor/'observations.json')
    assert len(observations) == 96
    assert sha(corridor/'observations.json') == read(corridor/'observation-seal.json')['observations_sha256']
    c = npz_cohort(corridor,observations,{'rgb','boxes','values'},'prepared_sha256')
    manifest = WORK/'ba-nfo-matched-20260919/manifest.json'
    assert sha(manifest) == read(manifest.with_name('protocol.json'))['manifest_sha256']
    rows = [r for r in read(manifest) if r['split'] == 'test']
    assert len(rows) == 500
    nfo = npz_cohort(WORK/'ba-nfo-20260919',rows,{'rgb','depth','raw','mixed','boxes','values'},'sha256')
    nfo['reference_arrays_loaded'] = False

    probe = WORK/'ba-depth-probe-20260918'
    selected = read(probe/'protocol.json')['selected']
    assert len(selected) == 160
    identities, mask_valid, second_nonzero = [], 0, 0
    for row in selected:
        path = probe/'inputs'/row['filename']
        digest = sha(path)
        assert digest == row['sha256'], ('ZJU source changed', path)
        with h5py.File(path,'r') as data:
            assert set(data) == {'rgb','depth','fr','hist_data','mask'}, ('Unexpected HDF fields',path)
            assert data['hist_data'].shape == (64,2) and data['mask'].shape == (64,)
            hist, mask = data['hist_data'][:], data['mask'][:].astype(bool)
            mask_valid += int(mask.sum())
            second_nonzero += int((mask & np.isfinite(hist[:,1]) & (hist[:,1]>0)).sum())
        identities.append(dict(file=str(path.relative_to(ROOT)),sha256=digest))
    zju = dict(frames=160,zones=160*64,fields=['depth','fr','hist_data','mask','rgb'],
        hist_data_shape=[64,2],valid_mask_zones=mask_valid,positive_second_parameter_zones=second_nonzero,
        authority='EXISTING_PUBLIC_REAL_ZJU_L5',interpretation='Two Gaussian parameters (location and scale) in author loader; not two detected returns or time-bin counts',
        native_depth_arrays_loaded=False,thin_rod_weak_peak_test='NOT_EVALUABLE_WITH_THIS_REPRESENTATION',identities=identities)

    seal_path = WORK/'corridor-astar-effect-20260917/replay/input-seal.json'
    seal = read(seal_path)
    raw = WORK/'corridor-public-single-20260917/source/returned-v1/capture-v1/raw.jsonl'
    bound = [(p,h) for p,h in seal['inputs'].items() if Path(p).resolve() == raw.resolve()]
    assert len(bound) == 1 and sha(raw) == bound[0][1]
    frames, counts, statuses, target_fields = 0, Counter(), Counter(), set()
    for line in raw.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        assert row['tof_model'] == 'HYPOTHETICAL_FINITE_FOOTPRINT_V1'
        assert row['tof_max_targets'] == 2 and row['tof_target_order'] == 'strongest'
        assert len(row['tof_zones']) == 64
        assert not any(k for k in row if any(s in k.lower() for s in ('histogram','cnh','photon')))
        for zone in row['tof_zones']:
            assert set(zone) == {'zone_id','theta_bounds_deg','phi_bounds_deg','target_count','targets'}
            count = zone['target_count']
            assert count == len(zone['targets']) and 0 <= count <= 2
            counts[count] += 1
            for target in zone['targets']:
                assert set(target) == {'distance_m','range_noise_sigma_m','signal_strength_proxy','status'}
                target_fields.update(target)
                statuses[target['status']] += 1
        frames += 1
    assert frames == 288
    dual = dict(frames=frames,zones=frames*64,zones_by_return_count=dict(counts),
        returned_targets=sum(k*v for k,v in counts.items()),statuses=dict(statuses),target_fields=sorted(target_fields),
        authority='HYPOTHETICAL_TWO_RETURN_SIMULATION',measured_photon_bins=False,
        raw_sha256=sha(raw),bound_input_seal_sha256=sha(seal_path))
    source_names = ['ba_nfo_data.py','run_ba_camera_corridor.py','ba_depth_probe.py',
        'mz115_zonal_tof.py','mz115_zonal_sensors.py','audit_tof_near_component_availability.py']
    result = dict(status='INPUT_AVAILABILITY_AUDIT_COMPLETE',question_status='REAL_THIN_FOREGROUND_WEAK_RETURN_NOT_EVALUABLE',
        user_hardware_statement='No owned 8x8 ToF board or measured data, confirmed 2026-09-20',
        cohorts=dict(camera_corridor=c,nfo_fixed_test=nfo,zju_existing_real=zju,retained_two_return_source=dual),
        total_records=96+500+160+288,model_calls=0,new_captures=0,simulator_calls=0,
        evaluator_jsonl_opened=False,native_depth_arrays_loaded=False,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in source_names},
        input_manifest_hashes={str(p.relative_to(ROOT)):sha(p) for p in [corridor/'observations.json',manifest,probe/'protocol.json',seal_path]},
        limitations=['Scoped existing cohorts only; not an exhaustive filesystem inventory',
            'Absence of recorded bins is not absence of physical near photons',
            'Private ray/reflectance histograms cannot be relabeled measured CNH',
            'ZJU second Gaussian parameter does not reveal separate foreground peak identity'],elapsed_s=time.perf_counter()-started)
    with (out/'availability.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','question_status','total_records','elapsed_s']}))
    print(json.dumps({k:{x:y for x,y in v.items() if x!='identities'} for k,v in result['cohorts'].items()},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    audit(parser.parse_args().output)
