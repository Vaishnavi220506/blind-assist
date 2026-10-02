"""Frozen mask-zone selector replayed on all 52 eligible Hypersim edges.

The selector, mask, H3 proxy and per-frame seed rule are imported unchanged
from the 15-case run (35eb3329). The 37 additional edges were never scored by
this selector; they share the original 19 frames/8 scenes, so this is a
rule-frozen reuse of consumed Development, not independent confirmation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cnh_rgb_clearance_geometry import reference_frame
from cnh_rgb_clearance_probe import summarize
from cnh_rgb_multizone_calibration import read, save, sha
from cnh_rgb_zone_sensor import sensor_readouts
from cnh_tof_dither_pilot import prediction_mask
from cnh_tof_mask_zone import estimate, mask_overlap

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
EXPANDED = WORK/'cnh-rgb-association-expanded-20261002'
PRIOR15 = WORK/'cnh-tof-mask-zone-20261002'
CACHE = WORK/'ba-nfo-depthpro-20260919'
OUT = WORK/'cnh-tof-mask-zone-expanded-20261002'
ARMS = ('depthpro', 'zone_q10', 'clean_query_bin', 'clean_maskbest_bin', 'clean_query_fit',
        'clean_maskbest_fit', 'stress_query_bin', 'stress_maskbest_bin', 'stress_query_fit', 'stress_maskbest_fit')
PRIMARY = ('clean_maskbest_bin', 'clean_query_bin')
RULE = ('Primary = 37 additional edges, clean_maskbest_bin vs clean_query_bin, <=2cm success (missing = failure). '
        'SUPPORTED if rescued-lost>=3 AND rescued>=2*lost AND >=1 rescue outside building ai_051; '
        'NOT_SUPPORTED if rescued<=lost; otherwise UNCERTAIN. Stress/fit arms, all-52 and '
        'near-body-line subsets are descriptive only. No arm, threshold or selector change after results.')


def ok(row, arm):
    e = row['errors_m'][arm]
    return e is not None and abs(e) <= .02


def paired(rows, a, b):
    return dict(rescued=[r['id'] for r in rows if ok(r, a) and not ok(r, b)],
                lost=[r['id'] for r in rows if ok(r, b) and not ok(r, a)])


def verdict(rows):
    p = paired(rows, *PRIMARY)
    rescued, lost = len(p['rescued']), len(p['lost'])
    outside = sum(not i.startswith('hyp-ai_051') for i in p['rescued'])
    if rescued <= lost:
        return 'NOT_SUPPORTED'
    if rescued-lost >= 3 and rescued >= 2*lost and outside >= 1:
        return 'SUPPORTED'
    return 'UNCERTAIN'


def main(out):
    if out.exists():
        raise FileExistsError('Preserve outputs; choose a fresh directory')
    out.mkdir(parents=True)
    started = time.perf_counter()
    selection, ordinal = read(EXPANDED/'selection.json'), read(EXPANDED/'ordinal-ledger.json')
    prior = {r['id']: r for r in read(PRIOR15/'case-ledger.json')}
    save(out/'PLAN.json', dict(rule=RULE, frozen_from='35eb3329 cnh_tof_mask_zone.py (estimate, mask_overlap, select_zone)',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (EXPANDED/'selection.json', EXPANDED/'ordinal-ledger.json',
                PRIOR15/'case-ledger.json', CACHE/'observations.json', CACHE/'manifest.json', Path(__file__))},
        role='EXPLORE; consumed Development; 37 additional edges share the original 19 frames/8 scenes'))
    assert len(selection) == len(ordinal) == 52 and [e['id'] for e in selection] == [o['id'] for o in ordinal]
    manifest = {r['id']: r for r in read(CACHE/'manifest.json')}
    cameras = {r['id']: r['camera_matrix'] for r in read(CACHE/'observations.json')}
    frames, ledger = {}, []
    for event, old in zip(selection, ordinal):
        fid = event['frame_id']
        if fid not in frames:
            ref = reference_frame(ROOT, manifest[fid], cameras[fid])
            seed = int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16)
            with np.load(CACHE/'predictions/native'/f'{fid}.npz', allow_pickle=False) as blob:
                depth = blob['native_depth']
            frames[fid] = dict(ref=ref, depth=depth, electronics=sensor_readouts(ref['radial'], cameras[fid], seed))
        f = frames[fid]
        prediction = old['prediction']
        assert prediction['status'] == 'OK'
        y, xhalf = prediction['edge_pixel']
        x = int(np.floor(xhalf))
        optical_per_radial = float(np.mean(f['ref']['optical_z_per_radial'][y, x:x+2]))
        factor = event['side']*prediction['lateral_factor']*optical_per_radial
        mask, meta = prediction_mask(f['depth'], prediction)
        observation = dict(id=event['id'], query_zone=event['zone_id'], radial_edge_factor=factor,
                           electronics=f['electronics'], **mask_overlap(mask, cameras[fid]))
        decoded = estimate(observation)
        estimates = dict(depthpro=old['estimates']['depthpro'], zone_q10=old['estimates']['zone_q10'], **decoded['estimates'])
        ledger.append(dict(id=event['id'], scene=event['scene'], frame_id=fid, query_zone=event['zone_id'],
            original=event['id'] in prior, gt_clearance_m=event['gt_clearance_m'], mask=meta,
            selection=decoded['selection'], estimates=estimates,
            errors_m={a: v-event['gt_clearance_m'] if v is not None else None for a, v in estimates.items()}))
        print(f'{len(ledger)}/52 query={event["zone_id"]} selected={decoded["selection"]["zone"]}', flush=True)
    # The 15 original rows must reproduce the committed run exactly.
    for r in ledger:
        if r['original']:
            for a in ARMS:
                assert r['estimates'][a] == prior[r['id']]['estimates'][a], (r['id'], a)
    save(out/'case-ledger.json', ledger)
    subsets = dict(additional37=lambda r: not r['original'], original15=lambda r: r['original'], all52=lambda r: True,
                   near_line_2cm=lambda r: abs(r['gt_clearance_m']) <= .02)
    result = dict(status='COMPLETE', rule=RULE, verdict=verdict([r for r in ledger if not r['original']]),
        original15_reproduced=True, frames=len(frames), seconds=None, subsets={})
    for name, keep in subsets.items():
        rows = [r for r in ledger if keep(r)]
        result['subsets'][name] = dict(n=len(rows), scenes=len({r['scene'] for r in rows}),
            changed_zone=sum(r['selection']['zone'] != r['query_zone'] for r in rows),
            arms={a: summarize(rows, a) for a in ARMS},
            paired={f'{c}_maskbest_{k}': paired(rows, f'{c}_maskbest_{k}', f'{c}_query_{k}')
                    for c in ('clean', 'stress') for k in ('bin', 'fit')},
            vs_q10=paired(rows, 'clean_maskbest_bin', 'zone_q10'))
    result['seconds'] = time.perf_counter()-started
    save(out/'result.json', result)
    print(json.dumps({k: (v['n'], v['changed_zone'], {a: round(v['arms'][a]['within_cm_all']['2']*v['n']) for a in ARMS},
          {p: (len(x['rescued']), len(x['lost'])) for p, x in v['paired'].items()},
          (len(v['vs_q10']['rescued']), len(v['vs_q10']['lost']))) for k, v in result['subsets'].items()}, indent=1))
    print('VERDICT', result['verdict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
