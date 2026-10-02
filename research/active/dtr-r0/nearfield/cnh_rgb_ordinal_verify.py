"""Independent raw-source audit of the fixed 15 ordinal-transport records.

Does not import the candidate transport or histogram implementation. Reuses
public camera/source geometry and the frozen boundary extractor, then checks
the extractor against its already-sealed original output. Source GT depth is
reduced to unordered histograms independently; labels are not read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from cnh_rgb_clearance_geometry import camera_geometry
from cnh_rgb_clearance_edge import extract_boundary, zone_map

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts.local/work'
CACHE = BASE / 'ba-nfo-depthpro-20260919'
PRIOR = BASE / 'cnh-rgb-clearance-probe-20261001'
EXPANDED = BASE / 'cnh-rgb-association-expanded-20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile_from_counts(counts, q, width=.05):
    """Independent loop implementation of piecewise-uniform bin inverse CDF."""
    total = float(sum(counts))
    if not total:
        return None
    target = q * total
    accumulated = 0.
    for k, count in enumerate(counts):
        if count > 0 and accumulated + count > target:
            return max((k + (target-accumulated)/count)*width, width/2 if k == 0 and target == 0 else 0.)
        accumulated += count
    # Exact q=1 follows the candidate's rightmost array-bin clamp. Its endpoint
    # issue is audited separately; none of the scientific cases should use it.
    return (len(counts)-.5)*width if counts[-1] == 0 else len(counts)*width


def metrics(rows, arm):
    errors = np.array([r['errors_m'][arm] for r in rows if r['errors_m'][arm] is not None])
    return dict(n=len(rows), valid=len(errors), hits2=int((abs(errors)<=.02).sum()),
                p50_cm=float(np.median(abs(errors))*100),
                p95_cm=float(np.percentile(abs(errors),95)*100))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    selection = read(PRIOR/'selection.json')
    original = {r['id']:r for r in read(PRIOR/'case-ledger.json')}
    expanded = read(EXPANDED/'ordinal-ledger.json')
    current = {r['id']:r for r in expanded}
    manifest = {r['id']:r for r in read(CACHE/'manifest.json')}
    cameras = {r['id']:r['camera_matrix'] for r in read(CACHE/'observations.json')}
    expanded_selection = {r['id']:r for r in read(EXPANDED/'selection.json')}
    assert len(selection) == 15 and len(current) == len(expanded) == 52
    assert all(e == expanded_selection[e['id']] for e in selection)
    checks = []
    inputs = {}
    rng = np.random.default_rng(2026100241)
    for e in selection:
        frame = e['frame_id']; row = manifest[frame]; camera = cameras[frame]
        geom = camera_geometry(ROOT, row, camera)
        depth_path = ROOT/'artifacts.local/datasets/hypersim-ba-nfo'/row['depth']
        cache_path = CACHE/'predictions/native'/f'{frame}.npz'
        for path in (depth_path,cache_path): inputs[str(path.relative_to(ROOT))] = sha(path)
        with h5py.File(depth_path,'r') as h: radial=h['dataset'][:].astype(np.float32)
        with np.load(cache_path,allow_pickle=False) as h: prediction=h['native_depth']
        zones = zone_map(camera)
        p = extract_boundary(prediction,geom['lateral_factor'],zones,e['zone_id'])
        assert p == original[e['id']]['prediction'] == current[e['id']]['prediction']
        assert p['status'] == 'OK'
        # Independently derive radial/optical conversion from full native rays.
        height,width = prediction.shape
        yy,xx = np.indices(prediction.shape)
        uv = np.stack(((xx+.5)*2/width-1,1-(yy+.5)*2/height,np.ones_like(xx)),axis=-1)
        rays = uv @ np.asarray(camera).T
        radial_to_z = (-rays[...,2]/np.linalg.norm(rays,axis=-1)).astype(np.float32)
        np.testing.assert_array_equal(radial_to_z,geom['optical_z_per_radial'])
        pred_radial = prediction/radial_to_z
        y,xhalf=p['edge_pixel']; x=int(np.floor(xhalf))
        conversion=float((radial_to_z[y,x]+radial_to_z[y,x+1])/2)
        prior=p['foreground_depth_m']/conversion
        cell=zones==e['zone_id']
        pred_values=pred_radial[cell]
        pred_valid=np.isfinite(pred_values)&(pred_values>0)
        truth_values=radial[cell]
        truth_valid=np.isfinite(truth_values)&(truth_values>0)&(truth_values<12.8)
        values=pred_values[pred_valid]
        # Python comparisons and independent np.histogram instead of bincount.
        q=float((np.count_nonzero(values<prior)+np.count_nonzero(values==prior)/2)/len(values))
        counts,_=np.histogram(truth_values[truth_valid],bins=np.arange(257)*.05)
        # Preserve finite-bin semantics at exact floating point boundaries by
        # independently checking that no input value lies exactly on an edge.
        perm_counts,_=np.histogram(rng.permutation(truth_values[truth_valid]),bins=np.arange(257)*.05)
        np.testing.assert_array_equal(counts,perm_counts)
        shuffled=rng.permutation(values)
        q2=float((np.count_nonzero(shuffled<prior)+np.count_nonzero(shuffled==prior)/2)/len(shuffled))
        assert q == q2
        radius=quantile_from_counts(counts,q)
        expected=current[e['id']]['details']
        np.testing.assert_allclose(q,expected['predicted_rank'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(radius,expected['radii']['ordinal_transport'],rtol=0,atol=1e-11)
        factor=p['side']*p['lateral_factor']*conversion
        clearance=factor*radius-.3
        np.testing.assert_allclose(clearance,current[e['id']]['estimates']['ordinal_transport'],rtol=0,atol=1e-11)
        sy,sx=np.asarray(e['support_yx']).T
        native_clear=float(np.median(e['side']*radial[sy,sx]*radial_to_z[sy,sx]*geom['lateral_factor'][sy,sx]-.3))
        np.testing.assert_allclose(native_clear,e['gt_clearance_m'],rtol=0,atol=1e-7)
        checks.append(dict(id=e['id'],frame_id=frame,scene=e['scene'],rank=q,
            radius_m=radius,clearance_m=clearance,error_m=clearance-e['gt_clearance_m'],
            zone_pixels=int(cell.sum()),predicted_valid=int(pred_valid.sum()),
            sensor_valid=int(truth_valid.sum()),valid_support_xor=int((pred_valid!=truth_valid).sum()),
            prediction_above_sensor_window=int((values>=12.8).sum()),
            permutation_invariant=True,original_edge_exact=True))
        print('verified',len(checks),'/15',flush=True)
    result_source=read(EXPANDED/'ordinal-result.json')
    arms=list(expanded[0]['estimates'])
    summaries={}
    for subset,rows in [('all',expanded),('original15',[r for r in expanded if r['original_record']]),
                        ('additional37',[r for r in expanded if not r['original_record']])]:
        summaries[subset]={}
        for arm in arms:
            for row in rows:
                value=row['estimates'][arm]
                assert value is not None
                np.testing.assert_allclose(value-row['gt_clearance_m'],row['errors_m'][arm],atol=1e-12,rtol=0)
            s=metrics(rows,arm); summaries[subset][arm]=s
            expected=result_source['summaries'][subset][arm]
            assert s['n']==expected['n'] and s['valid']==expected['valid']
            assert s['hits2']==round(expected['n']*expected['within_cm_all']['2'])
            np.testing.assert_allclose([s['p50_cm'],s['p95_cm']],
                [expected['absolute_error_cm_quantiles']['p50'],expected['absolute_error_cm_quantiles']['p95']],atol=1e-10,rtol=0)
    # Scene-based differences must not treat 52 records as independent.
    scenes=sorted({r['scene'] for r in expanded})
    n=np.array([sum(r['scene']==s for r in expanded) for s in scenes])
    sampled=np.random.default_rng(2026100201).integers(len(scenes),size=(2000,len(scenes)))
    for baseline in arms:
        if baseline=='ordinal_transport': continue
        deltas=np.array([sum(int(abs(r['errors_m']['ordinal_transport'])<=.02)-int(abs(r['errors_m'][baseline])<=.02)
                            for r in expanded if r['scene']==s) for s in scenes])
        ci=np.percentile(deltas[sampled].sum(1)/n[sampled].sum(1)*100,[2.5,97.5])
        np.testing.assert_allclose(ci,result_source['paired'][baseline]['ci95_pp'],rtol=0,atol=1e-10)
    result=dict(status='PASS',raw_source_verified_records=15,ledger_verified_records=52,
        frames=len({r['frame_id'] for r in expanded}),scenes=len(scenes),
        original_hits2=int(sum(abs(r['error_m'])<=.02 for r in checks)),
        support_mismatch_cases=sum(r['valid_support_xor']>0 for r in checks),
        original_edge_and_selection_exact=True,histogram_and_predicted_cdf_permutation_invariant=True,
        paired_scene_bootstrap_verified=True,source_sha256=inputs,summaries=summaries,cases=checks,
        limits=['Only original15 recomputed from native sources; all52 ledger arithmetic and scene CI independently checked',
                'Known target cell and clean GT selection remain; no target-ID recovery proof',
                'Histogram uses uniform first-visible pixel mass, not photon/reflectance mass',
                'Within-bin linear interpolation assumes uniform mass and is not measured subbin timing',
                'Predicted q=1 endpoint can map to empty last histogram bin; not triggered in original15'])
    for path in (PRIOR/'selection.json',PRIOR/'case-ledger.json',EXPANDED/'ordinal-ledger.json',EXPANDED/'ordinal-result.json',Path(__file__)):
        result.setdefault('audit_sha256',{})[str(path.relative_to(ROOT))]=sha(path)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','raw_source_verified_records','original_hits2','support_mismatch_cases']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=EXPANDED/'independent-ordinal-v2')
    run(parser.parse_args().out)
