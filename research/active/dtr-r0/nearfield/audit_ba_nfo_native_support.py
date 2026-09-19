"""Read-only native support audit of all 16 sealed hard-fit TRAIN positives.

Area fractions diagnose sampling; connected depth support is not object identity.
No model, sensor regeneration, fitting, cutoff selection or held-out access.
"""
import csv
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

import ba_nfo_data as data

ROOT = Path(__file__).resolve().parents[4]
FIT = ROOT / 'artifacts.local/work/ba-nfo-jointfit32-20260919'
OUT = ROOT / 'artifacts.local/work/ba-nfo-native-support-audit-20260919'


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def footprint_mean(a, h, w):
    """Exact rectangular block means, only for integral native-to-input scale."""
    nh, nw = a.shape[:2]
    assert nh % h == nw % w == 0
    return a.reshape(h, nh // h, w, nw // w, *a.shape[2:]).mean(axis=(1, 3))


def connected_support(mask, box):
    y0, x0, y1, x1 = box
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    ids, inside = np.unique(labels[y0:y1, x0:x1][mask[y0:y1, x0:x1]], return_counts=True)
    parts = []
    for idx, n in zip(ids, inside):
        x, y, w, h, total = map(int, stats[idx])
        parts.append(dict(inside=int(n), total=total, outside=total-int(n), bbox=[y, x, y+h, x+w]))
    assert sum(p['inside'] for p in parts) == int(mask[y0:y1, x0:x1].sum())
    return dict(components=parts, all_extend_outside=bool(parts) and all(p['outside'] > 0 for p in parts),
                total_connected_pixels=sum(p['total'] for p in parts), zone_pixels=int(mask[y0:y1, x0:x1].sum()))


def occupancy_summary(mask, fraction, unknown):
    f, u = fraction[mask], unknown[mask]
    return dict(pixels=int(mask.sum()), zero_near=int((f == 0).sum()), any_near=int((f > 0).sum()),
                below_quarter=int((f < .25).sum()), at_least_quarter=int((f >= .25).sum()),
                below_half=int((f < .5).sum()), at_least_half=int((f >= .5).sum()),
                mean_near_fraction=float(f.mean()) if f.size else None,
                mean_known_fraction=float((1-u).mean()) if f.size else None,
                mean_near_among_known=float((f/(1-u))[u < 1].mean()) if (u < 1).any() else None)


def main():
    start = time.perf_counter()
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'results.json').exists(), 'Do not overwrite completed audit'
    selection = json.loads((FIT / 'selection.json').read_text())
    selected = [(i, r) for i, r in enumerate(selection) if r['kind'] == 'positive']
    assert len(selected) == 16 and all(r['row']['split'] == 'train' for _, r in selected)
    failed = {r['id'] for r in json.loads((FIT / 'failed-zones.json').read_text())}
    protocol = dict(scope='READ_ONLY_16_CONSUMED_TRAIN_POSITIVES', question='Sampling loss or zone-clipped support?',
        backend='CPU: TASK_NOT_GPU_SUITABLE; 16 file decodes, exact array reductions and connected-component diagnostics; no neural compute',
        selection_sha256=data.sha(FIT/'selection.json'), scores_sha256=data.sha(FIT/'fit-scores.npz'),
        code_sha256=data.sha(__file__), definitions='Near: known axial depth <2m. Fractions over complete RGB footprints; unknown separate. Components: full-image 8-connected near mask, NOT instances.',
        checks=['reproduce prepared RGB/depth exactly', 'NN-positive footprint near occupancy', 'negative-labeled footprints containing near support', 'whole-image connected support beyond selected box', 'saved joint errors versus footprint occupancy'],
        stop='All 16 inspected; no training, validation, test, sensor alteration, sample deletion or revised old score')
    write('protocol.json', protocol)
    cameras = {r['scene_name']: r for r in csv.DictReader((data.WORK/'metadata_camera_parameters.csv').open())}
    source_hashes = {str(data.WORK/'metadata_camera_parameters.csv'): data.sha(data.WORK/'metadata_camera_parameters.csv')}
    with np.load(FIT/'fit-scores.npz') as a:
        scores = a['scores'].copy()
    results, panels, totals = [], [], dict(near=0, minority=0, quarter=0, fp=0, fp_any=0, fp_majority=0, fn=0, fn_minority=0)
    for i, selected_row in selected:
        r = selected_row['row']; ident = r['id']
        assert r['source'] == 'hypersim'
        rgb_path, dep_path = data.HYP/r['rgb'], data.HYP/r['depth']
        prepared_path = data.WORK/r['prepared']
        for p, key in [(rgb_path, 'rgb_sha256'), (dep_path, 'depth_sha256'), (prepared_path, 'sha256')]:
            actual = data.sha(p); assert actual == r[key], str(p)
            source_hashes[str(p)] = actual
        native_rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        with h5py.File(dep_path) as f:
            radial = f['dataset'][:].astype(np.float32)
        native_depth = radial * data.axis_factor(cameras[r['scene']], radial.shape)
        assert native_rgb.shape[:2] == native_depth.shape
        with np.load(prepared_path) as a:
            rgb, depth = a['rgb'].copy(), a['depth'].copy()
        h, w = depth.shape; nh, nw = native_depth.shape
        assert (h, w) == (192, 256) and nh % h == nw % w == 0
        sy, sx = nh//h, nw//w
        regenerated = cv2.resize(native_depth, (w, h), interpolation=cv2.INTER_NEAREST)
        regenerated[~np.isfinite(regenerated) | (regenerated <= .001)] = np.nan
        np.testing.assert_array_equal(regenerated, depth)
        np.testing.assert_array_equal(cv2.resize(native_rgb, (w, h), interpolation=cv2.INTER_AREA), rgb)
        # Independent index/block computations verify cv2 sampling coordinates.
        sampled = native_depth[::sy, ::sx].copy()
        sampled[~np.isfinite(sampled) | (sampled <= .001)] = np.nan
        np.testing.assert_array_equal(sampled, depth)
        block_rgb = footprint_mean(native_rgb.astype(np.float64), h, w)
        assert np.abs(block_rgb-rgb.astype(float)).max() <= .500001
        known = np.isfinite(native_depth) & (native_depth > .001)
        near = known & (native_depth < 2)
        fraction = footprint_mean(near.astype(float), h, w)
        unknown = footprint_mean((~known).astype(float), h, w)
        np.testing.assert_allclose(fraction, cv2.resize(near.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), atol=1e-7)
        y0, x0, y1, x1 = selected_row['box']; sl = np.s_[y0:y1, x0:x1]
        native_box = [y0*sy, x0*sx, y1*sy, x1*sx]
        low_known = np.isfinite(depth) & (depth > 0); truth = low_known & (depth < 2)
        pred = scores[i, 2] >= .081
        t, p, k, f, u = truth[sl], pred[sl], low_known[sl], fraction[sl], unknown[sl]
        pos = f[t]; fp = p & ~t & k; fn = ~p & t
        counts = dict(near=int(t.sum()), minority=int((pos < .5).sum()), quarter=int((pos <= .25).sum()),
                      fp=int(fp.sum()), fp_any=int((fp & (f > 0)).sum()), fp_majority=int((fp & (f >= .5)).sum()),
                      fn=int(fn.sum()), fn_minority=int((fn & (f < .5)).sum()))
        assert counts['near'] == selected_row['info']['near']
        for key in totals: totals[key] += counts[key]
        low_support = connected_support(truth, [y0, x0, y1, x1])
        native_support = connected_support(near, native_box)
        result = dict(id=ident, zone=selected_row['zone'], joint_gate_failed=ident in failed,
            source_shape=[nh, nw], scale=[sy, sx], counts=counts,
            positive_occupancy=dict(min=float(pos.min()), median=float(np.median(pos)), mean=float(pos.mean()), max=float(pos.max())),
            positive_unknown_fraction_max=float(u[t].max()),
            native_near_pixels_in_zone=native_support['zone_pixels'],
            near_area_input_pixel_equivalent=float(f.sum()),
            negative_labels_with_some_near=int(((~t)&k&(f>0)).sum()),
            negative_labels_with_majority_near=int(((~t)&k&(f>=.5)).sum()),
            error_occupancy={name:occupancy_summary(mask,f,u) for name,mask in [('tp',p&t),('fp',fp),('fn',fn)]},
            low_support=low_support, native_support=native_support,
            prepared_rgb_depth_exact=True)
        results.append(result)
        # Preserve diagnostic arrays for independent reductions, no modified input data.
        np.savez_compressed(OUT/f'{i:02d}-support.npz', fraction=f, unknown=u, truth=t, known=k, prediction=p)
        # Context shows actual support beyond the box; selected box is evaluator-only.
        cy0, cx0, cy1, cx1 = max(0,y0-12), max(0,x0-12), min(h,y1+12), min(w,x1+12)
        context = native_rgb[cy0*sy:cy1*sy,cx0*sx:cx1*sx].copy()
        cv2.rectangle(context, ((x0-cx0)*sx,(y0-cy0)*sy), ((x1-cx0)*sx-1,(y1-cy0)*sy-1), (255,210,0), 1)
        crop = native_rgb[y0*sy:y1*sy,x0*sx:x1*sx]
        native_gt = crop.copy(); nm = near[y0*sy:y1*sy,x0*sx:x1*sx]
        contours, _ = cv2.findContours(nm.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(native_gt, contours, -1, (0,255,70), 1)
        occupancy = np.stack([np.zeros_like(f),f,np.zeros_like(f)],-1)*255
        overlay = rgb[sl].copy()
        for mask,color in [(p&t,[30,220,80]),(fp,[255,60,60]),(fn,[40,100,255])]:
            overlay[mask] = .25*overlay[mask]+.75*np.array(color)
        cols = [context, native_gt, rgb[sl], occupancy.astype('uint8'), overlay]
        cols = [cv2.resize(x,(240,240),interpolation=cv2.INTER_NEAREST) for x in cols]
        header = np.full((52,1200,3),245,np.uint8)
        cv2.putText(header, f'{i:02d} {ident} z{selected_row["zone"]} {"FAIL" if ident in failed else "PASS"} near={counts["near"]} minority={counts["minority"]} FP={counts["fp"]}', (4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        cv2.putText(header, 'native context | native RGB + near contour | prepared RGB | footprint near fraction | joint: green TP/red FP/blue FN', (4,40),cv2.FONT_HERSHEY_SIMPLEX,.44,(20,20,20),1)
        panel = np.concatenate([header,np.concatenate(cols,axis=1)])
        panels.append(panel)
        cv2.imwrite(str(OUT/f'{i:02d}-native.jpg'),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        print(ident, json.dumps(counts), 'native_outside', native_support['all_extend_outside'], flush=True)
    for page in range(4):
        cv2.imwrite(str(OUT/f'contact-{page+1}.jpg'), cv2.cvtColor(np.concatenate(panels[page*4:page*4+4]),cv2.COLOR_RGB2BGR))
    write('source-hashes.json', source_hashes)
    write('results.json', dict(status='COMPLETE', frames=16, validation_frames=0, test_frames=0, totals=totals,
        all_native_components_extend_outside_zones=sum(r['native_support']['all_extend_outside'] for r in results),
        all_low_components_extend_outside_zones=sum(r['low_support']['all_extend_outside'] for r in results),
        rows=results, seconds=time.perf_counter()-start))
    print('AUDIT_COMPLETE', json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
