"""Frozen ZJU-L5 near-surface diagnostic; no training or alert integration.

Prediction phase never reads HDF5 depth. Evaluation is a separate invocation
after hashes of all public predictions are sealed. Missing ToF is UNKNOWN.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/mz140-depthor-20260915'
SMOKE = ROOT/'artifacts.local/work/zju-depthor-smoke-20260918'
MONO = ROOT/'artifacts.local/work/mz164-camera-metric-prior-20260916'
THRESHOLDS = (1., 1.5, 2., 3.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def rect(fr, shape):
    y0, x0, y1, x1 = map(int, fr)
    h, w = shape
    return slice(max(0, y0), min(h, max(0, y1))), slice(max(0, x0), min(w, max(0, x1)))


def expand_zones(hist, fr, mask, shape=(480, 640)):
    """Half-open official projected rectangles; overlap takes nearest return.

    This conservative union is a simple baseline, not physical surface truth.
    """
    out = np.full(shape, np.nan, np.float32)
    for value, box, ok in zip(hist[:, 0], fr, mask):
        if not ok or not np.isfinite(value) or not .001 < value < 10:
            continue
        s = rect(box, shape)
        out[s] = np.fmin(out[s], value)
    return out


def mixed_mask(gt, fr, valid):
    out = np.zeros(gt.shape, bool)
    count = 0
    for box in fr:
        s = rect(box, gt.shape)
        v = valid[s]
        if v.size == 0 or v.mean() < .8 or v.sum() < 64:
            continue
        p20, p80 = np.percentile(gt[s][v], [20, 80])
        if p20 < 2 and p80-p20 > 1:
            out[s] = True
            count += 1
    return out & valid, count


def boundary_mask(gt, valid, radius):
    edge = np.zeros(gt.shape, np.uint8)
    dx = valid[:, 1:] & valid[:, :-1] & (np.abs(gt[:, 1:]-gt[:, :-1]) > .5)
    dy = valid[1:] & valid[:-1] & (np.abs(gt[1:]-gt[:-1]) > .5)
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:] |= dy
    edge[:-1] |= dy
    return cv2.dilate(edge, np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool) & valid


def counts(pred, gt, domain, threshold):
    known = np.isfinite(pred) & (pred > .001) & (pred < 10)
    y = gt < threshold
    p = pred < threshold
    use = domain & known
    return dict(tp=int((use & p & y).sum()), fp=int((use & p & ~y).sum()),
                fn=int((use & ~p & y).sum()), tn=int((use & ~p & ~y).sum()),
                unknown_positive=int((domain & ~known & y).sum()),
                unknown_negative=int((domain & ~known & ~y).sum()),
                domain_pixels=int(domain.sum()), evaluated=int(use.sum()))


def rates(c):
    def div(a, b): return a/b if b else None
    return dict(**c, precision=div(c['tp'], c['tp']+c['fp']),
                recall=div(c['tp'], c['tp']+c['fn']),
                iou=div(c['tp'], c['tp']+c['fp']+c['fn']),
                coverage=div(c['evaluated'], c['domain_pixels']),
                positive_detection_fraction=div(c['tp'], c['tp']+c['fn']+c['unknown_positive']))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(SMOKE/'ZJUL5.zip') as z:
        names = z.namelist()
        meta = next(n for n in names if n.endswith('/data.json') or n == 'data.json')
        entries = json.loads(z.read(meta))['test']
        groups = {}
        for e in entries:
            groups.setdefault(e['filename'].split('/')[0], []).append(e)
        selected = []
        for scene, rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda e: e['filename'])
            ids = np.unique(np.linspace(0, len(rows)-1, min(20, len(rows))).astype(int))
            selected.extend(rows[i] for i in ids)
        assert 100 <= len(selected) <= 300
        for e in selected:
            p = out/'inputs'/e['filename']
            assert p.resolve().is_relative_to((out/'inputs').resolve())
            member = next(n for n in names if n.endswith('/'+e['filename']) or n == e['filename'])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(z.read(member))
            e['sha256'] = sha(p)
    write(out/'protocol.json', dict(id='BA_DEPTH_PROBE_ZJU_20260918',
        phase='EXPLORE_CONSUMED_PUBLIC_DIAGNOSTIC', selected=selected,
        sampling='20 evenly spaced lexical entries per official test scene; no outcome selection',
        question='Does frozen RGB+ToF improve mixed-zone near-surface localization against raw expansion?',
        arms=['raw_zone', 'unidepth_v2_rgb_only', 'depthor', 'reference_identity'],
        mono='Official metric UniDepthV2 ViT-S; estimated camera; no GT/ToF scale fit; different backbone so not a fusion-only causal ablation',
        primary='Mixed zones, reference-valid AND raw-known common pixels, depth<2m IoU; paired scene summaries',
        mixed='P20<2m and P80-P20>1m, >=80% valid reference and >=64 valid pixels, union without duplicate pixels',
        secondary='All thresholds 1/1.5/2/3m on common and full domains; boundary >.5m valid-neighbor jump at +/-5 and +/-10px',
        unknown='Never filled as far; report abstentions and domain coverage separately',
        corridor='NOT_EVALUABLE: no body pose, camera-to-body transform or confirmed intrinsics in samples; no invented image corridor',
        interpretation='Descriptive geometric diagnostic only; mixed-zone selection misses very thin <20%-area foreground; real reference imperfect',
        go='Mixed common C IoU>A IoU and C recall>=A recall, with improvement in majority of evaluable scenes; descriptive only',
        stop='One fixed inference per arm; no fitting, thresholds, weight changes, new capture or automatic successor',
        no_gain='Retain scoped negative; do not automatically reopen consumed MZ140/MZ142 training',
        checkpoint_sha256=sha(OLD/'depthor-zju-small.pt'), mono_checkpoint_sha256=sha(MONO/'model.safetensors'),
        archive_sha256=sha(SMOKE/'ZJUL5.zip'), runner_sha256=sha(__file__),
        prior='MZ140/MZ142 UE geometry failures retained; this is real-data diagnostic, not fresh confirmation'))
    print('FROZEN', len(selected), flush=True)


def predict(out):
    import torch
    from mz140_depthor import load_model
    from mz164_metric_prior import load_model as load_mono
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    protocol = json.loads((out/'protocol.json').read_text())
    assert sha(__file__) == protocol['runner_sha256']
    assert sha(OLD/'depthor-zju-small.pt') == protocol['checkpoint_sha256']
    assert not (out/'prediction-seal.json').exists()
    dest = out/'predictions'
    dest.mkdir(exist_ok=True)
    observations = []
    model, info = load_model(OLD/'upstream', OLD/'depthor-zju-small.pt', 'cuda')
    from src.utils.dataloader import dtof_to_sparse_depth
    with torch.inference_mode():
        for i, entry in enumerate(protocol['selected']):
            p = out/'inputs'/entry['filename']
            assert sha(p) == entry['sha256']
            with h5py.File(p) as f:
                rgb, hist, fr, mask = [np.array(f[k]) for k in ('rgb', 'hist_data', 'fr', 'mask')]
            sparse = dtof_to_sparse_depth(torch.tensor(hist).float(), torch.tensor(fr), torch.tensor(mask))
            image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].cuda()/255.
            torch.cuda.synchronize()
            t = time.perf_counter()
            pred = model(dict(image=image, sparse_depth=sparse[None].cuda()))[1]
            torch.cuda.synchronize()
            ms = (time.perf_counter()-t)*1000
            pred = pred[0, 0].cpu().numpy()
            assert pred.shape == (480, 640) and np.isfinite(pred).all()
            np.savez_compressed(dest/f'{i:03d}-depthor.npz', depth=np.clip(pred, .001, 10))
            np.savez_compressed(dest/f'{i:03d}-raw.npz', depth=expand_zones(hist, fr, mask))
            observations.append(dict(filename=entry['filename'],depthor_ms=ms,sparse_points=int((sparse>0).sum())))
            if i % 20 == 0: print('DEPTHOR', i, flush=True)
    del model
    torch.cuda.empty_cache()
    mono, mono_info = load_mono(MONO, 'cuda')
    with torch.inference_mode():
        for i, entry in enumerate(protocol['selected']):
            with h5py.File(out/'inputs'/entry['filename']) as f:
                rgb = np.array(f['rgb'])
            image = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
            torch.cuda.synchronize()
            t = time.perf_counter()
            pred = mono.infer(image, camera=None, normalize=True)['depth']
            torch.cuda.synchronize()
            observations[i]['mono_ms'] = (time.perf_counter()-t)*1000
            pred = pred[0, 0].float().cpu().numpy()
            assert pred.shape == (480, 640) and np.isfinite(pred).all()
            np.savez_compressed(dest/f'{i:03d}-mono.npz', depth=pred)
            if i % 20 == 0: print('MONO', i, flush=True)
    write(out/'runtime.json', dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
        depthor=info,mono=mono_info,observations=observations,
        backend='CUDA; reuse established GPU advantage; scalar evaluation on CPU TASK_NOT_GPU_SUITABLE'))
    write(out/'prediction-seal.json', dict(protocol_sha256=sha(out/'protocol.json'),
        files={p.name:sha(p) for p in sorted(dest.glob('*.npz'))}))
    print('SEALED',len(observations),flush=True)


def evaluate(out):
    protocol = json.loads((out/'protocol.json').read_text())
    seal = json.loads((out/'prediction-seal.json').read_text())
    assert seal['protocol_sha256'] == sha(out/'protocol.json')
    for name, digest in seal['files'].items():
        assert sha(out/'predictions'/name) == digest
    all_rows, boundary_rows, cases = [], [], []
    for i, entry in enumerate(protocol['selected']):
        scene = entry['filename'].split('/')[0]
        with h5py.File(out/'inputs'/entry['filename']) as f:
            gt, fr = np.array(f['depth']), np.array(f['fr'])
        valid = np.isfinite(gt) & (gt > .001) & (gt < 10)
        arms = {k:np.load(out/'predictions'/f'{i:03d}-{k}.npz')['depth'] for k in ('raw','mono','depthor')}
        arms['reference'] = gt
        common = valid & np.isfinite(arms['raw'])
        mixed, nz = mixed_mask(gt, fr, valid)
        cases.append(dict(filename=entry['filename'],scene=scene,mixed_zones=nz,mixed_pixels=int(mixed.sum()),
                          reference_pixels=int(valid.sum()),raw_common_pixels=int(common.sum())))
        domains = dict(full=valid,common=common,mixed_full=mixed,mixed_common=mixed & common)
        for domain, m in domains.items():
            for arm, pred in arms.items():
                for th in THRESHOLDS:
                    all_rows.append(dict(scene=scene,filename=entry['filename'],domain=domain,arm=arm,threshold=th,**counts(pred,gt,m,th)))
        for radius in (5,10):
            band = boundary_mask(gt,valid,radius)
            for dom, m in dict(full=band,common=band & common).items():
                for arm, pred in arms.items():
                    usable=m & np.isfinite(pred) & (pred>.001) & (pred<10)
                    boundary_rows.append(dict(scene=scene,arm=arm,radius=radius,domain=dom,pixels=int(usable.sum()),
                        domain_pixels=int(m.sum()),absolute_error_sum=float(np.abs(pred[usable]-gt[usable]).sum(dtype=np.float64))))
    def aggregate(rows, fields, sums, transform):
        grouped={}
        for r in rows:
            key=tuple(r[f] for f in fields)
            v=grouped.setdefault(key,{s:0 for s in sums})
            for s in sums:v[s]+=r[s]
        return [dict(zip(fields,k),**transform(v)) for k,v in grouped.items()]
    sums=['tp','fp','fn','tn','unknown_positive','unknown_negative','domain_pixels','evaluated']
    global_rows=aggregate(all_rows,['domain','arm','threshold'],sums,rates)
    scene_rows=aggregate(all_rows,['scene','domain','arm','threshold'],sums,rates)
    def br(v):return dict(**v,mae_m=v['absolute_error_sum']/v['pixels'] if v['pixels'] else None,
                         coverage=v['pixels']/v['domain_pixels'] if v['domain_pixels'] else None)
    b=aggregate(boundary_rows,['arm','radius','domain'],['pixels','domain_pixels','absolute_error_sum'],br)
    primary=[r for r in global_rows if r['domain']=='mixed_common' and r['threshold']==2]
    pa={r['arm']:r for r in primary}
    eligible=0;wins=0;paired=[]
    for scene in sorted({r['scene'] for r in scene_rows}):
        rr={r['arm']:r for r in scene_rows if r['scene']==scene and r['domain']=='mixed_common' and r['threshold']==2}
        if rr['raw']['iou'] is not None and rr['depthor']['iou'] is not None:
            eligible+=1;wins+=int(rr['depthor']['iou']>rr['raw']['iou'])
            paired.append(dict(scene=scene,raw_iou=rr['raw']['iou'],mono_iou=rr['mono']['iou'],depthor_iou=rr['depthor']['iou']))
    go=bool(eligible and pa['depthor']['iou']>pa['raw']['iou'] and pa['depthor']['recall']>=pa['raw']['recall'] and wins>eligible/2)
    write(out/'per-frame.json',dict(cases=cases,counts=all_rows,boundary=boundary_rows))
    write(out/'results.json',dict(status='GEOMETRY_COMPONENT_ONLY' if go else 'NO_JOINT_MIXED_IOU_RECALL_GAIN',
        frames=len(cases),scenes=len({r['scene'] for r in cases}),mixed_zones=sum(r['mixed_zones'] for r in cases),
        frames_with_mixed=sum(r['mixed_zones']>0 for r in cases),primary=primary,paired=paired,scene_iou_wins=wins,
        eligible_scenes=eligible,occupancy=global_rows,scene_occupancy=scene_rows,boundary=b,
        corridor='NOT_EVALUABLE',precision_gt='Reference identity ceiling only, not independent truth validation',
        stop='No training or automatic successor; preserve prior consumed UE negatives'))
    print(json.dumps(dict(primary=primary,paired=paired,go=go)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','predict','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    globals()[args.mode](args.output)
