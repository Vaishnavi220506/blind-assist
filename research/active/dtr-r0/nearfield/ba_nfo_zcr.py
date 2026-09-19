"""One parameter-free 2m zone-median contrast on frozen NFO probabilities."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import ba_nfo_matched as m
from ba_nfo_conditional_diagnostic import curve, ap
from ba_nfo_zone_readout import public_boxes, load_base

BASE = m.OUT
OUT = m.ROOT / 'artifacts.local/work/ba-nfo-zcr-20260919'
CUT = .081


def residual(scores, boxes):
    """All pixels participate, including evaluator-UNKNOWN and missing returns.

    Scores are B,H,W public 2m probabilities. No labels or return gate enters.
    torch.quantile uses the mean of both central samples for even zone sizes.
    Outside calibrated coverage is NaN (no relative-background definition).
    """
    out = torch.full_like(scores, float('nan'))
    for y0, x0, y1, x1 in boxes:
        z = scores[:, y0:y1, x0:x1]
        median = torch.quantile(z.flatten(1), .5, dim=1)
        out[:, y0:y1, x0:x1] = z - median[:, None, None]
    return out


def prediction(scores, relative, cutoff):
    return np.where(np.isfinite(relative), relative >= cutoff, scores >= CUT)


def select_cutoff(scores, truth):
    """Original validation objective: best mixed IoU at >=95% recall.

    Exact distinct float32 residuals, ties atomic. A lower cutoff wins IoU ties.
    This is one global calibration, not a test-selected or per-zone threshold.
    """
    c = curve(scores, truth)
    eligible = np.flatnonzero(c['recall'] >= .95)
    best = eligible[c['iou'][eligible] == c['iou'][eligible].max()][-1]
    return dict(cutoff=float(c['threshold'][best]), recall=float(c['recall'][best]),
                iou=float(c['iou'][best]), tp=int(c['tp'][best]), fp=int(c['fp'][best]),
                positives=int(truth.sum()), negatives=int((~truth).sum())), c


def domains(a):
    known, truth, mixed, small = m.masks(a['depth'], a['boxes'], 2.)
    far = np.zeros_like(known); pure = np.zeros_like(known); inside = np.zeros_like(known)
    for zi, (y0, x0, y1, x1) in enumerate(a['boxes']):
        sl = np.s_[y0:y1, x0:x1]; inside[sl] = True
        if np.isfinite(a['values'][zi]) and a['values'][zi] >= 2:
            far[sl] = True
            if known[sl].any() and not truth[sl].any():
                pure[sl] = True
    return truth, dict(full=known, mixed=mixed, small_foreground=small,
                      far_small=far & small, pure_far=pure & known,
                      public_far=far & known, outside=known & ~inside)


@torch.inference_mode()
def batches(rows, model, timing):
    boxes = public_boxes()
    for offset in range(0, len(rows), 24):
        selected = rows[offset:offset+24]; arrays = []
        for row in selected:
            path = m.OLD / row['prepared']
            assert m.sha(path) == row['sha256'], row['id']
            with np.load(path) as a:
                arrays.append({k:a[k].copy() for k in ['rgb', 'depth', 'values', 'boxes']})
            np.testing.assert_array_equal(arrays[-1]['boxes'], boxes)
        rgb = torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
        assert rgb.is_contiguous(), 'Match original NCHW inference layout'
        zones = torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
        torch.cuda.synchronize(); begin = time.perf_counter()
        scores = m.probabilities(model(rgb, zones), 'nfo')[:, 2]
        torch.cuda.synchronize(); after_model = time.perf_counter()
        relative = residual(scores, boxes)
        torch.cuda.synchronize(); after_readout = time.perf_counter()
        timing.append(dict(frames=len(selected), inference_seconds=after_model-begin,
                           median_seconds=after_readout-after_model))
        scores, relative = scores.cpu().numpy(), relative.cpu().numpy()
        for row, a, s, r in zip(selected, arrays, scores, relative):
            yield row, a, s, r
        print('ZCR_INFERENCE', selected[0]['split'], offset+len(selected), flush=True)


def preview(items, cutoff):
    import cv2
    panels = []
    for a, scores, relative in items:
        im = a['rgb']; truth, dm = domains(a); known = dm['full']
        gt = im.copy(); gt[truth] = [40,220,100]; gt[~known] = [160,80,180]
        cols = [im, gt]
        for pred in [scores >= CUT, prediction(scores, relative, cutoff)]:
            v = im.copy()
            for mask, color in [(pred & truth, [40,220,100]),
                                (pred & ~truth & known, [255,70,60]),
                                (~pred & truth, [50,120,255])]:
                v[mask] = (.25*v[mask]+.75*np.array(color)).astype(np.uint8)
            v[~known] = [160,80,180]; cols.append(v)
        row = np.concatenate(cols, axis=1)
        header = np.full((30, row.shape[1], 3), 245, np.uint8)
        for j, label in enumerate(['RGB', 'GT <2m', 'Frozen NFO', 'ZCR (val cutoff)']):
            cv2.putText(header, label, (j*m.W+4,20), cv2.FONT_HERSHEY_SIMPLEX, .45, (20,20,20), 1)
        panels.extend([header, row])
    if panels:
        cv2.imwrite(str(OUT/'comparison.jpg'), cv2.cvtColor(np.concatenate(panels), cv2.COLOR_RGB2BGR))


def run(repair_layout=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert repair_layout and not (OUT/'layout-repair.json').exists() and not (OUT/'results.json').exists(), 'Refuse repeat evaluation/calibration'
        m.write(OUT/'layout-repair.json', dict(reason='Baseline exact-count assertion failed with noncontiguous NumPy transpose batch; restore original contiguous NCHW input',
            original_protocol=json.loads((OUT/'protocol.json').read_text()),
            original_calibration=json.loads((OUT/'calibration.json').read_text()),
            repaired_code_sha256=m.sha(__file__), mechanism_and_selection_rule_unchanged=True))
        (OUT/'validation-curve.npz').rename(OUT/'layout-failed-validation-curve.npz')
    torch.set_num_threads(4); assert torch.cuda.is_available()
    sys.path.insert(0, str(m.ROOT/'tools'))
    from research_backend import torch_observation
    manifest = json.loads((BASE/'manifest.json').read_text())
    protocol = dict(id='ba-nfo-zcr-20260919', scope='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        question='Can within-zone relative background subtraction recover small near support?',
        mechanism='Frozen NFO original sigmoid/cummax 2m scores minus each full public zone median',
        coverage='All 64 original calibrated zones, including missing returns; no return or GT gate. Outside zones keep original .081 decisions.',
        median='Mean of central two values if even size; all pixels including evaluator UNKNOWN',
        calibration='Original 500 validation frames only: maximize mixed IoU at recall>=.95 over exact distinct residuals; ties lower cutoff. Seal before test inference.',
        test='Original 500 training-unseen but consumed Development frames; no test-selected cutoff or retuning',
        targets=dict(far_small_recall=.75, far_small_iou='>= original exact IoU', mixed_recall=.945, pure_far_fp='<= original exact FP'),
        small_definition='Original far-return >=2m, evaluator 0<near/known<=20%, near depth<2m',
        unknown='Excluded only from evaluation/calibration labels; never background, never excluded from median',
        stop='One score-minus-median arm only; no MAD, fit, zone parameters, auxiliary, connectivity, temporal or histogram readout',
        manifest_sha256=m.sha(BASE/'manifest.json'), checkpoint_sha256=m.sha(BASE/'trained-nfo.pt'),
        code_sha256=m.sha(__file__), original_cutoff=CUT, threshold_m=2,
        backend='torch CUDA inference and quantile; reuse retained encoder placement',
        placement_evidence=str(BASE/'backend.json'),
        cpu_analysis='TASK_NOT_GPU_SUITABLE: scalar counts, exact calibration sorting, tables and rendering')
    if not repair_layout:
        m.write(OUT/'protocol.json', protocol)
    model = load_base().cuda().eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    timing = []; begin = time.perf_counter()
    vals = []; labels = []; val_base = np.zeros(4, np.int64)
    for row, a, score, rel in batches([r for r in manifest if r['split']=='val'], model, timing):
        truth, dm = domains(a)
        vals.append(rel[dm['mixed']]); labels.append(truth[dm['mixed']])
        val_base += m.counts(score >= CUT, truth, dm['mixed'])
    vs, vy = np.concatenate(vals), np.concatenate(labels)
    calibration, vc = select_cutoff(vs, vy)
    calibration['baseline'] = m.metrics(val_base)
    calibration['validation_only'] = True
    m.write(OUT/'calibration.json', calibration)
    np.savez_compressed(OUT/'validation-curve.npz', **vc)
    del vals, labels, vs, vy, vc
    cutoff = calibration['cutoff']; calibration_sha = m.sha(OUT/'calibration.json')
    print('ZCR_CUTOFF_SEALED', calibration, flush=True)

    totals = defaultdict(lambda:np.zeros(4,np.int64))
    paired = defaultdict(lambda:np.zeros(4,np.int64)); scenes = defaultdict(lambda:np.zeros(4,np.int64))
    vectors = defaultdict(list); frames = []; unknown = 0; outside_equal = True; selected_previews = []
    preview_ids = json.loads((BASE/'preview-selection.json').read_text())['small_area']
    for row, a, score, rel in batches([r for r in manifest if r['split']=='test'], model, timing):
        truth, dm = domains(a); unknown += int((~dm['full']).sum())
        predictions = dict(nfo=score>=CUT, zcr=prediction(score,rel,cutoff))
        fmetrics = {arm:{} for arm in predictions}
        for name, mask in dm.items():
            for arm, pred in predictions.items():
                c = m.counts(pred, truth, mask); totals[arm,name] += c
                scenes[arm,name,row['scene']] += c; fmetrics[arm][name] = m.metrics(c)
            old, new = predictions['nfo'], predictions['zcr']
            paired[name] += [(mask & truth & ~old & new).sum(), (mask & truth & old & ~new).sum(),
                             (mask & ~truth & ~old & new).sum(), (mask & ~truth & old & ~new).sum()]
        outside_equal &= bool(np.array_equal(predictions['nfo'][dm['outside']], predictions['zcr'][dm['outside']]))
        for name, v in [('nfo',score), ('zcr',rel), ('truth',truth)]:
            vectors[name].append(v[dm['far_small']])
        frames.append(dict(id=row['id'], scene=row['scene'], metrics=fmetrics))
        if row['id'] in preview_ids:
            selected_previews.append((a,score,rel))
    metrics = {arm:{name:m.metrics(c) for (a,name),c in totals.items() if a==arm} for arm in ['nfo','zcr']}
    expected = json.loads((m.ROOT/'artifacts.local/work/ba-nfo-zone-readout-20260919/results.json').read_text())['metrics']['test']['nfo']['2.0']
    m.write(OUT/'baseline-reproduction.json',dict(actual=metrics['nfo'],expected=expected))
    for name in ['full','mixed','small_foreground','far_small','pure_far','public_far']:
        for key in ['tp','fp','fn','tn']:
            assert metrics['nfo'][name][key] == expected[name][key], (name,key)
    assert m.sha(OUT/'calibration.json') == calibration_sha
    assert m.sha(BASE/'trained-nfo.pt') == protocol['checkpoint_sha256']
    for name, counts in paired.items():
        assert metrics['zcr'][name]['tp']-metrics['nfo'][name]['tp'] == counts[0]-counts[1]
        assert metrics['zcr'][name]['fp']-metrics['nfo'][name]['fp'] == counts[2]-counts[3]
    b, n = metrics['nfo'], metrics['zcr']
    gates = dict(small_recall_75=n['far_small']['recall']>=.75,
                 small_iou_retained=n['far_small']['iou']>=b['far_small']['iou'],
                 mixed_recall_945=n['mixed']['recall']>=.945,
                 pure_far_fp_retained=n['pure_far']['fp']<=b['pure_far']['fp'])
    v = {k:np.concatenate(x) for k,x in vectors.items()}
    np.savez_compressed(OUT/'subgroup-scores.npz', **v)
    curves = {a:curve(v[a],v['truth']) for a in ['nfo','zcr']}
    np.savez_compressed(OUT/'subgroup-curves.npz', **{a+'_'+k:x for a,c in curves.items() for k,x in c.items()})
    result = dict(metrics=metrics, gates=gates, pass_all=all(gates.values()), calibration=calibration,
        ap={a:ap(c) for a,c in curves.items()}, unknown_pixels=unknown,
        baseline_counts_exact=True, outside_predictions_exact=outside_equal,
        checkpoint_unchanged=True, calibration_sha256=calibration_sha,
        paired={name:dict(zip(['rescued_tp','lost_tp','added_fp','removed_fp'],map(int,c))) for name,c in paired.items()},
        scenes=[dict(arm=a, domain=d, scene=s, **m.metrics(c)) for (a,d,s),c in scenes.items()],
        device=torch_observation(output=next(model.parameters())).__dict__, timing=timing,
        total_seconds=time.perf_counter()-begin)
    m.write(OUT/'results.json', result); m.write(OUT/'frames.json', frames)
    preview(selected_previews, cutoff)
    m.write(OUT/'completion.json',dict(status='COMPLETE',pass_all=result['pass_all']))
    print('ZCR_RESULT', json.dumps(dict(gates=gates, metrics=metrics, ap=result['ap'])), flush=True)


@torch.inference_mode()
def infer(sample, output, calibration):
    torch.set_num_threads(4)
    cutoff = json.loads(Path(calibration).read_text())['cutoff']
    model = load_base().cuda().eval().requires_grad_(False)
    with np.load(sample) as a:
        rgb = torch.from_numpy(a['rgb'].transpose(2,0,1).copy()[None]).cuda()
        zones = torch.from_numpy(m.public_zones(a['values'])[None]).cuda()
    scores = m.probabilities(model(rgb,zones),'nfo')[:,2]
    rel = residual(scores,public_boxes())[0].cpu().numpy(); s = scores[0].cpu().numpy()
    np.savez_compressed(output,scores=s,residual=rel,mask=prediction(s,rel,cutoff),cutoff=cutoff)
    print('PUBLIC_ZCR_INFERENCE_PASS',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--sample'); p.add_argument('--output')
    p.add_argument('--calibration',default=str(OUT/'calibration.json'))
    p.add_argument('--repair-layout',action='store_true'); args = p.parse_args()
    if args.sample:
        infer(args.sample,args.output,args.calibration)
    else:
        run(args.repair_layout)
