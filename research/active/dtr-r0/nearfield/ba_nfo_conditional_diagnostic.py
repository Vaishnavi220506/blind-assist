"""Consumed Development, frozen-model conditional ranking and spatial audit.

No training/calibration. GT defines an evaluator-only subgroup, not a router.
Exact distinct-score thresholds keep tied pixels together.
"""
import json
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ba_nfo_matched as m
from ba_nfo_rgb_control import RGBOnly

OUT = m.ROOT / 'artifacts.local/work/ba-nfo-conditional-20260919'
HYBRID = m.ROOT / 'artifacts.local/work/ba-nfo-hybrid-20260919'
RGB = m.ROOT / 'artifacts.local/work/ba-nfo-rgb-control-20260919'
ARMS = ['depth', 'nfo', 'rgb', 'hybrid']
LABELS = ['Depth RGB+ToF', 'NFO RGB+ToF', 'RGB-only NFO', 'Hybrid NFO+depth']


def curve(score, truth):
    order = np.argsort(-score, kind='stable')
    s, y = score[order], truth[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.r_[0, np.cumsum(y, dtype=np.int64)[ends]]
    fp = np.r_[0, ends+1-tp[1:]]
    p, n = int(y.sum()), int((~y).sum())
    recall = tp/p
    precision = np.divide(tp, tp+fp, out=np.ones(len(tp)), where=tp+fp > 0)
    return dict(threshold=np.r_[np.inf, s[ends]], tp=tp, fp=fp,
                recall=recall, precision=precision, iou=tp/(p+fp), fpr=fp/n)


def ap(c):
    return float(np.sum(np.diff(c['recall'])*c['precision'][1:]))


def describe(x):
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
                quantiles=dict(zip(['p05', 'p25', 'p50', 'p75', 'p95'],
                                   np.quantile(x, [.05, .25, .5, .75, .95]).tolist())),
                fraction_positive=float((x > 0).mean()))


def at_recall(c, recall):
    i = int(np.searchsorted(c['recall'], recall, side='left'))
    return {k: float(c[k][i]) for k in ['threshold', 'recall', 'precision', 'iou', 'fp', 'fpr']}


def distance(mask):
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    start = time.perf_counter()
    paths = dict(depth=m.OUT/'trained-depth.pt', nfo=m.OUT/'trained-nfo.pt',
                 rgb=RGB/'rgb-only.pt', hybrid=HYBRID/'trained-nfo.pt')
    cuts = {a: x['cutoff'] for a, x in json.loads((m.OUT/'calibration.json').read_text()).items()}
    cuts.update(rgb=json.loads((RGB/'calibration.json').read_text())['cutoff'],
                hybrid=json.loads((HYBRID/'calibration.json').read_text())['nfo']['cutoff'])
    rows = [r for r in json.loads((m.OUT/'manifest.json').read_text()) if r['split'] == 'test']
    protocol = dict(scope='Consumed synthetic Development; no training or cutoff selection',
        threshold_m=2, subgroup='finite public return >=2m; 0<GT near/known<=0.20; GT near depth<2m',
        unknown='exclude nonfinite truth, never count as far', frames=len(rows), cuts=cuts,
        score='original probabilities(): sigmoid then ordinal cummax for NFO; monotone depth score',
        ranking='all distinct float32 scores, ties atomic; pixel micro curves plus scene AP',
        spatial='Euclidean pixels at 192x256; distance to any GT near and to rescued subgroup TP; negative availability normalized',
        examples='three largest rescued TP frames, then two largest added FP frames without rescues',
        checkpoint_hashes={a:m.sha(p) for a,p in paths.items()}, manifest_sha256=m.sha(m.OUT/'manifest.json'),
        code_sha256=m.sha(__file__), backend='torch-cuda', device=torch.cuda.get_device_name(),
        cpu_analysis='TASK_NOT_GPU_SUITABLE: small sorted subgroup vectors, tables and rendering',
        placement_evidence=str(m.OUT/'backend.json'),
        decision='A: ranking dominance supports readout diagnosis; B: no ranking gain closes this auxiliary; C: crossing tradeoff; none proves a new branch works')
    m.write(OUT/'protocol.json', protocol)
    models = m.load_models('cuda')
    for arm, net in [('rgb', RGBOnly()), ('hybrid', m.Net('nfo'))]:
        payload = torch.load(paths[arm], map_location='cpu', weights_only=False)
        net.load_state_dict(payload['state_dict'])
        models[arm] = net.cuda().eval()
    vectors = defaultdict(list)
    totals = defaultdict(lambda: np.zeros(4, np.int64))
    zones, frames, previews = [], [], []
    all_delta = defaultdict(lambda: np.zeros(2, np.float64))
    unknown = 0
    with torch.inference_mode():
        for offset in range(0, len(rows), 24):
            batch = rows[offset:offset+24]
            arrays = []
            for r in batch:
                path = m.OLD/r['prepared']
                assert m.sha(path) == r['sha256'], r['id']
                with np.load(path) as data:
                    arrays.append({k:data[k].copy() for k in ['rgb', 'depth', 'values', 'boxes']})
            rgb = torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
            z = torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
            scores = {a:m.probabilities(net(rgb,z), net.arm).cpu().numpy()[:,2] for a,net in models.items()}
            for j, (r, a) in enumerate(zip(batch, arrays)):
                known, truth, mixed, small = m.masks(a['depth'], a['boxes'], 2.)
                unknown += int((~known).sum())
                mask = np.zeros(known.shape, bool)
                local_boxes = []
                for zi, (y0,x0,y1,x1) in enumerate(a['boxes']):
                    k, t = known[y0:y1,x0:x1], truth[y0:y1,x0:x1]
                    if np.isfinite(a['values'][zi]) and a['values'][zi]>=2 and 0<t.sum()<=.2*k.sum():
                        mask[y0:y1,x0:x1] = k
                        local_boxes.append((zi,y0,x0,y1,x1))
                pred = {arm:scores[arm][j]>=cuts[arm] for arm in ARMS}
                for arm in ARMS:
                    for domain, dm in [('full',known), ('mixed',mixed), ('small_foreground',small)]:
                        totals[arm,domain] += m.counts(pred[arm], truth, dm)
                delta = scores['hybrid'][j]-scores['nfo'][j]
                for label, dm in [('positive',truth), ('negative',known&~truth)]:
                    all_delta[label] += [delta[dm].sum(dtype=np.float64), dm.sum()]
                if not mask.any():
                    continue
                pos, neg = mask&truth, mask&~truth
                rescue = pos&~pred['nfo']&pred['hybrid']
                lost = pos&pred['nfo']&~pred['hybrid']
                added = neg&~pred['nfo']&pred['hybrid']
                removed = neg&pred['nfo']&~pred['hybrid']
                dnear, drescue = distance(truth), distance(rescue)
                # Connected near predictions with at least one rescued TP, eight-connected.
                _, labels = cv2.connectedComponents((pred['hybrid']&known).astype(np.uint8), connectivity=8)
                rescue_labels = np.unique(labels[rescue])
                connected = np.isin(labels, rescue_labels[rescue_labels>0])
                for arm in ARMS:
                    vectors[arm].append(scores[arm][j][mask])
                for name, value in dict(truth=truth, frame=np.full(mask.shape,offset+j),
                        distance_near=dnear, distance_rescue=drescue, connected=connected,
                        rescued=rescue, lost=lost, added=added, removed=removed).items():
                    vectors[name].append(value[mask])
                frame = dict(id=r['id'], scene=r['scene'], index=offset+j, zones=len(local_boxes),
                             rescued=int(rescue.sum()), lost=int(lost.sum()), added=int(added.sum()), removed=int(removed.sum()))
                frames.append(frame)
                for zi,y0,x0,y1,x1 in local_boxes:
                    sl = np.s_[y0:y1,x0:x1]
                    zones.append(dict(id=r['id'], scene=r['scene'], zone=int(zi),
                        positive=int(pos[sl].sum()), negative=int(neg[sl].sum()),
                        rescued=int(rescue[sl].sum()), added=int(added[sl].sum()),
                        positive_delta=float(delta[sl][pos[sl]].mean()), negative_delta=float(delta[sl][neg[sl]].mean())))
                previews.append((frame, a['rgb'], truth, known, mask, scores['nfo'][j].copy(),
                                 scores['hybrid'][j].copy(), rescue, added, lost, removed))
            if offset%120 == 0:
                print('CONDITIONAL_INFERENCE', offset, flush=True)
    v = {k:np.concatenate(x) for k,x in vectors.items()}
    assert np.isfinite(np.stack([v[a] for a in ARMS])).all()
    original = json.loads((m.OUT/'results.json').read_text())
    hybrid = json.loads((HYBRID/'results.json').read_text())
    rgb_result = json.loads((RGB/'results.json').read_text())
    for (arm,domain), c in totals.items():
        if arm in ['depth','nfo']:
            expected = original['metrics'][arm]['2.0'][domain]
        elif arm == 'hybrid':
            expected = hybrid['metrics'][arm]['2.0'][domain]
        else:
            expected = rgb_result['totals']['2.0']['small' if domain=='small_foreground' else domain]
        assert c.tolist() == [expected[k] for k in ['tp','fp','fn','tn']], (arm,domain,c,expected)
    assert unknown == hybrid['test_unknown_pixels']
    fixed = {}
    for arm in ARMS:
        fixed[arm] = m.metrics(m.counts(v[arm]>=cuts[arm], v['truth'], np.ones(len(v['truth']),bool)))
        expected = rgb_result['small_2m']['far'] if arm=='rgb' else hybrid['small_2m'][arm]['far']
        assert all(fixed[arm][k] == expected[k] for k in ['tp','fp','fn','tn']), arm
    paired = {k:int(v[k].sum()) for k in ['rescued','lost','added','removed']}
    assert list(paired.values()) == [1711,274,36741,5912], paired
    np.savez_compressed(OUT/'subgroup-scores.npz', **v)
    m.write(OUT/'frames.json', frames)
    m.write(OUT/'zones.json', zones)
    curves = {a:curve(v[a],v['truth']) for a in ARMS}
    np.savez_compressed(OUT/'exact-curves.npz', **{a+'_'+k:x for a,c in curves.items() for k,x in c.items()})
    matched = {str(r):{a:at_recall(c,r) for a,c in curves.items()} for r in [.1,.3,.5,.696119445,.8,.9,.95]}
    delta = v['hybrid']-v['nfo']
    shifts = {name:describe(delta[dm]) for name,dm in [('positive',v['truth']),('negative',~v['truth'])]}
    logit = lambda x: np.log(np.clip(x,1e-7,1-1e-7)/(1-np.clip(x,1e-7,1-1e-7)))
    ld = logit(v['hybrid'])-logit(v['nfo'])
    logit_shifts = {name:describe(ld[dm]) for name,dm in [('positive',v['truth']),('negative',~v['truth'])]}
    spatial = {}
    eligible_neg = (~v['truth'])&(v['nfo']<cuts['nfo'])
    for target in ['near','rescue']:
        spatial[target] = []
        previous = -1
        for radius in [2,4,8,16,32,float('inf')]:
            dm = (v['distance_'+target]>previous)&(v['distance_'+target]<=radius)
            available = int((dm&eligible_neg).sum())
            added = int((dm&v['added']).sum())
            spatial[target].append(dict(distance_bin=f'({previous},{radius}]', available_negative=available,
                added_fp=added, added_fraction=added/paired['added'],
                flip_rate=added/available if available else None))
            previous = radius
    scene_results = []
    for scene in sorted({f['scene'] for f in frames}):
        ids = [f['index'] for f in frames if f['scene']==scene]
        dm = np.isin(v['frame'],ids)
        scene_results.append(dict(scene=scene,pixels=int(dm.sum()),positive=int(v['truth'][dm].sum()),
            ap={a:ap(curve(v[a][dm],v['truth'][dm])) for a in ARMS}))
    budgets = [0,100,1000,5000,10000,65965,96794]
    low_fp = {str(b):{a:float(c['recall'][c['fp']<=b].max()) for a,c in curves.items()} for b in budgets}
    result = dict(fixed=fixed,paired=paired,positive_pixels=int(v['truth'].sum()),negative_pixels=int((~v['truth']).sum()),
        frames=len(frames),scenes=len(scene_results),zones=len(zones),unknown_full_test=unknown,
        average_precision={a:ap(c) for a,c in curves.items()}, matched_recall=matched, recall_at_fp_budget=low_fp,
        shifts=shifts,logit_shifts=logit_shifts,
        full_image_mean_delta={k:float(x[0]/x[1]) for k,x in all_delta.items()}, spatial=spatial,
        added_fp_in_rescue_component=int((v['added']&v['connected']).sum()),
        added_fp_in_rescue_zone=sum(z['added'] for z in zones if z['rescued']>0),
        added_fp_in_rescue_frame=sum(f['added'] for f in frames if f['rescued']>0),
        scene_ap=scene_results,original_counts_exact=True,seconds=time.perf_counter()-start)
    m.write(OUT/'results.json', result)
    plot(v, curves, fixed, shifts, previews)
    m.write(OUT/'receipt.json',dict(status='PASS', protocol_sha256=m.sha(OUT/'protocol.json'),
        backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,
        preserved={p.name:m.sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['receipt.json','completion.json']}))
    print(json.dumps({k:result[k] for k in ['average_precision','shifts','paired','recall_at_fp_budget','spatial']},indent=2),flush=True)


def plot_curves(curves, fixed):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,3,figsize=(16,4.8))
    for arm,label in zip(ARMS,LABELS):
        c = curves[arm]
        for ax,metric in zip(axes,['precision','iou','fp']):
            # Precision at the no-prediction endpoint is undefined, so hide it.
            start = 1 if metric == 'precision' else 0
            line, = ax.plot(c['recall'][start:],c[metric][start:],label=label,lw=1.6)
            ax.scatter(fixed[arm]['recall'],fixed[arm][metric],color=line.get_color(),s=40,zorder=4)
    for ax,title in zip(axes,['Precision','IoU','False-positive pixels']):
        ax.set(xlabel='Recall',ylabel=title,xlim=(0,1))
        ax.grid(alpha=.2)
    axes[2].set_yscale('symlog',linthresh=100)
    axes[2].set_ylim(bottom=0)
    axes[0].set_ylim(0,.25)
    axes[0].legend(fontsize=9)
    fig.suptitle('2m | far-return zones with 0 < near area <= 20% | consumed Development\nDots: original cutoffs; PR zoom 0-25% (exact full curves saved); diagnostic thresholds only',fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/'conditional-curves.png',dpi=170)
    fig.savefig(OUT/'conditional-curves.pdf')
    plt.close(fig)


def plot(v, curves, fixed, shifts, previews):
    plot_curves(curves, fixed)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    delta = v['hybrid']-v['nfo']
    for name,dm,color in [('Positive',v['truth'],'#008b70'),('Negative',~v['truth'],'#cd493d')]:
        axes[0].hist(delta[dm],bins=np.linspace(-1,1,81),density=True,histtype='step',lw=2,label=name,color=color)
        for ax,arm in zip(axes[1:],['nfo','hybrid']):
            ax.hist(v[arm][dm],bins=np.linspace(0,1,61),density=True,histtype='step',lw=2,label=name,color=color)
    for ax,title in zip(axes,['Hybrid score - NFO score','NFO score','Hybrid score']):
        ax.set(xlabel=title,ylabel='Density');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Paired conditional score shifts (scores are not calibrated probabilities)')
    fig.tight_layout();fig.savefig(OUT/'score-shifts.png',dpi=170);plt.close(fig)
    selected = sorted(previews,key=lambda p:(-p[0]['rescued'],p[0]['id']))[:3]
    selected += sorted([p for p in previews if p[0]['rescued']==0],key=lambda p:(-p[0]['added'],p[0]['id']))[:2]
    fig, axes = plt.subplots(len(selected),5,figsize=(16,3*len(selected)),squeeze=False)
    for rr,p in enumerate(selected):
        f,rgb,truth,known,mask,ns,hs,rescue,added,lost,removed = p
        gt = np.zeros_like(rgb);gt[truth]=[0,190,130];gt[~known]=[130,130,130]
        changes = (rgb*.25).astype(np.uint8)
        for dm,col in [(rescue,[0,255,100]),(added,[255,50,45]),(lost,[40,120,255]),(removed,[255,210,0])]:
            changes[dm]=col
        for ax,img,title in zip(axes[rr],[rgb,gt,ns,hs,changes],['RGB','GT near / subgroup outline','NFO score','Hybrid score','Rescue green / added FP red']):
            ax.imshow(img,**({'vmin':0,'vmax':1,'cmap':'magma'} if img.ndim==2 else {}))
            ax.contour(mask.astype(float),levels=[.5],colors='cyan',linewidths=.4)
            ax.set_title(title,fontsize=10);ax.axis('off')
        axes[rr,0].set_title(f"{f['id']}\nrescued={f['rescued']}, added FP={f['added']}",fontsize=8)
    fig.suptitle('Identity-selected diagnostic examples | cyan: evaluated subgroup\nBlue: lost TP; yellow: removed FP. Selection is illustrative, not independent evidence.')
    fig.tight_layout();fig.savefig(OUT/'spatial-examples.jpg',dpi=140);plt.close(fig)
    m.write(OUT/'preview-selection.json',[p[0] for p in selected])


def supplement():
    """Reanalyze retained subgroup scores without rerunning any model."""
    with np.load(OUT/'subgroup-scores.npz') as data:
        v = {k:data[k] for k in data.files}
    result = json.loads((OUT/'results.json').read_text())
    curves = {a:curve(v[a],v['truth']) for a in ARMS}
    for c in curves.values():
        assert c['tp'][-1] == v['truth'].sum()
        assert c['fp'][-1] == (~v['truth']).sum()
        assert (np.diff(c['tp'])>=0).all() and (np.diff(c['fp'])>=0).all()
    cut_table = {}
    for arm in ['nfo','hybrid']:
        cut_table[arm] = {str(c):m.metrics(m.counts(v[arm]>=c,v['truth'],np.ones(len(v['truth']),bool))) for c in [.051,.081]}
    result['crossed_cutoffs_diagnostic_only'] = cut_table
    no_rescue = ~np.isfinite(v['distance_rescue'])
    result['added_fp_no_rescue_frame'] = int((v['added']&no_rescue).sum())
    result['added_fp_farther_than_32_in_rescue_frame'] = int((v['added']&~no_rescue&(v['distance_rescue']>32)).sum())
    scene_d = np.array([s['ap']['hybrid']-s['ap']['nfo'] for s in result['scene_ap']])
    result['scene_ap_delta_summary'] = dict(mean=float(scene_d.mean()),median=float(np.median(scene_d)),
        improved=int((scene_d>0).sum()),worsened=int((scene_d<0).sum()),total=len(scene_d))
    # Scene-cluster resampling preserves within-scene correlation. Sort once.
    frames = json.loads((OUT/'frames.json').read_text())
    scenes = sorted({f['scene'] for f in frames})
    mapping = {f['index']:scenes.index(f['scene']) for f in frames}
    scene_idx = np.array([mapping[int(i)] for i in v['frame']])
    sorted_data = {}
    for arm in ['nfo','hybrid']:
        order = np.argsort(-v[arm],kind='stable')
        s = v[arm][order]
        ends = np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
        sorted_data[arm] = (v['truth'][order],scene_idx[order],ends)
    rng = np.random.default_rng(190919)
    differences = []
    for _ in range(300):
        multiplicity = np.bincount(rng.integers(len(scenes),size=len(scenes)),minlength=len(scenes))
        aps = {}
        for arm,(y,si,ends) in sorted_data.items():
            weights = multiplicity[si]
            tp = np.cumsum(weights*y)[ends]
            total = np.cumsum(weights)[ends]
            precision = np.divide(tp,total,out=np.zeros(len(tp)),where=total>0)
            aps[arm] = float(np.sum(np.diff(np.r_[0,tp/tp[-1]])*precision))
        differences.append(aps['hybrid']-aps['nfo'])
    result['scene_bootstrap_ap_delta'] = dict(replicates=300,seed=190919,
        percentile_95=np.quantile(differences,[.025,.975]).tolist(),
        scope='53 scene clusters of consumed Development; descriptive uncertainty, not equivalence or fresh confirmation')
    np.save(OUT/'scene-bootstrap-ap-deltas.npy',np.array(differences))
    m.write(OUT/'results.json',result)
    plot_curves(curves,result['fixed'])
    m.write(OUT/'analysis-receipt.json',dict(source_inference_receipt_sha256=m.sha(OUT/'receipt.json'),
        final_code_sha256=m.sha(__file__), score_cache_sha256=m.sha(OUT/'subgroup-scores.npz'),
        results_sha256=m.sha(OUT/'results.json'),curve_png_sha256=m.sha(OUT/'conditional-curves.png'),
        reason='cache-only cutoff attribution, scene uncertainty, absent-rescue accounting and legible curve axes'))
    print(json.dumps({k:result[k] for k in ['crossed_cutoffs_diagnostic_only','scene_ap_delta_summary','scene_bootstrap_ap_delta','added_fp_no_rescue_frame','added_fp_farther_than_32_in_rescue_frame']},indent=2))


if __name__ == '__main__':
    try:
        if '--cache-only' not in sys.argv:
            run()
        supplement()
        m.write(OUT/'completion.json',dict(status='PASS'))
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        m.write(OUT/'completion.json',dict(status='FAIL',error=repr(exc)))
        raise
