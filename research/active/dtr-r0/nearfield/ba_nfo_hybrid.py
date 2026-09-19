"""One matched NFO plus training-only continuous-depth auxiliary experiment."""
import json
import math
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ba_nfo_matched as m

BASE = m.OUT
OUT = m.ROOT / 'artifacts.local/work/ba-nfo-hybrid-20260919'
BINS = ['0-5%', '5-10%', '10-20%', '20-50%', '50-100%']


class Hybrid(m.Net):
    def __init__(self):
        super().__init__('nfo')
        self.auxiliary_output = nn.Conv2d(16, 1, 1)

    def forward_train(self, rgb, zones):
        # Reuse the original forward exactly; the temporary hook exposes only
        # shared decoder features to the training-only auxiliary head.
        features = []
        hook = self.output.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            logits = self(rgb, zones)
        finally:
            hook.remove()
        return logits, self.auxiliary_output(features[0])


def initialize():
    m.seed()
    model = Hybrid().cuda()
    m.seed()
    baseline = m.Net('nfo').cuda()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    return model


def train(data):
    assert not (OUT / 'hybrid-training.pt').exists()
    model = initialize()
    ids = [i for i, r in enumerate(data.rows) if r['split'] == 'train']
    # Identity order was frozen by the original manifest; no subgroup selection.
    scale_ids = ids[:32]
    with torch.no_grad():
        rgb, z, d = data.batch(scale_ids, 'cuda')
        nfo, depth = model.forward_train(rgb, z)
        ln = float(m.loss_fn(nfo, d, 'nfo'))
        ld = float(m.loss_fn(depth, d, 'depth'))
    weight = ln / ld
    assert math.isfinite(weight) and weight > 0
    m.write(OUT / 'loss-scale.json', dict(nfo=ln, depth=ld, weight=weight,
        ids=[data.rows[i]['id'] for i in scale_ids], rule='initial NFO loss / initial depth loss; frozen before any update'))
    m.seed()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    rng = np.random.default_rng(m.SEED)
    logs = []
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for epoch in range(12):
        model.train()
        order = rng.permutation(ids).tolist()
        for group in opt.param_groups:
            group['lr'] = .002 * (.1 + .9 * (1 + math.cos(math.pi * epoch / 11)) / 2)
        total = np.zeros(3)
        for start in range(0, len(order), 24):
            rgb, z, d = data.batch(order[start:start+24], 'cuda')
            opt.zero_grad(set_to_none=True)
            nfo, depth = model.forward_train(rgb, z)
            ln = m.loss_fn(nfo, d, 'nfo')
            ld = m.loss_fn(depth, d, 'depth')
            loss = ln + weight * ld
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += [float(loss.detach()), float(ln.detach()), float(ld.detach())]
        torch.cuda.synchronize()
        logs.append(dict(epoch=epoch+1, loss=total[0]/125, nfo_loss=total[1]/125,
                         depth_loss=total[2]/125, seconds=time.perf_counter()-begin))
        m.write(OUT / 'training-progress.json', logs)
        print('HYBRID_EPOCH', logs[-1], flush=True)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    torch.save(dict(state_dict=model.cpu().state_dict(), weight=weight, logs=logs,
                    protocol=protocol), OUT / 'hybrid-training.pt')
    # Deployment is literally the original NFO architecture, with no depth head.
    pruned = m.Net('nfo')
    pruned.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith('auxiliary_output.')})
    torch.save(dict(arm='nfo', state_dict=pruned.state_dict(), protocol=protocol,
                    manifest_sha256=m.sha(BASE/'manifest.json')), OUT/'trained-nfo.pt')
    model.cuda().eval(); pruned.cuda().eval()
    with torch.inference_mode():
        rgb, z, _ = data.batch(ids[:2], 'cuda')
        assert torch.equal(model(rgb, z), pruned(rgb, z))
    m.write(OUT/'training-receipt.json', dict(device=torch.cuda.get_device_name(),
        backend='CUDA', seconds=logs[-1]['seconds'], weight=weight,
        train_params=sum(p.numel() for p in model.parameters()),
        inference_params=sum(p.numel() for p in pruned.parameters()),
        identical_initial_main_weights=True, pruned_output_bit_identical=True,
        checkpoint_sha256=m.sha(OUT/'trained-nfo.pt')))
    return pruned


@torch.inference_mode()
def evaluate(data, model):
    val = [i for i, r in enumerate(data.rows) if r['split'] == 'val']
    hist = m.histogram(model, data, val, 'cuda')
    cal = m.calibration(hist)
    np.save(OUT/'validation-histogram.npy', hist)
    m.write(OUT/'calibration.json', {'nfo': cal})
    # Original controls and cutoffs are never retrained or recalibrated.
    m.OUT = BASE
    controls = m.load_models('cuda')
    cuts = {a: c['cutoff'] for a, c in json.loads((BASE/'calibration.json').read_text()).items()}
    cuts['hybrid'] = cal['cutoff']
    models = dict(controls, hybrid=model)
    ids = [i for i, r in enumerate(data.rows) if r['split'] == 'test']
    sums = defaultdict(lambda: np.zeros(4, np.int64))
    strata = defaultdict(lambda: np.zeros(4, np.int64))
    scenes = defaultdict(lambda: np.zeros(4, np.int64))
    paired = defaultdict(lambda: np.zeros(4, np.int64))
    frames = []
    unknown = 0
    for start in range(0, len(ids), 24):
        ii = ids[start:start+24]
        rgb, z, _ = data.batch(ii, 'cuda')
        preds = {a: (m.probabilities(net(rgb, z), net.arm).cpu().numpy() >= cuts[a])
                 for a, net in models.items()}
        for j, idx in enumerate(ii):
            d = data.depth[idx].numpy()
            values = data.zones[idx, 0].numpy().ravel()*8
            valid = data.zones[idx, 1].numpy().ravel().astype(bool)
            unknown += int((~np.isfinite(d)).sum())
            rawknown = np.zeros(d.shape, bool)
            for box, v in zip(data.boxes, valid):
                y0, x0, y1, x1 = box
                rawknown[y0:y1, x0:x1] = v
            for ti, t in enumerate(m.THRESHOLDS):
                key = str(float(t))
                known, truth, mixed, thin = m.masks(d, data.boxes, float(t))
                domains = dict(full=known, mixed=mixed, small_foreground=thin,
                               mixed_raw_known=mixed&rawknown, mixed_raw_unknown=mixed&~rawknown)
                for arm in models:
                    pred = preds[arm][j, ti]
                    for name, dom in domains.items():
                        c = m.counts(pred, truth, dom)
                        sums[arm, key, name] += c
                        if ti == 2 and name == 'mixed':
                            scenes[arm, data.rows[idx]['scene']] += c
                            frames.append(dict(id=data.rows[idx]['id'], arm=arm, counts=c.tolist()))
                for zi, (y0, x0, y1, x1) in enumerate(data.boxes):
                    k = known[y0:y1, x0:x1]; n = truth[y0:y1, x0:x1]
                    count, den = int(n.sum()), int(k.sum())
                    if not 0 < count < den:
                        continue
                    b = BINS[int(np.searchsorted([.05, .1, .2, .5], count/den, side='left'))]
                    state = 'missing' if not valid[zi] else ('near' if values[zi] < t else 'far')
                    local = {a: p[j, ti, y0:y1, x0:x1] for a, p in preds.items()}
                    for group in ['all', state]:
                        for arm in models:
                            strata[arm, key, b, group] += m.counts(local[arm], n, k)
                        old, new = local['nfo'], local['hybrid']
                        paired[key, b, group] += np.array([
                            (n&~old&new).sum(), (n&old&~new).sum(),
                            (k&~n&~old&new).sum(), (k&~n&old&~new).sum()])
        if start % 120 == 0:
            print('HYBRID_EVAL', start, flush=True)
    original = json.loads((BASE/'results.json').read_text())
    for (arm, t, name), counts in sums.items():
        if arm != 'hybrid':
            assert counts.tolist() == [original['metrics'][arm][t][name][k] for k in ['tp','fp','fn','tn']]
    metrics = {a: {str(float(t)): {name: m.metrics(c) for (aa, tt, name), c in sums.items()
                   if aa == a and tt == str(float(t))} for t in m.THRESHOLDS} for a in models}
    small = {a: {s: m.metrics(sum((strata[a, '2.0', b, s] for b in BINS[:3]), np.zeros(4, np.int64)))
                 for s in ['all', 'near', 'far', 'missing']} for a in models}
    far = small['hybrid']['far']; main = metrics['hybrid']['2.0']['mixed']
    reference = metrics['nfo']['2.0']['mixed']
    gate = dict(validation_feasible=cal['feasible'], far_recall_at_least_82=far['recall'] >= .82,
        far_iou_retained=far['iou'] >= small['nfo']['far']['iou'],
        mixed_iou_retained=main['iou'] >= reference['iou'],
        mixed_recall_loss_at_most_one_point=main['recall'] >= reference['recall']-.01)
    gate['pass'] = all(gate.values())
    result = dict(metrics=metrics, small_2m=small, calibration=cal, gate=gate,
        records=[dict(arm=a, threshold=t, area=b, return_state=s, **m.metrics(c))
                 for (a,t,b,s),c in strata.items()],
        paired=[dict(threshold=t, area=b, return_state=s, rescued_tp=int(c[0]), lost_tp=int(c[1]),
                     added_fp=int(c[2]), removed_fp=int(c[3])) for (t,b,s),c in paired.items()],
        per_scene={s:{a:m.metrics(scenes[a,s]) for a in models} for _,s in scenes},
        test_unknown_pixels=unknown, baseline_counts_exact=True,
        protocol_sha256=m.sha(OUT/'protocol.json'))
    m.write(OUT/'results.json', result)
    m.write(OUT/'per-frame-counts.json', frames)
    preview(data, models, cuts)
    print('HYBRID_RESULT', json.dumps(dict(gate=gate, small_2m=small, main=main)), flush=True)


@torch.inference_mode()
def preview(data, models, cuts):
    selection = json.loads((BASE/'preview-selection.json').read_text())['small_area']
    panels = []
    for name in selection:
        idx = next(i for i,r in enumerate(data.rows) if r['id'] == name)
        rgb = data.rgb[idx].permute(1,2,0).numpy()
        d = data.depth[idx].numpy(); known = np.isfinite(d); truth = known&(d<2)
        gt = rgb.copy(); gt[truth] = [40,220,100]; gt[~known] = [160,80,180]
        cols = [rgb.copy(), gt]
        x,z,_ = data.batch([idx], 'cuda')
        for arm, net in models.items():
            pred = m.probabilities(net(x,z), net.arm)[0,2].cpu().numpy() >= cuts[arm]
            im = rgb.copy()
            for mask, color in [(pred&truth,[40,220,100]),(pred&~truth&known,[255,70,60]),(~pred&truth,[50,120,255])]:
                im[mask] = (.25*im[mask]+.75*np.array(color)).astype(np.uint8)
            im[~known] = [160,80,180]; cols.append(im)
        row = np.concatenate(cols, 1); header = np.full((32,row.shape[1],3),245,np.uint8)
        for j,label in enumerate(['RGB','GT <2m','depth','NFO','NFO + auxiliary depth']):
            cv2.putText(header,label,(j*m.W+4,21),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        panels.extend([header,row])
    cv2.imwrite(str(OUT/'comparison.jpg'),cv2.cvtColor(np.concatenate(panels),cv2.COLOR_RGB2BGR))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Refuse accidental repeat'
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    m.write(OUT/'protocol.json', dict(id='ba-nfo-hybrid-20260919', phase='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        hypothesis='Continuous-depth auxiliary preserves sparse foreground while NFO retains localization',
        manifest_sha256=m.sha(BASE/'manifest.json'), seed=m.SEED, epochs=12, batch=24,
        optimizer='unchanged AdamW lr .002, weight decay .0001, cosine to 10%, gradient clip 5',
        loss='unchanged BCE + .2 ordinal plus frozen lambda * unchanged log-depth SmoothL1',
        lambda_rule='initial loss ratio on first 32 TRAIN identities; no updates, no validation or test selection',
        architecture='original NFO plus 17-parameter 1x1 auxiliary head on shared last decoder features; head removed for inference',
        calibration='original validation 2m mixed recall>=95%, maximize IoU on .001..999 grid; fixed across distances',
        target='far-return <=20% recall>=82%, subgroup IoU>=original NFO, mixed IoU>=original NFO, mixed recall loss<=1pp; all joint',
        checkpoint='epoch12 only; no sweep, rescue, successor or added data; baseline controls frozen',
        limitation='simulated ToF and consumed Hypersim; mixed-domain IoU is not full-image or final-alert quality',
        backend='CUDA; retained same encoder-decoder placement benchmark',
        code_sha256=m.sha(__file__), baseline_code_sha256=m.sha(m.__file__)))
    try:
        m.OUT = OUT
        data = m.Data(json.loads((BASE/'manifest.json').read_text()))
        model = train(data)
        evaluate(data, model)
        m.write(OUT/'completion.json', dict(status='COMPLETE', gate=json.loads((OUT/'results.json').read_text())['gate']))
    except BaseException as exc:
        m.write(OUT/'completion.json', dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
