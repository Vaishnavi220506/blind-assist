"""Matched, trainable RGB/ToF depth versus near-field supervision pilot.

This entry point is independent of the prior public-data frozen-encoder run.
Only RGB and six public zone fields enter Net.forward; truth stays in losses
and evaluation. Run `all`, then `infer` with either retained checkpoint.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'artifacts.local/work/ba-nfo-20260919'
OUT = ROOT / 'artifacts.local/work/ba-nfo-matched-20260919'
THRESHOLDS = np.array([1., 1.5, 2., 3.], dtype=np.float32)
SEED = 190921
H, W = 192, 256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def public_zones(values):
    """Range, valid, missing, coverage, zone-center x/y; never GT location."""
    v = np.asarray(values, dtype=np.float32).reshape(8, 8)
    valid = np.isfinite(v)
    y, x = np.meshgrid((np.arange(8)+.5)/8*1.6-.8,
                       (np.arange(8)+.5)/8*1.6-.8, indexing='ij')
    return np.stack([np.nan_to_num(v, nan=0.)/8, valid, ~valid,
                     np.ones_like(v), x, y]).astype(np.float32)


def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, b), nn.SiLU(),
                         nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.rgb0 = block(3, 12)
        self.rgb1 = block(12, 24, 2)
        self.rgb2 = block(24, 40, 2)
        self.rgb3 = block(40, 64, 2)
        self.tof = block(6, 16)
        self.fuse = block(80, 64)
        self.up2 = block(104, 40)
        self.up1 = block(64, 24)
        self.up0 = block(36, 16)
        self.output = nn.Conv2d(16, 1 if arm == 'depth' else 4, 1)

    def forward(self, rgb, zones):
        x0 = self.rgb0(rgb.float()/255.-.5)
        x1 = self.rgb1(x0)
        x2 = self.rgb2(x1)
        x3 = self.rgb3(x2)
        z = self.tof(zones)
        # The observed zones cover only the central 80% rectangle. Features
        # are categorical regional observations, not dense per-pixel depth.
        hh, ww = x3.shape[-2:]
        y0, y1, x0i, x1i = round(.1*hh), round(.9*hh), round(.1*ww), round(.9*ww)
        z = F.interpolate(z, (y1-y0, x1i-x0i), mode='nearest')
        z = F.pad(z, (x0i, ww-x1i, y0, hh-y1))
        x = self.fuse(torch.cat([x3, z], 1))
        for skip, module in [(x2, self.up2), (x1, self.up1), (x0, self.up0)]:
            x = module(torch.cat([F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        return self.output(x)


def loss_fn(output, depth, arm):
    valid = torch.isfinite(depth) & (depth > 0)
    safe = torch.nan_to_num(depth, nan=1.).clamp(.1, 80.)
    if arm == 'depth':
        loss = F.smooth_l1_loss(output[:, 0], safe.log(), reduction='none')
        return loss[valid].mean()
    limits = torch.as_tensor(THRESHOLDS, device=output.device)[None, :, None, None]
    target = (safe[:, None] < limits).float()
    pixel = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    ordinal = F.relu(output.sigmoid()[:, :-1]-output.sigmoid()[:, 1:])
    return pixel.permute(0, 2, 3, 1)[valid].mean() + .2*ordinal.permute(0, 2, 3, 1)[valid].mean()


def probabilities(output, arm):
    if arm == 'depth':
        limits = torch.as_tensor(THRESHOLDS, device=output.device)[None, :, None, None]
        return (limits.log()-output).sigmoid()
    # Cummax enforces nested threshold masks; raw violation is reported too.
    return torch.cummax(output.sigmoid(), dim=1).values


def masks(depth, boxes, threshold):
    known = np.isfinite(depth) & (depth > 0)
    near = known & (depth < threshold)
    mixed = np.zeros(depth.shape, bool)
    thin = np.zeros(depth.shape, bool)
    for y0, x0, y1, x1 in boxes:
        k, n = known[y0:y1, x0:x1], near[y0:y1, x0:x1]
        count, positives = int(k.sum()), int(n.sum())
        # Include arbitrarily small resolved foreground (one or more pixels).
        # No P20 condition. 'Thin' means small near area, not a semantic rod.
        if count and 0 < positives < count:
            mixed[y0:y1, x0:x1] = True
            if positives/count <= .2:
                thin[y0:y1, x0:x1] = True
    return known, near, mixed & known, thin & known


def counts(pred, truth, domain):
    return np.array([np.sum(pred & truth & domain), np.sum(pred & ~truth & domain),
                     np.sum(~pred & truth & domain), np.sum(~pred & ~truth & domain)], np.int64)


def metrics(c):
    tp, fp, fn, tn = map(int, c)
    ratio = lambda a, b: a/b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=ratio(tp, tp+fp+fn),
                recall=ratio(tp, tp+fn), precision=ratio(tp, tp+fp),
                false_positive_rate=ratio(fp, fp+tn), pixels=tp+fp+fn+tn)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    allrows = json.loads((OLD/'prepared-manifest.json').read_text())
    rows = []
    families = {}
    for split, budget in [('train', 3000), ('val', 500), ('test', 500)]:
        candidates = [r for r in allrows if r['source'] == 'hypersim' and r['split'] == split]
        families[split] = {r['scene'][:6] for r in candidates}
        # Preserve the pre-existing family split, then sample by identity hash.
        candidates.sort(key=lambda r: hashlib.sha256((str(SEED)+r['id']).encode()).hexdigest())
        rows.extend(dict(r, family=r['scene'][:6]) for r in candidates[:budget])
        assert len(candidates) >= budget
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not families[a] & families[b], (a, b)
    assert len({r['id'] for r in rows}) == 4000
    # Selection and all budgets are written before inspecting selected labels.
    write(manifest_path, rows)
    protocol = dict(id='BA_NFO_MATCHED_20260919', phase='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        question='Does direct near-field supervision outperform same-architecture continuous depth supervision?',
        seed=SEED, resolution=[H, W], source='Hypersim only; existing prepared data',
        split_counts=dict(Counter(r['split'] for r in rows)),
        scene_counts={s:len({r['scene'] for r in rows if r['split']==s}) for s in families},
        family_split={k:sorted(v) for k, v in families.items()},
        split_limit='Unseen training families but previously evaluated Development data; not fresh confirmation',
        arms=['depth', 'nfo'], encoder='same small U-Net, all layers trained from scratch; shared initial backbone weights',
        inputs=['RGB','8x8 range','valid','missing','coverage','zone center x','zone center y'],
        simulator='Existing fixed central-80%-FOV single-return dominant 10cm-bin inverse-square proxy; 1cm+2% noise; 5% dropout; 0.1-8m; no hardware-equivalence claim',
        depth_loss='valid-pixel SmoothL1(log z), target clamp 0.1..80m',
        nfo_loss='valid-pixel BCE four thresholds plus 0.2 ordinal violation; cummax nested probabilities at inference',
        thresholds_m=THRESHOLDS.tolist(), fit_samples=32, fit_steps=512, fit_batch=8,
        fit_gate='Each arm: fit32 2m full-known calibrated recall>=0.95 and IoU>=0.65; diagnose if missed before full fit',
        optimizer='AdamW lr0.002 weight_decay0.0001; cosine to 10% for full fit',
        epochs=12, batch=24, checkpoint='last epoch only; no architecture/loss/budget sweep',
        calibration='Each arm independently on val 2m mixed-known pixels: maximize IoU at recall>=0.95; fixed 0.001..0.999 grid. Same selected cutoff all distance bins; test never tunes.',
        depth_calibration='sigmoid(log(threshold)-logdepth) is a monotone score, not calibrated probability; cutoff is one global depth multiplier',
        primary='2m all-known mixed pixels, including missing ToF zones; also report raw-known/unknown and small-near-area<=20%',
        gate='test primary IoU delta>=0.03, recall delta>=-0.01; positive per-scene IoU delta in >=60% of evaluable test scenes and positive scene-macro mean',
        limits=['shared synthetic dataset assets','one seed','192x256 cannot prove subpixel thin-rod recovery','no BODY/HEAD or alert evaluation','fit failure diagnoses implementation, generalization failure closes this recipe only'],
        budget_stop='One two-arm full fit following fit32; no rescue or successor',
        existing_data_sha256=sha(OLD/'prepared-manifest.json'), manifest_sha256=sha(manifest_path),
        code_sha256=sha(__file__), old_simulator_code_sha256=sha(Path(__file__).with_name('ba_nfo_data.py')))
    write(OUT/'protocol.json', protocol)
    print('PREPARED_SELECTION', protocol['split_counts'], protocol['scene_counts'], flush=True)
    return rows


class Data:
    def __init__(self, rows):
        self.rows = rows
        rgb, depth, zones, boxes = [], [], [], None
        audit = Counter()
        for i, r in enumerate(rows):
            path = OLD/r['prepared']
            assert sha(path) == r['sha256'], r['id']
            with np.load(path) as a:
                im, d = a['rgb'], a['depth']
                assert im.dtype == np.uint8 and im.shape == (H, W, 3)
                assert d.shape == (H, W) and np.isfinite(d).any()
                assert not (np.isfinite(d) & (d <= 0)).any()
                assert np.allclose(a['boxes'], boxes) if boxes is not None else True
                boxes = a['boxes'].copy()
                rgb.append(im.transpose(2, 0, 1).copy())
                depth.append(d.astype(np.float32))
                zones.append(public_zones(a['values']))
                audit['known_depth_pixels'] += int(np.isfinite(d).sum())
                audit['missing_depth_pixels'] += int(np.isnan(d).sum())
                audit['valid_returns'] += int(np.isfinite(a['values']).sum())
            if i % 500 == 0:
                print('CONTENT_VALIDATED', i, '/', len(rows), flush=True)
        self.rgb = torch.from_numpy(np.stack(rgb))
        self.depth = torch.from_numpy(np.stack(depth))
        self.zones = torch.from_numpy(np.stack(zones))
        self.boxes = boxes
        write(OUT/'content-admission.json', dict(status='VALIDATED_4000_RGB_DENSE_DEPTH_CALIBRATED_PREPARED_PAIRS',
              rows=len(rows), array_hashes_verified=True, audit=dict(audit),
              camera_metadata_sha256=sha(OLD/'metadata_camera_parameters.csv'),
              unknown='NaN preserved; not labeled far; missing returns have separate indicator'))

    def batch(self, ids, device):
        return tuple(a[ids].to(device) for a in [self.rgb, self.zones, self.depth])


def calibration(hist):
    # Exact counts for the fixed cutoff grid with quantized bin edges.
    pos, neg = hist
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    fn = pos.sum()-tp
    recall = tp/max(1, pos.sum())
    iou = tp/np.maximum(1, tp+fp+fn)
    possible = np.flatnonzero((recall[1:1000] >= .95))+1
    chosen = int(possible[np.argmax(iou[possible])]) if len(possible) else 1
    return dict(cutoff=chosen/1000, recall=float(recall[chosen]), iou=float(iou[chosen]),
                feasible=bool(len(possible)), positives=int(pos.sum()), negatives=int(neg.sum()))


@torch.inference_mode()
def histogram(model, data, ids, device, domain='mixed'):
    hist = np.zeros((2, 1001), dtype=np.int64)
    model.eval()
    for start in range(0, len(ids), 24):
        ii = ids[start:start+24]
        rgb, z, _ = data.batch(ii, device)
        p = probabilities(model(rgb, z), model.arm)[:, 2].cpu().numpy()
        for idx, score in zip(ii, p):
            d = data.depth[idx].numpy()
            known, truth, mixed, _ = masks(d, data.boxes, 2.)
            dom = mixed if domain == 'mixed' else known
            bins = np.clip(np.floor(score*1000).astype(int), 0, 1000)
            hist[0] += np.bincount(bins[dom & truth], minlength=1001)
            hist[1] += np.bincount(bins[dom & ~truth], minlength=1001)
    return hist


def model_pair(device):
    seed()
    depth = Net('depth')
    seed()
    nfo = Net('nfo')
    base = depth.state_dict()
    nfo.load_state_dict({k:base[k] if not k.startswith('output.') else v for k,v in nfo.state_dict().items()})
    for key in base:
        if not key.startswith('output.'):
            assert torch.equal(base[key], nfo.state_dict()[key])
    return {m.arm:m.to(device) for m in (depth, nfo)}


def backend(data):
    sys.path.insert(0, str(ROOT/'tools'))
    from research_backend import BackendCandidate, select_backend, torch_observation
    models = {dev:Net('depth').to(dev) for dev in ['cpu', 'cuda']}
    batches = {dev:data.batch(list(range(8)), dev) for dev in models}
    def probe(dev):
        m = models[dev]
        m.zero_grad(set_to_none=True)
        rgb, z, d = batches[dev]
        output = m(rgb, z)
        loss_fn(output, d, 'depth').backward()
        return output.detach()
    candidates = {dev:BackendCandidate('torch-'+dev, dev, lambda d=dev:probe(d),
                 lambda o:torch_observation(output=o), torch.cuda.synchronize if dev=='cuda' else lambda:None)
                  for dev in models}
    receipt = select_backend('batch-tensor', cpu=candidates['cpu'], gpu=candidates['cuda'],
                            record_path=OUT/'backend.json', warmups=1, repeats=2)
    del models, batches
    torch.cuda.empty_cache()
    return receipt['selected_device_type']


def fit(data, device, small=False):
    protocol = json.loads((OUT/'protocol.json').read_text())
    train = [i for i,r in enumerate(data.rows) if r['split']=='train']
    if small:
        train = [i for i in train if int((data.depth[i]<2).sum()) >= 32][:32]
        assert len(train) == 32
        write(OUT/'fit32-ids.json', [data.rows[i]['id'] for i in train])
    models = model_pair(device)
    receipt = {}
    for arm, model in models.items():
        dest = OUT/f'{"fit32" if small else "trained"}-{arm}.pt'
        assert not dest.exists(), f'Refusing to overwrite trained checkpoint: {dest}'
        seed()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
        initial = calibration(histogram(model, data, train[:32], device, 'full')) if small else None
        rng = np.random.default_rng(SEED)
        epochs = 1 if small else protocol['epochs']
        logs = []
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            if small:
                batches = [rng.choice(train, 8, replace=False).tolist() for _ in range(protocol['fit_steps'])]
            else:
                order = rng.permutation(train).tolist()
                batches = [order[k:k+24] for k in range(0, len(order), 24)]
                lr = .002*(.1+.9*(1+math.cos(math.pi*epoch/max(1, epochs-1)))/2)
                for group in optimizer.param_groups:
                    group['lr'] = lr
            total = 0.
            for step, ii in enumerate(batches):
                rgb, z, d = data.batch(ii, device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(rgb, z)
                loss = loss_fn(pred, d, arm)
                assert torch.isfinite(loss), (arm, epoch, step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach())
                if small and step % 128 == 0:
                    print('FIT32', arm, step, float(loss), flush=True)
            logs.append(dict(epoch=epoch+1, loss=total/len(batches), elapsed_seconds=time.perf_counter()-start))
            write(OUT/f'progress-{"fit32" if small else "trained"}-{arm}.json', logs)
            print('EPOCH', arm, 'fit32' if small else 'full', logs[-1], flush=True)
        torch.save(dict(arm=arm, state_dict=model.cpu().state_dict(), protocol=protocol,
                        protocol_sha256=sha(OUT/'protocol.json'), manifest_sha256=sha(OUT/'manifest.json'),
                        initialization='identical RGB/ToF/fusion/decoder weights across arms', logs=logs), dest)
        model.to(device)
        final = calibration(histogram(model, data, train, device, 'full')) if small else None
        receipt[arm] = dict(checkpoint=str(dest), sha256=sha(dest), params=sum(p.numel() for p in model.parameters()),
                            initial=initial, final=final, elapsed_seconds=time.perf_counter()-start)
        if small:
            receipt[arm]['pass'] = final['recall']>=.95 and final['iou']>=.65
    write(OUT/f'{"fit32" if small else "training"}-receipt.json', receipt)
    if small:
        make_preview(data, train[:6], models, {a:r['final']['cutoff'] for a,r in receipt.items()}, device, 'fit32-preview.jpg')
        assert all(r['pass'] for r in receipt.values()), 'FIT32_GATE_MISSED: inspect labels/implementation/capacity before full fit'
    return receipt


def load_models(device):
    models = {}
    for arm in ['depth', 'nfo']:
        payload = torch.load(OUT/f'trained-{arm}.pt', map_location='cpu', weights_only=False)
        assert payload['manifest_sha256'] == sha(OUT/'manifest.json')
        model = Net(arm).to(device)
        model.load_state_dict(payload['state_dict'])
        models[arm] = model.eval()
    return models


@torch.inference_mode()
def make_preview(data, ids, models, cuts, device, filename):
    panels = []
    for idx in ids:
        rgb = data.rgb[idx].permute(1,2,0).numpy()
        d = data.depth[idx].numpy()
        truth = np.isfinite(d) & (d<2)
        cols = [rgb.copy()]
        gt = rgb.copy(); gt[truth] = (.35*gt[truth]+.65*np.array([40,220,100])).astype(np.uint8)
        gt[~np.isfinite(d)] = [160,80,180]
        cols.append(gt)
        x,z,_ = data.batch([idx], device)
        for arm in ['depth','nfo']:
            pred = probabilities(models[arm].eval()(x,z),arm)[0,2].cpu().numpy() >= cuts[arm]
            im = rgb.copy()
            for mask,color in [(pred&truth,[40,220,100]),(pred&~truth&np.isfinite(d),[255,70,60]),(~pred&truth,[50,120,255])]:
                im[mask] = (.25*im[mask]+.75*np.array(color)).astype(np.uint8)
            im[~np.isfinite(d)] = [160,80,180]
            cols.append(im)
        row = np.concatenate(cols,axis=1)
        header = np.full((38,row.shape[1],3),245,np.uint8)
        labels = ['RGB '+data.rows[idx]['id'], 'GT <2m', 'continuous depth', 'direct near-field']
        for j,label in enumerate(labels):
            cv2.putText(header,label,(j*W+4,23),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
        panels.extend([header,row])
    image = np.concatenate(panels,axis=0)
    cv2.imwrite(str(OUT/filename), cv2.cvtColor(image,cv2.COLOR_RGB2BGR))


@torch.inference_mode()
def evaluate(data, device):
    models = load_models(device)
    val = [i for i,r in enumerate(data.rows) if r['split']=='val']
    test = [i for i,r in enumerate(data.rows) if r['split']=='test']
    calibrations = {}
    for arm, model in models.items():
        hist = histogram(model, data, val, device)
        calibrations[arm] = calibration(hist)
        np.save(OUT/f'validation-histogram-{arm}.npy',hist)
    write(OUT/'calibration.json',calibrations)  # sealed before test inference
    cuts = {a:r['cutoff'] for a,r in calibrations.items()}
    sums = defaultdict(lambda: np.zeros(4,np.int64))
    scene_sums = defaultdict(lambda: np.zeros(4,np.int64))
    frame_rows = []
    violations = Counter()
    for start in range(0,len(test),24):
        ids = test[start:start+24]
        rgb,z,_ = data.batch(ids,device)
        outputs = {a:m(rgb,z) for a,m in models.items()}
        raw_nfo = outputs['nfo'].sigmoid()
        violations['adjacent_probability_pairs'] += int(raw_nfo[:,:-1].numel())
        violations['raw_violations'] += int((raw_nfo[:,:-1]>raw_nfo[:,1:]).sum())
        scores = {a:probabilities(o,a).cpu().numpy() for a,o in outputs.items()}
        for j,idx in enumerate(ids):
            row = data.rows[idx]
            d = data.depth[idx].numpy()
            rawknown = np.zeros((H,W),bool)
            for box, valid in zip(data.boxes,data.zones[idx,1].numpy().ravel()):
                y0,x0,y1,x1 = box
                rawknown[y0:y1,x0:x1] = bool(valid)
            for k,t in enumerate(THRESHOLDS):
                known,truth,mixed,thin = masks(d,data.boxes,float(t))
                domains = dict(full=known,mixed=mixed,small_foreground=thin,
                               mixed_raw_known=mixed&rawknown,mixed_raw_unknown=mixed&~rawknown)
                for arm in models:
                    pred = scores[arm][j,k]>=cuts[arm]
                    for name,domain in domains.items():
                        c = counts(pred,truth,domain)
                        sums[arm,str(float(t)),name] += c
                        if k==2 and name=='mixed':
                            scene_sums[arm,row['scene']] += c
                            frame_rows.append(dict(id=row['id'],scene=row['scene'],arm=arm,counts=c.tolist()))
                    sums[arm,str(float(t)),'full_uncalibrated'] += counts(scores[arm][j,k]>=.5,truth,known)
            if start%120==0 and j==0:
                print('TEST_EVALUATED',start,'/',len(test),flush=True)
    result = {a:{t:{name:metrics(c) for (aa,tt,name),c in sums.items() if aa==a and tt==t}
                 for t in map(lambda x:str(float(x)),THRESHOLDS)} for a in models}
    scenes = {s:{a:metrics(scene_sums[a,s]) for a in models} for _,s in scene_sums}
    deltas = [v['nfo']['iou']-v['depth']['iou'] for v in scenes.values()
              if v['nfo']['iou'] is not None and v['depth']['iou'] is not None and v['depth']['tp']+v['depth']['fn']>0]
    base,cand = result['depth']['2.0']['mixed'],result['nfo']['2.0']['mixed']
    gain = cand['iou']-base['iou']
    recall_delta = cand['recall']-base['recall']
    wins = sum(x>0 for x in deltas)
    gate = dict(iou_delta=gain,recall_delta=recall_delta,scene_wins=wins,evaluable_scenes=len(deltas),
                scene_macro_iou_delta=float(np.mean(deltas)),
                validation_calibration_feasible=all(c['feasible'] for c in calibrations.values()),
                pass_gate=bool(all(c['feasible'] for c in calibrations.values()) and gain>=.03 and recall_delta>=-.01 and len(deltas)>1 and wins/len(deltas)>=.6 and np.mean(deltas)>0))
    write(OUT/'results.json',dict(metrics=result,per_scene=scenes,gate=gate,calibration=calibrations,
                                ordinal_raw_violations=dict(violations),nested_output_violations=0,
                                scope='Synthetic consumed Development; no final alerts changed',
                                protocol_sha256=sha(OUT/'protocol.json')))
    write(OUT/'per-frame-counts.json',frame_rows)
    # Predeclared identity order, first eligible frame per distinct scene, no
    # selection by model success. Separate small-area examples are GT-selected.
    examples,seen = [],set()
    thin_examples,thin_seen = [],set()
    for idx in test:
        s = data.rows[idx]['scene']
        _,truth,mixed,thin = masks(data.depth[idx].numpy(),data.boxes,2.)
        if (truth&mixed).any() and s not in seen and len(examples)<6:
            examples.append(idx);seen.add(s)
        if (truth&thin).any() and s not in thin_seen and len(thin_examples)<6:
            thin_examples.append(idx);thin_seen.add(s)
    write(OUT/'preview-selection.json',dict(rule='first in sealed identity order per scene with GT mixed/small-area positives; no prediction selection',
              mixed=[data.rows[i]['id'] for i in examples], small_area=[data.rows[i]['id'] for i in thin_examples]))
    make_preview(data,examples,models,cuts,device,'comparison.jpg')
    make_preview(data,thin_examples,models,cuts,device,'small-foreground.jpg')
    make_curves()
    print('RESULT',json.dumps(gate),flush=True)
    return gate


def make_curves():
    # Dev tradeoff curves are shown to explain calibration; test is one fixed point.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for arm in ['depth','nfo']:
        pos,neg=np.load(OUT/f'validation-histogram-{arm}.npy')
        tp=np.cumsum(pos[::-1])[::-1]; fp=np.cumsum(neg[::-1])[::-1]
        recall=tp/max(1,pos.sum()); fpr=fp/max(1,neg.sum())
        iou=tp/np.maximum(1,pos.sum()+fp)
        axes[0].plot(fpr,recall,label=arm)
        axes[1].plot(recall,iou,label=arm)
    axes[0].set(xlabel='False positive pixel rate',ylabel='Recall',title='Validation: 2m mixed zones')
    axes[1].set(xlabel='Recall',ylabel='IoU',title='Validation operating-point tradeoff')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'validation-curves.png',dpi=160);plt.close(fig)


def inference(checkpoint, sample, output):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Net(payload['arm']).to(device).eval()
    model.load_state_dict(payload['state_dict'])
    with np.load(sample) as a:
        rgb=torch.from_numpy(a['rgb'].transpose(2,0,1).copy())[None].to(device)
        zones=torch.from_numpy(public_zones(a['values']))[None].to(device)
    with torch.inference_mode():
        native=model(rgb,zones)
        score=probabilities(native,model.arm)[0].cpu().numpy()
    # Inference deliberately never reads dense truth or evaluator strata.
    fields=dict(scores=score,thresholds_m=THRESHOLDS,
                native_output=native[0].cpu().numpy(),arm=np.array(model.arm))
    calibration_path=Path(checkpoint).parent/'calibration.json'
    if calibration_path.exists():
        cutoff=json.loads(calibration_path.read_text())[model.arm]['cutoff']
        fields.update(masks=score>=cutoff,cutoff=np.array(cutoff))
    np.savez_compressed(output,**fields)
    print('INFERENCE',output,score.shape,device,
          torch.cuda.get_device_name() if device=='cuda' else 'ACCELERATOR_UNAVAILABLE',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['all','prepare','fit32','train','evaluate','infer'])
    p.add_argument('--checkpoint',type=Path);p.add_argument('--sample',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.action=='infer':
        inference(args.checkpoint,args.sample,args.output);return
    rows=prepare()
    if args.action=='prepare':return
    data=Data(rows)
    if args.action in ['all','fit32']:
        device=backend(data)
    else:
        device=json.loads((OUT/'backend.json').read_text())['selected_device_type']
    if args.action in ['all','fit32']:fit(data,device,True)
    if args.action in ['all','train']:
        assert all(r['pass'] for r in json.loads((OUT/'fit32-receipt.json').read_text()).values())
        fit(data,device)
    if args.action in ['all','evaluate']:evaluate(data,device)


if __name__=='__main__':
    main()
