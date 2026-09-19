"""One matched full-TRAIN supervision contrast; no calibration or test access."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np
import torch

import ba_nfo_matched as m
from ba_nfo_zcr import domains
from ba_nfo_zone_readout import load_base
from ba_nfo_frozen_transfer import evaluation_domains, gates
from audit_ba_nfo_spatial_tradeoff import transitions

OUT = m.ROOT / 'artifacts.local/work/ba-nfo-fullsupport-20260919'
ARMS = ['original', 'control', 'support']
CUT = .081
EPOCHS = 12
BATCH = 24


def support_loss(output, depth, small, pure):
    """Each extra domain is a whole known zone, not foreground-only pixels.

    Full-image supervision includes outside coverage and all other pixels.
    Empty batch strata use full loss, retaining the total coefficient of one.
    """
    full = m.loss_fn(output, depth, 'nfo')
    extra = []
    for mask in [small, pure]:
        known = mask & torch.isfinite(depth) & (depth > 0)
        extra.append(m.loss_fn(output, depth.masked_fill(~known, float('nan')), 'nfo')
                     if known.any() else full)
    return .5 * full + .25 * extra[0] + .25 * extra[1]


def read_array(row):
    path = m.OLD / row['prepared']
    assert m.sha(path) == row['sha256'], row['id']
    with np.load(path) as a:
        return {k: a[k].copy() for k in ['rgb', 'depth', 'values', 'boxes']}


def seal():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'One frozen recipe only'
    manifest = json.loads((m.OUT / 'manifest.json').read_text())
    train = [r for r in manifest if r['split'] == 'train']
    val = [r for r in manifest if r['split'] == 'val']
    assert len(train) == 3000 and len(val) == 500
    for key in ['id', 'scene', 'family']:
        assert not {r[key] for r in train} & {r[key] for r in val}
    rng = np.random.default_rng(m.SEED)
    orders = np.stack([rng.permutation(len(train)) for _ in range(EPOCHS)])
    np.save(OUT / 'orders.npy', orders)
    m.write(OUT / 'train-manifest.json', train)
    m.write(OUT / 'development-manifest.json', val)
    protocol = dict(id='ba-nfo-fullsupport-20260919', phase='EXPLORE_CONSUMED_SYNTHETIC_DEVELOPMENT',
        question='Does broad small-support supervision improve cross-scene localization at the original fixed cutoff?',
        hypothesis='Representative full training plus explicit whole-small-zone and pure-far supervision can avoid the selected32 adaptation failure.',
        contrast='Same original NFO architecture, from-scratch initialization, training images, order, optimizer and budget; only training loss differs.',
        original='Retained original trained NFO reference, unchanged; previous joint32 and late32 transfer failures remain negative controls in report.',
        initialization='Exact original m.model_pair(cpu)[nfo] initialization, seed190921, copied identically to both new arms',
        epochs=EPOCHS, batch=BATCH, updates_per_arm=1500, seed=m.SEED,
        optimizer='AdamW lr .002*(.1+.9*(1+cos(pi*epoch/11))/2), weight_decay .0001, clip norm5',
        control_loss='Original full-known four-threshold BCE + .2 ordinal',
        support_loss='.5 L_full + .25 L_small + .25 L_pure_far; each original loss separately known-pixel normalized; empty batch stratum substitutes L_full',
        small='All known pixels in every ToF-geometry zone with 0<near2m/known<=.2, regardless return; one-pixel, near2m and missing-return cases included',
        pure_far='Canonical known pixels in zones with finite public return>=2m and no known near2m pixels',
        inputs='Unchanged full RGB192x256 and six public8x8 fields only; labels select loss masks only, never forward inputs',
        checkpoint='Last epoch only, no intermediate evaluation or selection', cutoff=CUT,
        evaluation='All original500val, disjoint train scenes/families but previously consumed; no originaltest observations',
        promotion='Four user gates vs retained original: far-small recall>=.75, IoU>=original, mixed recall>=.945, pure-far FP<=original. Also support must not regress vs matchedcontrol in these four metrics and improve far-small recall or IoU strictly.',
        diagnostics='Full, outside, all-small, near-depth strata, six families and paired TP/FP transitions; do not replace main denominators',
        interpretation='This is not an isolated test of coverage against32: start, coverage, weighting and budget differ from32. The new matched contrast isolates this loss recipe at full TRAIN coverage.',
        stop='One seed, two12epoch fits and one500frame evaluation; no threshold/loss/budget/architecture sweep or automatic successor',
        backend='CUDA; reuse original architecture training and latefusion CPU/GPU placement evidence; record actual runtime',
        source_manifest_sha256=m.sha(m.OUT / 'manifest.json'), orders_sha256=m.sha(OUT / 'orders.npy'),
        source_checkpoint_sha256=m.sha(m.OUT / 'trained-nfo.pt'),
        code_sha256={p.name:m.sha(p) for p in [m.ROOT/'research/active/dtr-r0/nearfield'/n for n in ['ba_nfo_fullsupport.py','ba_nfo_matched.py','ba_nfo_zcr.py']]})
    m.write(OUT / 'protocol.json', protocol)
    return train, val, orders, protocol


def training_data(rows):
    rgb, depth, zones, small, pure = [], [], [], [], []
    coverage = Counter(); per_family = defaultdict(Counter)
    for i, row in enumerate(rows):
        a = read_array(row); truth, dm = domains(a)
        rgb.append(a['rgb'].transpose(2, 0, 1).copy()); depth.append(a['depth'])
        zones.append(m.public_zones(a['values'])); small.append(dm['small_foreground']); pure.append(dm['pure_far'])
        c = dict(known=int(dm['full'].sum()), unknown=int((~dm['full']).sum()),
                 small_known=int(dm['small_foreground'].sum()), small_near=int((dm['small_foreground'] & truth).sum()),
                 small_near_gt1p8=int((dm['small_foreground'] & truth & (a['depth']>1.8)).sum()),
                 pure_far_known=int(dm['pure_far'].sum()), frames=1,
                 frames_with_small=int(dm['small_foreground'].any()), frames_with_pure_far=int(dm['pure_far'].any()))
        coverage.update(c); per_family[row['family']].update(c)
        if (i+1)%500 == 0: print('TRAIN_DATA', i+1, flush=True)
    arrays = [torch.from_numpy(np.stack(x)) for x in [rgb, zones, depth, small, pure]]
    m.write(OUT/'training-coverage.json', dict(total=dict(coverage), per_family={k:dict(v) for k,v in per_family.items()},
        scenes=len({r['scene'] for r in rows}), families=len(per_family)))
    return arrays


def train(arrays, orders, protocol):
    initial = m.model_pair('cpu')['nfo'].state_dict()
    torch.save(initial, OUT/'initial-state.pt')
    receipt = {}; ratios = {'small':[], 'pure_far':[]}; empty = Counter()
    known_counts = (torch.isfinite(arrays[2]) & (arrays[2]>0)).sum((1,2)).numpy()
    domain_counts = [a.sum((1,2)).numpy() for a in arrays[3:]]
    for order in orders:
        for start in range(0, len(order), BATCH):
            ii = order[start:start+BATCH]; total = int(known_counts[ii].sum())
            for name, c in zip(ratios, domain_counts):
                n = int(c[ii].sum())
                if n: ratios[name].append(1+.5*total/n)
                else: empty[name] += 1
    m.write(OUT/'loss-coefficients.json', dict(relative_to_unfocused_pixel={k:dict(min=min(v), median=float(np.median(v)), max=max(v)) for k,v in ratios.items()},
        empty_batch_strata=dict(empty), meaning='Per-known-pixel coefficients, not measured gradients; disjoint small/pure masks, identical four-head formulas'))
    for arm in ['control', 'support']:
        dest = OUT/f'{arm}.pt'; assert not dest.exists()
        m.seed(); model = m.Net('nfo'); model.load_state_dict(initial)
        assert all(torch.equal(v, initial[k]) for k,v in model.state_dict().items())
        model.cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
        start_time = time.perf_counter(); logs = []
        for epoch, order in enumerate(orders):
            lr = .002*(.1+.9*(1+math.cos(math.pi*epoch/(EPOCHS-1)))/2)
            for group in opt.param_groups: group['lr'] = lr
            total = 0.
            for start in range(0, len(order), BATCH):
                ii = order[start:start+BATCH].tolist()
                rgb, zones, depth, small, pure = [a[ii].cuda() for a in arrays]
                opt.zero_grad(set_to_none=True); output = model(rgb,zones)
                assert output.device.type == 'cuda'
                loss = m.loss_fn(output,depth,'nfo') if arm=='control' else support_loss(output,depth,small,pure)
                assert torch.isfinite(loss), (arm,epoch,start)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
                total += float(loss.detach())
            logs.append(dict(epoch=epoch+1, loss=total/125, lr=lr, seconds=time.perf_counter()-start_time))
            m.write(OUT/f'progress-{arm}.json', logs); print('FULL_TRAIN',arm,logs[-1],flush=True)
        torch.cuda.synchronize()
        torch.save(dict(state_dict=model.cpu().state_dict(),arm='nfo',experiment_arm=arm,logs=logs,
                        protocol_sha256=m.sha(OUT/'protocol.json'),initial_sha256=m.sha(OUT/'initial-state.pt')),dest)
        receipt[arm] = dict(sha256=m.sha(dest), updates=1500, epochs=12, params=sum(p.numel() for p in model.parameters()),
                            seconds=time.perf_counter()-start_time, orders_sha256=m.sha(OUT/'orders.npy'))
        del model, opt; torch.cuda.empty_cache()
    m.write(OUT/'training-receipt.json',receipt)


@torch.inference_mode()
def evaluate(rows, protocol):
    models = {'original':load_base()}
    for arm in ['control','support']:
        models[arm] = m.Net('nfo')
        models[arm].load_state_dict(torch.load(OUT/f'{arm}.pt',map_location='cpu',weights_only=False)['state_dict'])
    models = {k:v.cuda().eval() for k,v in models.items()}
    sums = defaultdict(lambda:np.zeros(4,np.int64)); family = defaultdict(lambda:np.zeros(4,np.int64))
    paired = defaultdict(Counter); packed = []; frames = []; start_time = time.perf_counter()
    for start in range(0,len(rows),BATCH):
        batch = rows[start:start+BATCH]; arrays = [read_array(r) for r in batch]
        rgb = torch.from_numpy(np.stack([a['rgb'].transpose(2,0,1).copy() for a in arrays])).cuda()
        zones = torch.from_numpy(np.stack([m.public_zones(a['values']) for a in arrays])).cuda()
        predictions = {k:(m.probabilities(net(rgb,zones),'nfo')[:,2]>=CUT).cpu().numpy() for k,net in models.items()}
        for i,(row,a) in enumerate(zip(batch,arrays)):
            truth,dm = evaluation_domains(a); fm = {arm:{} for arm in ARMS}
            for domain,mask in dm.items():
                for arm in ARMS:
                    c=m.counts(predictions[arm][i],truth,mask); sums[arm,domain]+=c
                    family[row['family'],arm,domain]+=c; fm[arm][domain]=m.metrics(c)
                for old,new in [('original','control'),('original','support'),('control','support')]:
                    paired[old+'->'+new,domain].update(transitions(predictions[old][i],predictions[new][i],truth,mask))
            packed.append(np.stack([np.packbits(predictions[arm][i].ravel(),bitorder='little') for arm in ARMS]))
            frames.append(dict(id=row['id'],scene=row['scene'],family=row['family'],metrics=fm))
    metrics={arm:{d:m.metrics(c) for (aa,d),c in sums.items() if aa==arm} for arm in ARMS}
    historical=json.loads((m.ROOT/'artifacts.local/work/ba-nfo-frozen-transfer500-20260919/results.json').read_text())['metrics']['nfo']
    assert metrics['original']==historical, 'Original baseline must reproduce exactly'
    baseline, candidate=metrics['control'],metrics['support']
    matched=dict(far_small_recall=candidate['far_small']['recall']>=baseline['far_small']['recall'],
        far_small_iou=candidate['far_small']['iou']>=baseline['far_small']['iou'],
        mixed_recall=candidate['mixed']['recall']>=baseline['mixed']['recall'],
        pure_far_fp=candidate['pure_far']['fp']<=baseline['pure_far']['fp'],
        strict_small_gain=candidate['far_small']['recall']>baseline['far_small']['recall'] or candidate['far_small']['iou']>baseline['far_small']['iou'])
    result=dict(metrics=metrics,user_gates={a:gates(metrics[a],metrics['original']) for a in ARMS[1:]},matched_guards=matched,
        family_metrics={f:{arm:{d:m.metrics(c) for (ff,aa,d),c in family.items() if ff==f and aa==arm} for arm in ARMS} for f in sorted({r['family'] for r in rows})},
        paired={pair:{d:dict(c) for (pp,d),c in paired.items() if pp==pair} for pair in ['original->control','original->support','control->support']},
        original_counts_exact=True,development_frames=len(rows),test_frames_read=0,evaluation_seconds=time.perf_counter()-start_time)
    result['promotable_development']=all(result['user_gates']['support'].values()) and all(matched.values())
    result['status']='PASS_DEVELOPMENT_REQUIRES_CONFIRMATION' if result['promotable_development'] else 'REJECT_EXACT_FULLSUPPORT_RECIPE'
    assert m.sha(m.OUT/'trained-nfo.pt')==protocol['source_checkpoint_sha256']
    np.savez_compressed(OUT/'predictions.npz',masks=np.stack(packed))
    m.write(OUT/'frames.json',frames); m.write(OUT/'results.json',result)
    m.write(OUT/'completion.json',dict(status=result['status'],device=torch.cuda.get_device_name(),torch=torch.__version__,
        updates_per_arm=1500,original_checkpoint_unchanged=True,test_frames_read=0,prediction_arms=ARMS,packing='little-bit-order 500x3x6144'))
    print('FULLSUPPORT_RESULT',json.dumps({k:result[k] for k in ['status','user_gates','matched_guards']}),flush=True)


def main():
    torch.set_num_threads(4); assert torch.cuda.is_available()
    rows,val,orders,protocol=seal()
    arrays=training_data(rows)
    train(arrays,orders,protocol)
    del arrays
    evaluate(val,protocol)


if __name__=='__main__': main()
