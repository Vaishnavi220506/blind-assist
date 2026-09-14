"""Matched direct corridor BCE / BCE+pair training; small fit check first."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'tools'))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from mz120_occupancy import encode
from mz136_corridor_pair import CorridorNet, corridor_rank
from run_mz107_four_sensor import sha, write, readrows, truth, metrics
from run_mz111_spatial_temporal import event_metrics
from run_mz120_occupancy import batch
from research_backend import BackendCandidate, DeviceObservation, select_backend

SEED = 136014
STEPS = 800
MARGIN = 1.
WEIGHT = .25


def load(capture, image_root):
    receipt = json.loads((capture/'receipt.json').read_text())
    assert receipt['status'] == 'PASS'
    assert sha(capture/'spec.json') == receipt['spec_sha256']
    for filename in ('raw.jsonl', 'evaluator.jsonl'):
        assert sha(capture/filename) == receipt['hashes'][filename]
    rows = readrows(capture/'raw.jsonl')
    es = readrows(capture/'evaluator.jsonl')
    assert [r['id'] for r in rows] == [e['id'] for e in es]
    spec = json.loads((capture/'spec.json').read_text())
    inputs, images, yaw, episode = [], [], 0., None
    for r in rows:
        if r['episode_id'] != episode: yaw = 0.
        if r['imu_valid']: yaw += r['delta_yaw']
        episode = r['episode_id']
        inputs.append(encode(r, yaw))
        path = (image_root/r['rgb_path']).resolve()
        assert path.is_relative_to(image_root.resolve())
        assert sha(path) == receipt['hashes'][r['rgb_path']]
        im = cv2.imread(str(path)); assert im is not None and im.shape == (360,640,3)
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    data = {k:torch.from_numpy(np.stack([v[k] for v in inputs])) for k in inputs[0]}
    data['rgb'] = torch.from_numpy(np.stack(images).transpose(0,3,1,2))
    return rows, es, spec, data


def pair_indices(rows, spec):
    episodes = {}
    for i,r in enumerate(rows): episodes.setdefault(r['episode_id'],[]).append(i)
    pairs = []
    for p in spec['pairs']:
        a,b = [episodes[ep] for ep in p['episodes']]
        assert len(a)==len(b)
        assert [rows[i]['time_s'] for i in a] == [rows[i]['time_s'] for i in b]
        pairs.extend(dict(a=i,b=j,pair_id=p['pair_id'],family=p['category'],
                          split=p.get('split'),scene_group=p.get('scene_group',p['pair_id']))
                     for i,j in zip(a,b))
    assert sorted([i for p in pairs for i in (p['a'],p['b'])]) == list(range(len(rows)))
    return pairs


def digest_state(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_schedule(pairs, steps):
    ix=np.array([[p['a'],p['b']] for p in pairs])
    rng=np.random.default_rng(SEED)
    schedule=ix[rng.integers(0,len(ix),(steps,4))].reshape(steps,8)
    aug=np.repeat(np.random.default_rng(SEED+1).random((steps,2,4,1,1,1),dtype=np.float32),2,axis=2)
    return schedule, aug


def predict(model, data):
    model.eval(); scores=[]
    with torch.no_grad():
        for k in range(0,len(data['rgb']),8):
            scores.extend(model(batch(data,np.arange(k,min(k+8,len(data['rgb']))))).cpu().tolist())
    return np.array(scores)


def fit(data, gt, pairs, initial, out, steps, augment=True):
    schedule, aug=make_schedule(pairs,steps)
    np.savez_compressed(out/'schedule.npz',indices=schedule,augmentation=aug)
    target=torch.tensor(gt,dtype=torch.float32,device='cuda')
    results={}
    for arm in ('bce','pair'):
        model=CorridorNet().cuda(); model.load_state_dict(initial)
        assert digest_state(model.state_dict())==digest_state(initial)
        model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        progress=[]; tick=time.perf_counter()
        for step,idx in enumerate(schedule):
            assert time.perf_counter()-tick<1200, '20 minute fit cap'
            b=batch(data,idx)
            contrast=torch.from_numpy(.75+.5*aug[step,0]).cuda()
            offset=torch.from_numpy(-.1+.2*aug[step,1]).cuda()
            if augment:
                b['rgb']=((b['rgb']-.5)*contrast+.5+offset).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            scores=model(b); classification=F.binary_cross_entropy_with_logits(scores,target[idx])
            rank=corridor_rank(scores,target[idx],MARGIN)
            loss=classification+(WEIGHT*rank if arm=='pair' else 0)
            loss.backward(); optimizer.step()
            if (step+1)%100==0:
                torch.cuda.synchronize()
                progress.append(dict(step=step+1,classification=float(classification.detach()),
                                     rank=float(rank.detach()),seconds=time.perf_counter()-tick))
                write(out/(arm+'-progress.json'),progress)
                print(json.dumps(dict(arm=arm,**progress[-1])),flush=True)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
        torch.save(model.state_dict(),out/(arm+'-model.pt'))
        scores=predict(model,data); np.save(out/(arm+'-scores.npy'),scores)
        results[arm]=dict(training_seconds=elapsed,scores=scores,
                          model_sha256=sha(out/(arm+'-model.pt')))
        del model,optimizer
    return results


def fit_scores(gt,scores,pairs):
    idx=np.array(sorted({i for p in pairs for i in (p['a'],p['b'])}))
    pred=scores>=0
    both=sum(pred[p['a']]==gt[p['a']] and pred[p['b']]==gt[p['b']] for p in pairs)
    changed=[p for p in pairs if gt[p['a']]!=gt[p['b']]]
    order=sum((scores[p['a']]-scores[p['b']])*(int(gt[p['a']])-int(gt[p['b']]))>0 for p in changed)
    return dict(metrics=metrics(gt[idx],pred[idx]),accuracy=float((pred[idx]==gt[idx]).mean()),
                both_correct=int(both),pairs=len(pairs),both_rate=both/max(1,len(pairs)),
                order_correct=int(order),changed_pairs=len(changed))


def run(args):
    out=args.output.resolve(); assert not out.exists() and out.is_relative_to((ROOT/'artifacts.local').resolve())
    out.mkdir(parents=True)
    (out/'source-snapshot').mkdir()
    for name in ('run_mz136_corridor_pair.py','mz136_corridor_pair.py','mz120_occupancy.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/'source-snapshot'/name)
    assert torch.cuda.is_available(); torch.set_num_threads(4); torch.manual_seed(SEED)
    rows,es,spec,data=load(args.capture,args.image_root or args.capture)
    gt=np.array([truth(e) for e in es],bool); pairs=pair_indices(rows,spec)
    if args.mode=='pilot':
        # 24 complete aligned pairs, four families. Consumed data, fit diagnosis only.
        selected=[]
        for family in sorted({p['family'] for p in pairs}):
            pp=[p for p in pairs if p['family']==family and p['pair_id'].endswith('pair0')]
            selected.extend(pp[::2][:6])
        assert len(selected)==24
    else:
        selected=[p for p in pairs if p['split']=='train']
        assert selected and all(p['split'] in ('train','dev','test') for p in pairs)
        groups={s:{p['scene_group'] for p in pairs if p['split']==s} for s in ('train','dev','test')}
        assert not groups['train']&groups['dev'] and not groups['train']&groups['test'] and not groups['dev']&groups['test']
    model=CorridorNet(); initial=copy.deepcopy(model.state_dict()); torch.save(initial,out/'initial.pt')
    probe=batch(data,np.array([p[k] for p in selected[:4] for k in ('a','b')]))
    gpu=copy.deepcopy(model).cuda().eval(); cpu=model.eval(); cp={k:v.cpu() for k,v in probe.items()}
    def g():
        with torch.no_grad():return gpu(probe)
    def c():
        with torch.no_grad():return cpu(cp)
    backend=select_backend('model-inference',cpu=BackendCandidate('torch-cpu','cpu',c,
        lambda v:DeviceObservation(v.device.type,'host CPU','torch '+torch.__version__)),
        gpu=BackendCandidate('torch-cuda','cuda',g,
        lambda v:DeviceObservation(v.device.type,torch.cuda.get_device_name(),'torch '+torch.__version__),torch.cuda.synchronize),
        record_path=out/'backend.json')
    assert backend['selected_device_type']=='cuda'
    del gpu,cpu,model,cp,probe
    write(out/'freeze.json',dict(mode=args.mode,seed=SEED,steps=args.steps,augment=not args.no_augment,margin=MARGIN,rank_weight=WEIGHT,
        score='logsumexp(CORE logits)-log(8); sigmoid not calibrated probability',
        parameters=sum(v.numel() for k,v in initial.items() if k!='cells'),
        source={n:sha(args.capture/n) for n in ('spec.json','raw.jsonl','evaluator.jsonl','receipt.json')},
        initial_sha256=sha(out/'initial.pt'),code={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz136_corridor_pair.py'),Path(__file__).with_name('mz120_occupancy.py')]},
        selected_pairs=selected,pairs=pairs,scope='CONSUMED_FIT_DIAGNOSIS' if args.mode=='pilot' else 'PROSPECTIVE_CONTROLLED_SIMULATION_DEVELOPMENT'))
    results=fit(data,gt,selected,initial,out,args.steps,augment=not args.no_augment)
    summary={arm:dict(fit=fit_scores(gt,r['scores'],selected),training_seconds=r['training_seconds'],model_sha256=r['model_sha256']) for arm,r in results.items()}
    learned=all(s['fit']['accuracy']>=.95 and s['fit']['both_rate']>=.90 for s in summary.values())
    write(out/'summary.json',dict(mode=args.mode,arms=summary,small_sample_fit_pass=learned,
        decision='PROCEED_SCENE_GROUP_COMPARISON' if learned else 'DIAGNOSE_FIT_BEFORE_CAPTURE_EXPANSION'))
    torch.cuda.empty_cache()
    write(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.json'),
        scores_sha256={a:sha(out/(a+'-scores.npy')) for a in results},
        freeze_sha256=sha(out/'freeze.json'),process_lifetime='foreground; CUDA resources released at exit'))
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--image-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('pilot','full'),default='pilot');p.add_argument('--steps',type=int,default=STEPS)
    p.add_argument('--no-augment',action='store_true')
    run(p.parse_args())
