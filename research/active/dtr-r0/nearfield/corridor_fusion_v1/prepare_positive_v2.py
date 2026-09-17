"""Append complete consumed MZ146/158 groups to authenticated public cache."""
import hashlib
import json
import pickle
from collections import defaultdict,Counter
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path.insert(0,str(HERE.parent))
from public_return_tokens import encode_tokens
from mz171_return_labels import make_witness_labels
from tolerance_eval import geometry,classify
WORK=ROOT/'artifacts.local/work'
OLD=WORK/'corridor-public-positive-20260917/preparation'
OUT=WORK/'corridor-public-positive-v2-20260917/preparation'
CAPS={'anchor':'mz136-corridor-pair-20260914','old':'mz170-mean-confirmation-20260916',
      'new':'corridor-depth-confirmation-recovery-20260917','mz146':'mz146-fresh-corridor-confirmation-20260916',
      'mz158':'mz158-crossview-agreement-20260916'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'data-seal.json').exists()
    bindings={}
    def bind(p,h=None):
        actual=sha(p);assert h is None or actual==h,str(p);bindings[str(p)]=actual
    ds=read(OLD/'data-seal.json');bind(OLD/'data-seal.json')
    for p,h in ds['bindings'].items():bind(p,h)
    for p,h in ds['outputs'].items():bind(OLD/p,h)
    for p in [Path(__file__),HERE/'public_return_tokens.py',HERE.parent/'mz171_return_labels.py',HERE/'tolerance_eval.py']:bind(p)
    z=np.load(OLD/'public-tokens.npz');lab=np.load(OLD/'offline-targets.npz')
    tokens=list(z['tokens']);valid=list(z['valid']);targets=list(lab['target']);known=list(lab['known']);meta=read(OLD/'metadata.json')
    assert len(meta)==768 and list(z['ids'])==[m['id'] for m in meta]
    threshold=ds['A_threshold'];assert threshold==.3917890013717321
    ap=WORK/'corridor-depth-e1-20260917/run-v2/A-model.pkl'
    bind(ap,'d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef');model=pickle.loads(ap.read_bytes())
    frame_specs={};score_receipts={}
    for cohort,stem in CAPS.items():
        cap=WORK/stem/'source/returned-v1/capture-v1'
        receipt=read(cap/'receipt.json');assert receipt['status']=='PASS';bind(cap/'receipt.json')
        bind(cap/'spec.json',receipt['spec_sha256']);spec=read(cap/'spec.json')
        for name in ['raw.jsonl','evaluator.jsonl']:bind(cap/name,receipt['hashes'][name])
        fs=[f for f in spec['frames'] if cohort!='anchor' or f['split']=='train']
        frame_specs.update({f['id']:f for f in fs})
        if cohort not in ['mz146','mz158']:continue
        assert len(fs)==288 and all(f['split']=='confirmation' for f in fs)
        rows=[json.loads(s) for s in (cap/'raw.jsonl').read_text().splitlines()]
        es=[json.loads(s) for s in (cap/'evaluator.jsonl').read_text().splitlines()]
        ids=[r['id'] for r in rows];assert ids==[e['id'] for e in es] and set(ids)=={f['id'] for f in fs}
        frames={f['id']:f for f in fs}
        if cohort=='mz146':
            cache=WORK/'corridor-depth-specialist-20260917'
            bind(cache/'meta-feature-seal.json');bind(cache/'meta-inputs.npz',read(cache/'meta-feature-seal.json')['sha256'])
            zz=np.load(cache/'meta-inputs.npz');base=zz['base'];saved=zz['A']
            # Per-frame authenticated RGB keys establish the aggregate cache order.
            for i,r in enumerate(rows):
                entry=np.load(cache/'meta-features'/(r['id']+'.npz'))
                assert str(entry['rgb_sha256'])==receipt['hashes'][r['rgb_path']]
                assert np.array_equal(base[i],entry['base'])
        else:
            cache=WORK/stem/'learned-v1';done=read(cache/'completion.json');assert done['status']=='PASS'
            bind(cache/'completion.json');bind(cache/'prediction-seal.json',done['prediction_seal_sha256'])
            ps=read(cache/'prediction-seal.json');bind(cache/'predictions.json',ps['predictions_sha256'])
            bind(cache/'input-seal.json',ps['input_seal_sha256'])
            bind(cache/'features.npz',ps['context']['features_sha256'])
            pred=read(cache/'predictions.json');assert [p['id'] for p in pred]==ids
            base=np.load(cache/'features.npz')['static'];saved=np.array([p['scores']['static'] for p in pred])
        score=model.predict_proba(base)[:,1];np.testing.assert_array_equal(score,saved)
        score_receipts[cohort]=dict(frozen_A_bitwise_score_parity=True,frames=288)
        yaw=0.;episode=None
        for i,(row,e) in enumerate(zip(rows,es)):
            if row['episode_id']!=episode:yaw=0.
            if row['imu_valid']:yaw+=row['delta_yaw']
            episode=row['episode_id'];pub=encode_tokens(row,yaw);labels=make_witness_labels(row,e)
            assert not (labels['known']&~pub['valid']).any() and not labels['known'][128:].any()
            f=frames[row['id']];g=geometry(e)
            meta.append(dict(id=row['id'],cohort=cohort,family=f['family'],group=f['scene_group'],
                scene_index=int(f['scene_group'].rsplit('_scene',1)[1]),episode_id=row['episode_id'],time_s=row['time_s'],
                packet=row['tof_packet_received'],truth=g['strict'],stratum=classify(g,.05),A_score=float(score[i]),A=bool(score[i]>=threshold)))
            tokens.append(pub['tokens']);valid.append(pub['valid']);targets.append(labels['target']);known.append(labels['known'])
    tokens=np.stack(tokens);valid=np.stack(valid);targets=np.stack(targets);known=np.stack(known)
    assert tokens.shape==(1344,132,21) and len(meta)==len({m['id'] for m in meta})==1344
    assert all(m['A'] is None and m['A_score'] is None for m in meta if m['cohort']=='anchor')
    assert np.array_equal(tokens[:768],z['tokens']) and np.array_equal(targets[:768],lab['target'])
    groups=defaultdict(list);duplicates=defaultdict(list)
    for i,m in enumerate(meta):
        groups[m['group']].append(i)
        h=hashlib.sha256(tokens[i].tobytes()+valid[i].tobytes()).hexdigest();duplicates[h].append(m['id'])
    assert len(groups)==112
    group_signatures=defaultdict(list);group_inventory=[]
    for group,ii in groups.items():
        assert len(ii)==12 and len({meta[i]['episode_id'] for i in ii})==2 and len({meta[i]['cohort'] for i in ii})==1
        payload=[{k:frame_specs[meta[i]['id']].get(k) for k in ['camera','body_origin_m','objects','time_s']} for i in ii]
        signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();group_signatures[signature].append(group)
        opportunity=[]
        for i in ii:
            m=meta[i];witness=bool(((targets[i]>0)&known[i]).any())
            if m['A'] is not None and m['truth'] and not m['A']:
                opportunity.append(dict(id=m['id'],clear=m['stratum']!='boundary',sampled_witness=witness))
        group_inventory.append(dict(group=group,cohort=meta[ii[0]]['cohort'],family=meta[ii[0]]['family'],scene_index=meta[ii[0]]['scene_index'],
            frames=len(ii),geometry_camera_hash=signature,A_FN=opportunity,clear_supported_A_FN=sum(o['clear'] and o['sampled_witness'] for o in opportunity)))
    counts={c:dict(frames=sum(m['cohort']==c for m in meta),groups=sum(g['cohort']==c for g in group_inventory),
        clear_supported_A_FN=sum(g['clear_supported_A_FN'] for g in group_inventory if g['cohort']==c)) for c in CAPS}
    for c in ['mz146','mz158']:assert counts[c]['clear_supported_A_FN']==0
    inventory=dict(status='PASS',cohorts=counts,groups=group_inventory,score_authentication=score_receipts,
        identical_public_token_sets=[v for v in duplicates.values() if len(v)>1],identical_geometry_camera_groups=[v for v in group_signatures.values() if len(v)>1],
        duplicate_scope='Exact float32 full-token+valid bytes; complete group camera/body/objects/time signatures. Does not prove statistical independence.',
        original_test48_decoded=False,anchor_frame_loss_eligible=False,new_clear_positive_opportunities=0,
        authority='COMPLETE_CONSUMED_DEVELOPMENT_COHORTS_NO_FRAME_SELECTION_NO_NEW_CAPTURE')
    np.savez_compressed(OUT/'public-tokens.npz',tokens=tokens,valid=valid,ids=[m['id'] for m in meta])
    np.savez_compressed(OUT/'offline-targets.npz',target=targets,known=known)
    write(OUT/'metadata.json',meta);write(OUT/'inventory.json',inventory)
    assert all(sha(p)==h for p,h in bindings.items())
    write(OUT/'data-seal.json',dict(bindings=bindings,A_threshold=threshold,outputs={n:sha(OUT/n) for n in ['public-tokens.npz','offline-targets.npz','metadata.json','inventory.json']},
        authority='AUTHENTICATED_PUBLIC_TOKENS_SEPARATE_OFFLINE_LABELS_CONSUMED1344'))
    print(json.dumps(dict(status='PASS',cohorts=counts,duplicate_token_sets=len(inventory['identical_public_token_sets']),duplicate_scene_groups=len(inventory['identical_geometry_camera_groups'])),indent=2))

if __name__=='__main__':
    with threadpool_limits(4):main()
