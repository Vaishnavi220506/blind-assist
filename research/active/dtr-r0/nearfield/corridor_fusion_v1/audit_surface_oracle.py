"""Independent raw-lineage and frozen-feature audit; imports no oracle helpers."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[5]
WORK = ROOT/'artifacts.local/work'
OUT = WORK/'corridor-surface-oracle-20260917'
CAP = WORK/'corridor-depth-confirmation-recovery-20260917/source/returned-v1/capture-v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def overlap(p, width=.3):
    p = np.asarray(p)
    return bool(np.all(p[:, 1] >= [.2, -width, .4]) and
                np.all(p[:, 0] <= [3.6, width, 2.05]))


def counts(y, p):
    return dict(frames=len(y), TP=int(sum(y&p)), FP=int(sum(~y&p)),
                FN=int(sum(y&~p)), TN=int(sum(~y&~p)))


def main():
    frozen = read(OUT/'freeze.json')
    for p, h in frozen['bindings'].items():
        assert digest(Path(p)) == h, p
    seal = read(OUT/'prediction-seal.json')
    for name, key in [('scores.npz','scores_sha256'), ('oracle-features.npz','features_sha256'),
                      ('oracle-returns.json','returns_sha256'), ('freeze.json','freeze_sha256')]:
        assert digest(OUT/name) == seal[key]
    ids = frozen['ids']
    def ordered(name):
        rows = {r['id']:r for r in map(json.loads, (CAP/name).read_text().splitlines())}
        return [rows[i] for i in ids]
    rows, es = ordered('raw.jsonl'), ordered('evaluator.jsonl')
    records = read(OUT/'oracle-returns.json'); cases = read(OUT/'cases.json')
    basez = np.load(WORK/'corridor-intrusion-20260917/diagnosis/new-base-features.npz')
    variants = np.load(OUT/'oracle-features.npz'); scores = np.load(OUT/'scores.npz')
    assert list(basez['ids']) == list(variants['ids']) == list(scores['ids']) == ids
    base = basez['base']
    names = read(WORK/'mz143-corridor-evidence-20260916/run-v1/feature-names.json')
    names = names['sensor_names']+names['geometry_names']; index = {n:i for i,n in enumerate(names)}
    support = np.array(['.support_' in n for n in names]); assert support.sum() == 768
    model = pickle.loads((WORK/'corridor-depth-e1-20260917/run-v2/A-model.pkl').read_bytes())
    saved = read(WORK/'corridor-depth-confirmation-20260917/evaluation-v1/predictions.json')
    assert np.array_equal(model.predict_proba(base)[:,1], scores['A'])
    assert np.array_equal(scores['A'], [r['A_score'] for r in saved])
    for arm, key in [('A_sampled_support','sampled'), ('A_full_surface_support','full')]:
        assert np.array_equal(base[:,~support], variants[key][:,~support])
        assert np.array_equal(model.predict_proba(variants[key])[:,1], scores[arm])
    total = Counter(); truths=[]; states=[]; point_flags=[]; face_flags=[]; hull_flags=[]; diagnoses=[]
    for fi, (row,e,rec,case) in enumerate(zip(rows,es,records,cases)):
        assert row['id'] == e['id'] == rec['id'] == case['id']
        origin=np.array(e['body_origin_m']); objects={o['name']:o for o in e['native_bounds']}
        bounds={n:np.stack([np.array(o['center_m'])-o['extent_m']-origin,
                            np.array(o['center_m'])+o['extent_m']-origin],-1) for n,o in objects.items()}
        eligible={n for n,b in bounds.items() if overlap(b)}
        truth=bool(eligible); state='positive' if any(overlap(b,.25) for b in bounds.values()) else (
            'boundary' if any(overlap(b,.35) for b in bounds.values()) else 'negative')
        assert truth == case['truth'] and state == case['state']
        truths.append(truth); states.append(state)
        native={z['zone_id']:z for z in e['zonal_tof_native']}
        saved_slots={(s['zone'],s['slot']):s for s in rec['slots']}
        reached_point=reached_face=reached_hull=False; returned=set(); allowed=[]
        for zone in row['tof_zones']:
            for si,target in enumerate(zone['targets']):
                zid=zone['zone_id']; s=saved_slots[(zid,si)]; total['slots']+=1
                usable=bool(row['tof_packet_received'] and target['status'] in ('SIM_VALID','SIM_MERGED') and
                    target['distance_m']>0 and target['range_noise_sigma_m']>=0 and target['signal_strength_proxy']>=0)
                assert usable == s['usable']; assert usable and s['completed']
                total['usable']+=1
                nz=native[zid]; lin=[v for v in nz['returned_lineage'] if v['target_index']==si]
                assert len(lin)==1 and lin[0]['hit_indices']
                rays={h['subray']:h for h in nz['private_rays']}; hits=[rays[k] for k in lin[0]['hit_indices']]
                pieces={}; points=[]
                for h in hits:
                    name=h['actor_id'].rsplit('/',1)[-1]; returned.add(name)
                    p=np.array(h['hit_point_m'])-origin; b=bounds[name]
                    assert np.all(p>=b[:,0]-1e-5) and np.all(p<=b[:,1]+1e-5)
                    matches=[(a,side) for a in range(3) for side in range(2) if abs(p[a]-b[a,side])<=1e-5]
                    assert len(matches)==1
                    axis, side=matches[0]; face=b.copy(); face[axis,:]=b[axis,side]
                    pieces[(h['actor_id'],f'{axis}:{"MIN" if side==0 else "MAX"}')]=face
                    points.append(p)
                faces=np.array([pieces[k] for k in sorted(pieces)]); points=np.array(points)
                assert np.array_equal(faces, s['faces']) and np.array_equal(points,s['points'])
                assert sorted(pieces)==sorted(set(tuple(k) for k in s['surface_keys']))
                sp=np.stack([points.min(0),points.max(0)],-1)
                fp=np.stack([faces[:,:,0].min(0),faces[:,:,1].max(0)],-1)
                ix=[index[f'zone{zid:02d}.slot{si}.support_{axis}_{side}'] for axis in 'xyz' for side in ['lo','hi']]
                allowed.extend(ix)
                assert np.array_equal(variants['sampled'][fi,ix],sp.flatten().astype(np.float32))
                assert np.array_equal(variants['full'][fi,ix],fp.flatten().astype(np.float32))
                pbits=[overlap(np.stack([p,p],-1)) for p in points]
                fb=any(overlap(f) for f in faces); hb=overlap(fp)
                assert fb==s['face_union_possible'] and hb==s['full_hull_possible']
                assert any(pbits)==s['sampled_point_possible']
                for p in points:
                    assert any(np.all(p>=f[:,0]-1e-5) and np.all(p<=f[:,1]+1e-5) for f in faces)
                total['contributors']+=len(points); total['corridor_contributors']+=sum(pbits)
                total['hull_bridges']+=hb!=fb; total['multi_face']+=len(faces)>1
                total['multi_actor']+=len({k[0] for k in pieces})>1
                reached_point |= any(pbits); reached_face |= fb; reached_hull |= hb
        mask=np.ones(len(names),bool); mask[allowed]=False
        for key in ['sampled','full']:assert np.array_equal(base[fi,mask],variants[key][fi,mask])
        assert reached_point==rec['sampled_point_reachable'] and reached_face==rec['full_face_reachable']
        assert rec['fallback_slots']==0
        point_flags.append(reached_point); face_flags.append(reached_face); hull_flags.append(reached_hull)
        private={h['actor_id'].rsplit('/',1)[-1] for z in native.values() for h in z['private_rays'] if 'actor_id' in h}
        category=('FULL_FACE_PRESENT_SAMPLED_POINT_PRESENT' if reached_point else 'UNSAMPLED_FULL_FACE_COMPLETION_OPPORTUNITY') if reached_face else (
            'RETURNED_CORRIDOR_OBJECT_BUT_NO_RESOLVED_INTERSECTING_FACE' if returned&eligible else
            'TOF_PACKET_MISSING' if not row['tof_packet_received'] else
            'PRIVATE_OBJECT_HIT_WITHOUT_USABLE_RETURN' if private&eligible else 'NO_NATIVE_RAY_HIT_ON_CORRIDOR_OBJECT')
        assert category == case['diagnosis']['category']
        diagnoses.append(category)
        total['zero_usable_frames']+=not allowed
    y=np.array(truths); clear=np.array(states)!='boundary'; a=scores['A']>=frozen['threshold']
    assert counts(y[clear],a[clear])==dict(frames=216,TP=95,FP=27,FN=13,TN=81)
    for arm in ['A_sampled_support','A_full_surface_support']:assert np.array_equal(a,scores[arm]>=frozen['threshold'])
    controls={}
    for name,p in [('sampled_points',point_flags),('exact_face_union',face_flags),('face_hull',hull_flags)]:
        p=np.array(p); controls[name]={k:counts(y[ix],p[ix]) for k,ix in [('strict',np.ones(len(y),bool)),('clear',clear),('boundary',~clear)]}
        controls[name]['clear_A_TP_lost']=int(sum(clear&y&a&~p))
        controls[name]['clear_A_FN_recovered']=int(sum(clear&y&~a&p))
    fn=[dict(id=ids[i],family=es[i]['family'],category=diagnoses[i]) for i in np.flatnonzero(clear&y&~a)]
    result=dict(status='PASS',frames=len(ids),raw_native_reconstruction=True,non_support_bitwise_equal=True,
        baseline_score_bitwise_equal=True,all_oracle_scores_reproduced=True,all_alerts_unchanged=True,
        native_counts=dict(total),readout_only_information_controls=controls,clear_FN=fn,
        clear_FN_categories=dict(Counter(r['category'] for r in fn)),
        warning='Reachability is evaluator-only information control, not frozen-A output, achievable upper bound, or deployable method',
        audited_seal_sha256=digest(OUT/'prediction-seal.json'))
    (OUT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(4):main()
