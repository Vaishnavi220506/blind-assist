"""Fixed observable zone audit; no new alert policy or fitting."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from tof_corridor_calibration import score_frame

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/ba-core-workpoint-transfer-20260920'
OUT=ROOT/'artifacts.local/work/ba-onset-zone-audit-20260920'
T=.4071309640537889
TARGETS=('f0005','f0293')
NEIGHBORS=('f0004','f0005','f0006','f0292','f0293','f0294')
CODE=('audit_onset_zones_20260920.py','tof_corridor_calibration.py','ba_camera_corridor.py')
FEATURES=('valid_count','possible_count','definite_count','positive_zone_count','max_depth',
          'top1','top2_sum','top3_sum','total_mass','concentration','component_count',
          'largest_component_mass','largest_component_size')
DOMINANCE=('top1','total_mass','largest_component_mass','positive_zone_count','max_depth')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def components(masses):
    unseen={i for i,v in enumerate(masses) if v>0};result=[]
    while unseen:
        stack=[min(unseen)];unseen.remove(stack[0]);group=[]
        while stack:
            i=stack.pop();group.append(i);y,x=divmod(i,8)
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                j=yy*8+xx
                if 0<=yy<8 and 0<=xx<8 and j in unseen:unseen.remove(j);stack.append(j)
        result.append(dict(zones=sorted(group),size=len(group),mass=sum(masses[j] for j in group)))
    return sorted(result,key=lambda r:(-r['mass'],r['zones']))


def summarize(zones):
    masses=[z['effective_joint'] for z in zones];ordered=sorted(masses,reverse=True)
    cs=components(masses);total=sum(masses)
    f=dict(valid_count=sum(z['valid'] for z in zones),possible_count=sum(z['possible'] for z in zones),
        definite_count=sum(z['definite'] for z in zones),positive_zone_count=sum(v>0 for v in masses),
        max_depth=max((z['depth'] for z in zones if z['possible']),default=0),
        top1=ordered[0],top2_sum=sum(ordered[:2]),top3_sum=sum(ordered[:3]),total_mass=total,
        concentration=ordered[0]/total if total else None,component_count=len(cs),
        largest_component_mass=max((c['mass'] for c in cs),default=0),
        largest_component_size=max((c['size'] for c in cs),default=0))
    return dict(features=f,components=cs,positive_mask=[i for i,m in enumerate(masses) if m>0])


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    text=Path(__file__).with_name('ONSET_ZONE_AUDIT_PROTOCOL_20260920.md')
    with (OUT/'protocol-before-run.md').open('xb') as f:f.write(text.read_bytes())
    write(OUT/'protocol.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={n:sha(SOURCE/n) for n in ('observations.json','observation-seal.json','predictions.json',
              'prediction-seal.json','frame-results.json','evaluation-seal.json','private-lineage.json','spec.json','capture/evaluator/geometry.json')},
        code={n:sha(Path(__file__).with_name(n)) for n in CODE},protocol_text_sha256=sha(OUT/'protocol-before-run.md'),
        scope='CONSUMED_OBSERVABLE_DIAGNOSTIC_WITH_SEPARATE_EVALUATOR_ATTRIBUTION',targets=TARGETS,
        features=FEATURES,dominance_tuple=DOMINANCE,backend='TASK_NOT_GPU_SUITABLE'))
    print('FROZEN observable audit; no candidate')


def verify():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol-before-run.md')==p['protocol_text_sha256']
    for n,h in p['inputs'].items():assert sha(SOURCE/n)==h,n
    for n,h in p['code'].items():assert sha(Path(__file__).with_name(n))==h,n


def extract():
    verify()
    assert len(components([1 if i in (7,8) else 0 for i in range(64)]))==2
    assert len(components([1 if i in (7,15) else 0 for i in range(64)]))==1
    original={r['id']:r for r in read(SOURCE/'predictions.json')};observations=read(SOURCE/'observations.json')
    assert len(original)==len(observations)==432
    for seal in ('observation-seal.json','prediction-seal.json'):
        for n,h in read(SOURCE/seal)['hashes'].items():assert sha(SOURCE/n)==h,n
    rows=[]
    for obs in observations:
        assert sha(SOURCE/obs['path'])==obs['sha256']
        with np.load(SOURCE/obs['path'],allow_pickle=False) as data:boxes,values=data['boxes'],data['values']
        assert boxes.shape==(64,4) and values.shape==(64,)
        scored=score_frame(boxes,values);saved=original[obs['id']]
        for k in ('score','anchors','zone_scores'):assert scored[k]==saved[k],(obs['id'],k)
        assert scored['baseline']==saved['raw']
        anchors={r['zone']:r for r in scored['anchors']};scores={r['zone']:r for r in scored['zone_scores']}
        zones=[]
        for z in range(64):
            a=anchors.get(z);s=scores.get(z);valid=a is not None
            zones.append(dict(zone=z,valid=valid,range_m=float(values[z]) if valid else None,box=boxes[z].tolist(),
                interval_m=a['interval_m'] if valid else None,possible=a['possible'] if valid else False,
                definite=a['definite'] if valid else False,depth=s['depth'] if valid else None,
                angular=s['angular_given_depth'] if valid else None,joint=s['joint'] if valid else None,
                effective_joint=s['joint'] if valid and a['possible'] else 0))
        summary=summarize(zones);assert summary['features']['top1']==scored['score']
        rows.append({**{k:obs[k] for k in ('id','clip_id','frame_in_clip','time_s')},'zones':zones,**summary})
    write(OUT/'observable-features.json',rows)
    write(OUT/'feature-seal.json',dict(status='COMPLETE',frames=432,protocol_sha256=sha(OUT/'protocol.json'),
        hashes={'observable-features.json':sha(OUT/'observable-features.json')},original_parity=True,label_joined=False))
    print('FEATURES_SEALED 432x64; original score/anchors/raw parity')


def evaluate():
    verify();seal=read(OUT/'feature-seal.json')
    for n,h in seal['hashes'].items():assert sha(OUT/n)==h
    features=read(OUT/'observable-features.json');labels={r['id']:r for r in read(SOURCE/'frame-results.json')}
    rows=[]
    for f in features:
        l=labels[f['id']];assert all(l[k]==f[k] for k in ('clip_id','time_s','frame_in_clip'))
        rows.append({**f,**{k:l[k] for k in ('truth','boundary','layout_relation','layer','background','type_id')},
                     'strong':l['predictions']['candidate']['alert']})
    core=lambda r:r['layout_relation']!='BOUNDARY'
    groups=dict(core_positive=lambda r:core(r) and r['truth'],other_positive=lambda r:core(r) and r['truth'] and r['id'] not in TARGETS,
        head_positive=lambda r:core(r) and r['truth'] and r['layer']=='HEAD',
        strong_head_positive=lambda r:core(r) and r['truth'] and r['layer']=='HEAD' and r['strong'],
        body_positive=lambda r:core(r) and r['truth'] and r['layer']=='BODY',
        core_negative=lambda r:core(r) and not r['truth'],inside_negative=lambda r:r['layout_relation']=='INSIDE' and not r['truth'],
        outside_negative=lambda r:r['layout_relation']=='OUTSIDE',boundary=lambda r:r['layout_relation']=='BOUNDARY')
    distribution={}
    for name,mask in groups.items():
        selected=[r for r in rows if mask(r)];stats={}
        for k in FEATURES:
            vals=[r['features'][k] for r in selected if r['features'][k] is not None]
            stats[k]=dict(defined=len(vals),min=min(vals) if vals else None,median=float(np.median(vals)) if vals else None,max=max(vals) if vals else None)
        distribution[name]=dict(n=len(selected),features=stats)
    neg=[r for r in rows if groups['core_negative'](r)];assert len(neg)==210 and distribution['core_positive']['n']==78
    comparisons={}
    for target in [r for r in rows if r['id'] in TARGETS]:
        ranks={}
        for k in FEATURES:
            v=target['features'][k]
            vals=[r['features'][k] for r in neg if r['features'][k] is not None]
            ranks[k]=dict(value=v,negative_defined=len(vals),negative_ge=sum(x>=v for x in vals) if v is not None else None,
                          negative_le=sum(x<=v for x in vals) if v is not None else None)
        same=[r['id'] for r in neg if r['positive_mask']==target['positive_mask']]
        dominate=[r['id'] for r in neg if all(r['features'][k]>=target['features'][k] for k in DOMINANCE)]
        comparisons[target['id']]=dict(negative_ranks=ranks,same_positive_mask_negative_ids=same,
            support_magnitude_dominating_negative_ids=dominate,dominance_tuple=DOMINANCE,
            identical_mask_not_identical_ranges=True)
    write(OUT/'labeled-audit.json',rows)
    write(OUT/'results.json',dict(distributions=distribution,targets=comparisons,
        selected_frames=[r for r in rows if r['id'] in NEIGHBORS],scope='FEATURE_DIAGNOSTIC_NOT_CLASSIFIER_OR_INFORMATION_CEILING',
        original_predictions_unchanged=True))
    write(OUT/'audit-seal.json',dict(status='COMPLETE',hashes={n:sha(OUT/n) for n in ('results.json','labeled-audit.json','feature-seal.json')}))
    print(json.dumps({i:dict(same_mask_neg=len(c['same_positive_mask_negative_ids']),dominating_neg=len(c['support_magnitude_dominating_negative_ids']),features={k:c['negative_ranks'][k] for k in ('top1','total_mass','largest_component_mass','positive_zone_count')}) for i,c in comparisons.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('freeze','extract','evaluate'));globals()[p.parse_args().stage]()
