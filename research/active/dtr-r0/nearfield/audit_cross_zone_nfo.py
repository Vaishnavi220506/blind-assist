"""Consumed 500-frame component anchor ceiling; no new predictions or models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from cross_zone_anchor_core import component_anchors,trace_sensor,zone_map

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
BASE=WORK/'ba-nfo-frozen-transfer500-20260919'
OUT=WORK/'ba-cross-zone-anchor-ceiling-20260920'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def domains(depth,boxes,values):
    known=np.isfinite(depth)&(depth>0);near=known&(depth<2.)
    mixed=np.zeros_like(known);small=mixed.copy();far=mixed.copy();pure=mixed.copy();inside=mixed.copy()
    for zi,(y0,x0,y1,x1) in enumerate(boxes):
        s=np.s_[y0:y1,x0:x1];inside[s]=True
        k,n=int(known[s].sum()),int(near[s].sum())
        if k and 0<n<k:
            mixed[s]=True
            if n/k<=.2:small[s]=True
        if np.isfinite(values[zi]) and values[zi]>=2.:
            far[s]=True
            if k and not n:pure[s]=True
    return near,dict(full=known,mixed=mixed&known,small_foreground=small&known,
        far_small=far&small&known,pure_far=pure&known,public_far=far&known,outside=known&~inside)


def counts(pred,truth,domain):
    return dict(tp=int((pred&truth&domain).sum()),fp=int((pred&~truth&domain).sum()),
        fn=int((~pred&truth&domain).sum()),tn=int((~pred&~truth&domain).sum()))


def opportunity(labels,zmap,possible,pure,all_possible):
    arrays={k:np.zeros(labels.shape,bool) for k in ('any_possible','any_pure','local_possible','local_pure','local_other_range','other_possible','other_pure')}
    for label,donors in possible.items():
        mask=labels==label;local=np.isin(zmap,list(donors));arrays['any_possible']|=mask
        arrays['local_possible']|=mask&local
        owned=np.isin(zmap,list(all_possible.get(label,set())))
        # Strict cross-zone-only: no observed local contributor at ANY range.
        arrays['local_other_range']|=mask&owned&~local
        arrays['other_possible']|=mask&~owned
        if label in pure:
            arrays['any_pure']|=mask
            arrays['local_pure']|=mask&np.isin(zmap,list(pure[label]))
            arrays['other_pure']|=mask&~owned
    return arrays


def main(out=OUT):
    out=Path(out);protocol=read(out/'protocol.json')
    assert not (out/'nfo-results.json').exists() and not (out/'nfo-frames.json').exists()
    for name,digest in protocol['code_hashes'].items():
        assert sha(Path(__file__).with_name(name))==digest,name
    for relative,digest in protocol['input_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
    sys.path.insert(0,str(ROOT/'tools'))
    from research_backend import BackendCandidate,DeviceObservation,select_backend
    select_backend('scalar-scoring',cpu=BackendCandidate('numpy-cv2-cpu','cpu',
        lambda:cv2.connectedComponents(np.zeros((4,4),np.uint8),connectivity=8),
        lambda _:DeviceObservation('cpu',platform.processor(),'NumPy/OpenCV',('CPU',))),
        cpu_reason='TASK_NOT_GPU_SUITABLE',capabilities={'scope':'CPU metadata and exact saved-input lineage only'},record_path=out/'backend.json')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    rows=read(BASE/'manifest.json');expected=read(BASE/'results.json')['metrics']['nfo']
    assert len(rows)==500 and all(r['split']=='val' for r in rows)
    with np.load(BASE/'predictions.npz',allow_pickle=False) as data:
        predictions=np.unpackbits(data['masks'][:,0],axis=1,bitorder='little').reshape(500,192,256).astype(bool)
    totals={};frames=[];component_totals=Counter();unknown=0;started=time.perf_counter()
    for i,row in enumerate(rows):
        path=WORK/'ba-nfo-20260919'/row['prepared'];assert sha(path)==row['sha256'],row['id']
        with np.load(path,allow_pickle=False) as source:
            depth,boxes,values=[source[k].copy() for k in ('depth','boxes','values')]
        traces=trace_sensor(depth,row['id'],boxes,values)
        truth,dm=domains(depth,boxes,values);unknown+=int((~dm['full']).sum())
        number,labels=cv2.connectedComponents(truth.astype(np.uint8),connectivity=8)
        possible,pure=component_anchors(labels,traces,2.)
        all_possible,_=component_anchors(labels,traces,float('inf'))
        zmap=zone_map(boxes,depth.shape);opp=opportunity(labels,zmap,possible,pure,all_possible)
        frame=dict(id=row['id'],family=row['family'],scene=row['scene'],domains={},components=[],
            exact_saved_return_replay=True,observed_returns=sum(t['observed'] for t in traces))
        for name,domain in dm.items():
            c=counts(predictions[i],truth,domain);fn=domain&truth&~predictions[i]
            op={key:int((fn&mask).sum()) for key,mask in opp.items()}
            op['no_possible_anchor']=int((fn&~opp['any_possible']).sum())
            op['outside_tof_with_pure_anchor']=int((fn&(zmap<0)&opp['any_pure']).sum())
            frame['domains'][name]=dict(baseline=c,missed_pixel_opportunity=op)
            totals.setdefault(name,dict(baseline=Counter(),missed_pixel_opportunity=Counter()))
            totals[name]['baseline'].update(c);totals[name]['missed_pixel_opportunity'].update(op)
        fn=dm['far_small']&truth&~predictions[i]
        for label in np.unique(labels[fn]):
            pixels=(labels==label)&fn
            missed_by_zone={int(z):int((pixels&(zmap==z)).sum()) for z in np.unique(zmap[pixels])}
            frame['components'].append(dict(label=int(label),total_component_pixels=int((labels==label).sum()),
                far_small_fn_pixels=int(pixels.sum()),missed_pixels_by_zone=missed_by_zone,
                possible_donor_zones=sorted(possible.get(int(label),set())),pure_donor_zones=sorted(pure.get(int(label),set()))))
            component_totals['missed_components']+=1
            component_totals['with_any_possible']+=int(int(label) in possible)
            component_totals['with_any_pure']+=int(int(label) in pure)
        frame['zone_lineage']=[dict(zone_id=t['zone_id'],observed=t['observed'],distance_m=t['distance_m'],reason=t['reason'],
            winner_bin=t['winner_bin'],contributing_label_counts={int(k):int(v) for k,v in zip(*np.unique(labels.ravel()[t['pixel_indices']],return_counts=True))}) for t in traces]
        frames.append(frame)
    for name,section in totals.items():
        for k,v in section['baseline'].items():
            assert v==expected[name][k],('Old baseline mismatch',name,k,v,expected[name][k])
        c=section['baseline'];op=section['missed_pixel_opportunity']
        assert op['other_possible']+op['local_possible']+op['local_other_range']+op['no_possible_anchor']==c['fn']
        section['fractions_of_original_fn']={k:v/c['fn'] if c['fn'] else None for k,v in op.items()}
        section['optimistic_recall_ceiling_with_other_pure']=(c['tp']+op['other_pure'])/(c['tp']+c['fn']) if c['tp']+c['fn'] else None
    with (out/'nfo-frames.json').open('x',encoding='utf-8') as stream:json.dump(frames,stream,indent=2,allow_nan=False)
    result=dict(status='COMPLETE',frames=500,split='CONSUMED_ORIGINAL_VAL500',threshold_m=2.,nfo_cutoff=.081,
        baseline_exact_replay=True,sensor_scalar_replay_frames=500,component_definition='8-connected reference-known depth<2m; can merge distinct objects; optimistic envelope',
        totals=totals,far_small_component_totals=dict(component_totals),unknown_reference_pixels=unknown,
        model_calls=0,new_captures=0,new_observations=0,elapsed_s=time.perf_counter()-started,
        frames_sha256=sha(out/'nfo-frames.json'),protocol_sha256=sha(out/'protocol.json'),
        warning='Privileged anchor availability only, not implemented RGB association, depth prediction, FPR improvement or real hardware evidence')
    with (out/'nfo-results.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','frames','far_small_component_totals','elapsed_s')}))
    print(json.dumps(result['totals']['far_small'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=OUT)
    main(parser.parse_args().output)
