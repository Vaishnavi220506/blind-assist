"""One frozen RGB component/multiple observed-zone association diagnostic."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from tof_lateral_core import lateral_relation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921'
AXIS=ROOT/'artifacts.local/work/ba-axis-evidence-20260921'
OUT=ROOT/'artifacts.local/work/ba-rgb-multizone-20260921'
PROTOCOL=HERE/'RGB_MULTIZONE_PROTOCOL_20260921.md'
PAIRS={'selection':[(702,678),(711,687)],'evaluation':[(486,462),(487,463),(494,470),(1063,1039),(1070,1046)]}
FOCAL=640/(2*np.tan(np.deg2rad(50)))


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def verify():
    for p,h in read(OUT/'protocol.json')['inputs'].items():assert sha(ROOT/p)==h,p
    old=ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
    assert not any((old/n).exists() for n in ('test-start.json','test-logits.npy','test-metrics.json'))


def seal(name,files):
    write(OUT/name,dict(protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(OUT/n) for n in files}))


def connected_pair(zones):
    return any((z%8<7 and z+1 in zones) or z+8 in zones for z in zones)


def extent_relation(x0,x1,interval):
    # Complete component pixel-edge extent, with one native-pixel pad.
    return lateral_relation(((max(0,x0-1)-320)/FOCAL,(min(640,x1+1)-320)/FOCAL),interval)


def probe(rgb,boxes,zone_records):
    assert rgb.shape==(360,640,3)
    # Native pixel centers lying inside the public low-resolution box edges.
    native=[]
    for y0,x0,y1,x1 in boxes:
        native.append([int(np.ceil(y0*360/192-.5)),int(np.ceil(x0*640/256-.5)),
                       int(np.ceil(y1*360/192-.5)),int(np.ceil(x1*640/256-.5))])
    yy0,xx0=min(b[0] for b in native),min(b[1] for b in native)
    yy1,xx1=max(b[2] for b in native),max(b[3] for b in native)
    gray=cv2.cvtColor(rgb[yy0:yy1,xx0:xx1],cv2.COLOR_BGR2GRAY)
    threshold,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labels=np.zeros((360,640),np.int32);components=[];offset=0
    for polarity in (0,255):
        count,local,stats,_=cv2.connectedComponentsWithStats((binary==polarity).astype(np.uint8),connectivity=8)
        for label in range(1,count):
            x,y,w,h,area=map(int,stats[label]);cid=offset+label
            mask=local==label
            labels[yy0:yy1,xx0:xx1][mask]=cid
            clipped=x==0 or y==0 or x+w==gray.shape[1] or y+h==gray.shape[0]
            components.append(dict(component=cid,polarity=polarity,area=area,
                bbox=[x+xx0,y+yy0,x+w+xx0,y+h+yy0],clipped=clipped))
        offset+=count-1
    eligible=[]
    for c in components:
        anchors=[]
        if not c['clipped']:
            for z,b in enumerate(native):
                if zone_records[z]['valid'] and np.all(labels[b[0]:b[2],b[1]:b[3]]==c['component']):anchors.append(z)
        intervals=[zone_records[z]['interval_m'] for z in anchors]
        intersection=[max((i[0] for i in intervals),default=0.),min((i[1] for i in intervals),default=0.)]
        qualifies=bool(len(anchors)>=2 and connected_pair(set(anchors)) and intersection[0]<=intersection[1])
        c.update(anchors=anchors,common_interval=intersection if intervals else None,eligible=qualifies)
        if qualifies:
            c['envelope']=[min(i[0] for i in intervals),max(i[1] for i in intervals)];eligible.append(c)
    sample_y=np.floor((np.arange(192)+.5)*360/192).astype(int)
    sample_x=np.floor((np.arange(256)+.5)*640/256).astype(int)
    sampled=labels[sample_y[:,None],sample_x[None,:]]
    queries=[]
    for z,r in enumerate(zone_records):
        if not(r['valid'] and r['depth_state']=='CONTAINED' and r['possible'] and r['horizontal_relation']=='CROSSING'):continue
        candidates=[];y0,x0,y1,x1=boxes[z]
        for c in eligible:
            overlap=int(np.sum(sampled[y0:y1,x0:x1]==c['component']))
            if overlap<4 or max(r['interval_m'][0],c['common_interval'][0])>min(r['interval_m'][1],c['common_interval'][1]):continue
            envelope=[min(r['interval_m'][0],c['envelope'][0]),max(r['interval_m'][1],c['envelope'][1])]
            candidates.append(dict(component=c['component'],sampled_overlap=overlap,interval_envelope=envelope,
                relation=extent_relation(c['bbox'][0],c['bbox'][2],envelope)))
        queries.append(dict(zone=z,candidates=candidates,relation=candidates[0]['relation'] if len(candidates)==1 else 'UNKNOWN'))
    return dict(otsu_threshold=float(threshold),footprint=[xx0,yy0,xx1,yy1],components=components,queries=queries),labels


def extract():
    assert not OUT.exists(),'No overwrite or retry'
    ids=read(SOURCE/'identities.json');indices=sorted({i for pairs in PAIRS.values() for p in pairs for i in p})
    files=[SOURCE/n for n in ('observations.npz','identities.json','private-lineage.json','source-admission.json')]
    files += [AXIS/n for n in ('protocol.json','public-seal.json','public-descriptors.json','analysis-seal.json','joined-frame-results.json')]
    files += [SOURCE/ids[i]['rgb_path'] for i in indices]
    files += [Path(__file__),PROTOCOL,HERE/'tof_lateral_core.py']
    for name in ('public-seal.json','analysis-seal.json'):
        s=read(AXIS/name);assert s['protocol_sha256']==sha(AXIS/'protocol.json')
        for n,h in s['hashes'].items():assert sha(AXIS/n)==h
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',dict(id='ba-rgb-multizone-20260921',pairs=PAIRS,scope='CONSUMED_DIAGNOSTIC',
        time_utc=datetime.now(timezone.utc).isoformat(),inputs={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        backend='CPU',reason='TASK_NOT_GPU_SUITABLE',original_test_activated=False,alerts_changed=False))
    (OUT/'protocol-before-run.md').write_bytes(PROTOCOL.read_bytes())
    public={r['index']:r for r in read(AXIS/'public-descriptors.json')}
    with np.load(SOURCE/'observations.npz') as obs:boxes=obs['boxes']
    results=[];files=[]
    for i in indices:
        image_path=SOURCE/ids[i]['rgb_path'];assert sha(image_path)==ids[i]['rgb_sha256']
        data,labels=probe(cv2.imread(str(image_path)),boxes,public[i]['zones'])
        results.append(dict(id='transfer:'+ids[i]['id'],index=i,**data))
        name=f'regions-{i:04d}.npz';np.savez_compressed(OUT/name,labels=labels);files.append(name)
    write(OUT/'public-results.json',results)
    write(OUT/'extraction-receipt.json',dict(evaluator_values_read=False,private_lineage_hashed_only=True,
        raw_rgb_hashes_checked=14,frames=len(results),segmentation_recipes=1,parameter_retries=0))
    seal('public-seal.json',files+['public-results.json','extraction-receipt.json']);verify()
    print('SEALED_PUBLIC',len(results),'frames')


def evaluate():
    verify();s=read(OUT/'public-seal.json')
    assert s['protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in s['hashes'].items():assert sha(OUT/n)==h
    public={r['index']:r for r in read(OUT/'public-results.json')}
    truth={r['index']:r for r in read(AXIS/'joined-frame-results.json')}
    assignments=[(i,q) for i,r in public.items() for q in r['queries'] if len(q['candidates'])==1]
    lineage=read(SOURCE/'private-lineage.json') if assignments else None
    rows=[]
    sy=np.floor((np.arange(192)+.5)*360/192).astype(int)
    sx=np.floor((np.arange(256)+.5)*640/256).astype(int)
    for role,pairs in PAIRS.items():
        for neg,pos in pairs:
            assert truth[neg]['base_group_id']==truth[pos]['base_group_id'] and truth[neg]['time_s']==truth[pos]['time_s']
            assert not truth[neg]['truth'] and truth[pos]['truth']
            for i in (neg,pos):
                r=public[i];q=r['queries'];coverage=[]
                if any(len(x['candidates'])==1 for x in q):
                    with np.load(OUT/f'regions-{i:04d}.npz') as a:labels=a['labels']
                    for x in q:
                        if len(x['candidates'])!=1:continue
                        pixels=np.asarray(lineage[i]['traces'][x['zone']]['pixel_indices'],int)
                        cid=x['candidates'][0]['component']
                        kept=int(np.sum(labels[sy[pixels//256],sx[pixels%256]]==cid))
                        coverage.append(dict(zone=x['zone'],contributors=len(pixels),covered=kept,complete=kept==len(pixels)))
                rows.append(dict(id=r['id'],role=role,truth=truth[i]['truth'],group=truth[i]['base_group_id'],
                    components=len(r['components']),unclipped=sum(not c['clipped'] for c in r['components']),
                    maximum_anchor_count=max((len(c['anchors']) for c in r['components']),default=0),
                    eligible_components=sum(c['eligible'] for c in r['components']),queries=len(q),
                    assigned=sum(len(x['candidates'])==1 for x in q),unknown=sum(x['relation']=='UNKNOWN' for x in q),
                    outside=sum(x['relation']=='OUTSIDE' for x in q),native_corridor_samples=truth[i]['native_target_corridor_samples'],
                    assignment_coverage=coverage))
    useful=all(any(r['role']==role and not r['truth'] and r['queries']>0 and r['outside']==r['queries'] for r in rows) for role in PAIRS)
    useful=useful and not any(r['truth'] and r['outside'] for r in rows) and all(c['complete'] for r in rows for c in r['assignment_coverage'])
    result=dict(status='COMPLETE',scope='CONSUMED_DIAGNOSTIC',frames=rows,useful_local_feasibility=useful,
        assignments=len(assignments),private_lineage_values_read=bool(assignments),alerts_changed=False,
        original_test_activated=False,automatic_successor=False)
    write(OUT/'result.json',result);seal('evaluation-seal.json',['result.json']);verify();print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('extract','evaluate'))
    globals()[p.parse_args().stage]()
