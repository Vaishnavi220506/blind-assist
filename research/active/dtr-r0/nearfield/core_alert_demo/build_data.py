"""Export portable replay data after exact parity on all three frozen cohorts.

Only outputs under artifacts.local/work/ba-core-alert-demo-20260920 are written.
Labels are joined after every observation-only policy decision is computed.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from policy import CoreAlertPolicy, CALIBRATION_THRESHOLD, STRONG_THRESHOLD, NEARFIELD
from tof_fov45_core import boxes45
from audit_existing_strata_20260920 import tally

ROOT=NEARFIELD.parents[3]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'ba-core-alert-demo-20260920'
SITE=OUT/'site'
ARMS=('calibration','strong','hold')
COHORTS=(
    ('validation','新布局验证','冻结策略的一次新完整布局验证；Core通过，Boundary仍存在明显漏报。','ba-core-hold-validation-20260920',None),
    ('transfer','迁移反例','此前完整布局：保留两次起报延迟，展示策略的能力边界。','ba-core-workpoint-transfer-20260920','primary'),
    ('development','开发来源','最初完整布局与后续封存重放；这是已使用的开发证据。','ba-core-transfer-20260920','context'),
)
TYPE_LABELS=dict(head_horizontal='横向悬空体',head_hanging_plane='悬挂面',
    head_protruding_edge='头部突出边缘',body_protruding_plane='身体突出面',
    body_suspended_solid='身体悬空体',body_large_solid='身体大型实体')
LAYOUT_LABELS=dict(INSIDE='相交布局',BOUNDARY='边界布局',OUTSIDE='外侧布局')
HASHES={}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(path,digest):
    assert sha(path)==digest, str(path)
    HASHES[str(path.relative_to(ROOT))]=digest


def seals(directory,frames):
    protocol_digest=sha(directory/'protocol.json')
    HASHES[str((directory/'protocol.json').relative_to(ROOT))]=protocol_digest
    for name in ('observation-seal.json','prediction-seal.json','evaluation-seal.json'):
        if not (directory/name).exists():
            assert name=='observation-seal.json' and frames==864
            continue
        seal=read(directory/name)
        assert seal['status']=='COMPLETE'
        if frames==864 and name=='evaluation-seal.json':
            # The historical causal evaluator binds its protocol/coverage through
            # the recursively verified prediction seal rather than repeating them.
            assert 'prediction-seal.json' in seal['hashes']
        else:
            assert seal['frames']==frames and seal['protocol_sha256']==protocol_digest
        for n,h in seal['hashes'].items():
            verify(directory/n,h)
        HASHES[str((directory/name).relative_to(ROOT))]=sha(directory/name)


def geometry():
    boxes=boxes45();f=640/(2*np.tan(np.deg2rad(50)))
    zones=[]
    for y0,x0,y1,x1 in boxes:
        pixel=[float(x0*640/256),float(y0*360/192),float(x1*640/256),float(y1*360/192)]
        zones.append(dict(box=[int(y0),int(x0),int(y1),int(x1)],pixel_box=pixel,
            slopes_a=[(pixel[0]-320)/f,(pixel[2]-320)/f],
            slopes_b=[(pixel[1]-180)/f,(pixel[3]-180)/f]))
    return dict(volume=dict(x=[-.3,.3],y=[-.2,.9],z=[.3,3.]),
                native_width=640,native_height=360,zone_geometry=zones)


def main():
    SITE.mkdir(parents=True,exist_ok=True)
    causal=WORK/'ba-causal-event-readout-20260920'
    seals(causal,864)
    causal_predictions=read(causal/'predictions.json')
    causal_metrics=read(causal/'results.json')['cohorts']
    cohorts=[];parity=[];asset_hashes={};frame_count=0
    for cohort_id,title,subtitle,folder,reference_key in COHORTS:
        source=WORK/folder;seals(source,432)
        protocol=read(source/'protocol.json')
        for filename in ('tof_corridor_calibration.py','ba_camera_corridor.py','tof_fov45_core.py'):
            verify(NEARFIELD/filename,protocol['code_hashes'][filename])
        observations=read(source/'observations.json');original=read(source/'predictions.json')
        assert len(observations)==len(original)==432
        reference=causal_predictions[reference_key] if reference_key else original
        assert len(reference)==432
        policy=CoreAlertPolicy();computed=[]
        for i,(obs,old,ref) in enumerate(zip(observations,original,reference)):
            assert obs['id']==old['id']==ref['id']==f'f{i:04d}'
            for key in ('clip_id','frame_in_clip','time_s'):
                assert obs[key]==old[key]==ref[key]
            verify(source/obs['path'],obs['sha256'])
            with np.load(source/obs['path'],allow_pickle=False) as data:
                values=data['values'].copy();boxes=data['boxes'].copy()
            output=policy.step(boxes=boxes,values=values,clip_id=obs['clip_id'],
                               frame_id=obs['id'],time_s=obs['time_s'])
            d=output['decision']
            expected_flags=ref['flags'] if reference_key else {a:ref['predictions'][a]['alert'] for a in ARMS}
            assert [d['calibration'],d['strong'],d['alert']]==[expected_flags[a] for a in ARMS]
            assert output['score']==ref['score']
            assert d['unknown']==(ref['unknown'] if reference_key else ref['predictions']['hold']['unknown'])
            if reference_key:
                assert d['previous_strong']==ref['previous_strong']
                assert d['held_only']==bool(ref['hold_trigger'] and not ref['flags']['strong'])
            else:
                assert d['held_only']==ref['predictions']['hold']['held_only']
            raw=old['predictions']['raw'] if cohort_id=='development' else old['raw']
            assert output['raw']==raw
            if 'anchors' in old:
                anchors=[dict(zone=z['zone'],possible=z['possible'],definite=z['definite'],interval_m=z['interval']) for z in output['zones'] if z['interval'] is not None]
                factors=[{k:z[k] for k in ('zone','joint','depth','angular_given_depth')} for z in output['zones'] if z['interval'] is not None]
                assert anchors==old['anchors'] and factors==old['zone_scores']
            computed.append(dict(obs=obs,output=output,values=[float(v) if np.isfinite(v) else None for v in values]))
        # Inference on this complete cohort has finished before evaluator label access.
        labels=read(source/'frame-results.json')
        assert len(labels)==432
        rows=[];clips=OrderedDict();assets=SITE/'assets'/cohort_id;assets.mkdir(parents=True,exist_ok=True)
        for item,label in zip(computed,labels):
            obs,output=item['obs'],item['output'];assert obs['id']==label['id']
            for key in ('clip_id','frame_in_clip','time_s'):
                assert obs[key]==label[key]
            rgb_source=source/obs['rgb_path'];verify(rgb_source,obs['rgb_sha256'])
            destination=assets/(obs['id']+rgb_source.suffix.lower())
            if not destination.exists() or sha(destination)!=obs['rgb_sha256']:
                shutil.copyfile(rgb_source,destination)
            assert sha(destination)==obs['rgb_sha256']
            rgb=destination.relative_to(SITE).as_posix();asset_hashes[rgb]=obs['rgb_sha256']
            if obs['clip_id'] not in clips:
                clips[obs['clip_id']]=dict(id=obs['clip_id'],
                    label=TYPE_LABELS[label['type_id']]+' · '+LAYOUT_LABELS[label['layout_relation']]+' · '+label['background'],
                    type=label['type_id'],layer=label['layer'],layout=label['layout_relation'],
                    background=label['background'],frames=[])
            clips[obs['clip_id']]['frames'].append(dict(id=obs['id'],time_s=obs['time_s'],rgb=rgb,
                values=item['values'],**output,evaluation={k:label[k] for k in ('truth','boundary','relation','target_camera_bounds_m')}))
            rows.append({**label,'demo_decision':output['decision']})
        assert len(clips)==36 and all(len(c['frames'])==12 for c in clips.values())
        metrics={}
        for name,old_name,mask in (('all','all432',lambda r:True),('core','core288',lambda r:r['layout_relation']!='BOUNDARY'),('boundary','boundary144',lambda r:r['layout_relation']=='BOUNDARY')):
            metrics[name]={}
            for arm,key in (('calibration','calibration'),('strong','strong'),('hold','alert')):
                m=tally(rows,lambda r,k=key:r['demo_decision'][k],mask,.2,'clip_id')
                m.pop('zero_return_frames')
                m['prediction_unknown']=sum(r['demo_decision']['unknown'] for r in rows if mask(r))
                expected=(causal_metrics[reference_key]['metrics'][arm][old_name] if reference_key else read(source/'results.json')['metrics'][arm][old_name])
                assert m==expected,(cohort_id,name,arm)
                metrics[name][arm]=m
        cohorts.append(dict(id=cohort_id,title=title,subtitle=subtitle,source=folder,
                            metrics=metrics,clips=list(clips.values())))
        frame_count+=len(rows)
        parity.append(dict(cohort=cohort_id,frames=432,clips=36,all_three_flags=True,
            score_and_current_unknown=True,raw_support=True,zone_factors=cohort_id!='validation',
            all_core_boundary_metrics=True))
    assert frame_count==1296 and len(asset_hashes)==1296
    payload=dict(version=1,thresholds=dict(calibration=CALIBRATION_THRESHOLD,strong=STRONG_THRESHOLD),
                 geometry=geometry(),cohorts=cohorts)
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    # External local JS works both from file:// and from the optional HTTP launcher.
    data_path=SITE/'demo-data.js';data_path.write_text('window.CORE_DEMO_DATA='+encoded+';\n',encoding='utf-8')
    assert json.loads(data_path.read_text(encoding='utf-8')[len('window.CORE_DEMO_DATA='):-2])==payload
    receipt=dict(status='PASS',built_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='ENGINEERING_REPLAY_EXPORT_NO_NEW_EXPERIMENT',frames=frame_count,clips=108,
        source_seals_and_payload_hashes=HASHES,exported_asset_sha256=asset_hashes,
        parity=parity,export_sha256=sha(data_path),
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('policy.py','build_data.py','test_policy.py')},
        label_separation='All observation-only policy outputs computed before evaluator frame labels are read for each cohort.',
        limitations=['Saved RGB is a replay illustration, not an inference input.',
                    'Core results do not generalize to Boundary, natural scenes, hardware, post-event release or target-device latency.',
                    'zone_factors in new validation are generated by the same frozen scorer; that source sealed raw/scalar scores, not per-zone factors.'])
    (OUT/'build-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',frames=1296,clips=108,assets=1296,
                         data_bytes=data_path.stat().st_size,site=str(SITE)),ensure_ascii=False))


if __name__=='__main__':
    main()
