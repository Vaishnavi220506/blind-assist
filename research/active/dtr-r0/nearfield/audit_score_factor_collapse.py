"""Frozen descriptive factor attribution; never modifies or emits predictions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import time

from audit_calibration_footprint import read, sha

ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'artifacts.local/work/ba-core-transfer-20260920'
GAP=ROOT/'artifacts.local/work/ba-calibration-footprint-audit-20260920/audit.json'
OUT=ROOT/'artifacts.local/work/ba-score-factor-collapse-20260920'
DOC=Path(__file__).with_name('SCORE_FACTOR_COLLAPSE_PROTOCOL_20260920.md')


def write(path,data):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(data,handle,indent=2,allow_nan=False)


def attribution(before,gap,after,threshold):
    if any(x is None or x['o'] is None for x in (before,gap,after)):
        return dict(status='NOT_EVALUABLE_UNDEFINED_FACTOR')
    dr=median([before['d'],after['d']]);orr=median([before['o'],after['o']])
    keep_depth=gap['d']*orr;keep_overlap=dr*gap['o']
    overlap_restores=keep_depth>=threshold;depth_restores=keep_overlap>=threshold
    category=('EITHER_REPLACEMENT_SUFFICIENT' if overlap_restores and depth_restores else
              'OVERLAP_REPLACEMENT_ONLY' if overlap_restores else
              'DEPTH_REPLACEMENT_ONLY' if depth_restores else
              'BOTH_REPLACEMENTS_REQUIRED' if dr*orr>=threshold else 'REFERENCE_NOT_EXPLANATORY')
    log_d=math.log(dr/gap['d']) if dr>0 and gap['d']>0 else None
    log_o=math.log(orr/gap['o']) if orr>0 and gap['o']>0 else None
    if log_d is not None and log_o is not None:
        assert math.isclose(log_d+log_o,math.log(dr*orr/gap['s']),abs_tol=1e-12)
    return dict(status='EVALUABLE',category=category,d_ref=dr,o_ref=orr,
        d_relative=gap['d']/dr if dr else None,o_relative=gap['o']/orr if orr else None,
        CF_depth_keep_depth_replace_overlap=keep_depth,
        CF_overlap_replace_depth_keep_overlap=keep_overlap,
        overlap_replacement_reaches_threshold=overlap_restores,
        depth_replacement_reaches_threshold=depth_restores,
        reference_product=dr*orr,signed_log_depth=log_d,signed_log_overlap=log_o)


def check_source():
    checks={}
    for name in ('prediction-seal.json','observation-seal.json','evaluation-seal.json'):
        seal=read(SOURCE/name)
        assert seal['status']=='COMPLETE' and seal['frames']==432
        assert seal['protocol_sha256']==sha(SOURCE/'protocol.json')
        for file,digest in seal['hashes'].items():
            assert sha(SOURCE/file)==digest,file
            checks[file]=digest
    old=read(SOURCE/'protocol.json')
    for name in ('tof_corridor_calibration.py','ba_camera_corridor.py'):
        assert sha(Path(__file__).with_name(name))==old['code_hashes'][name]
    return checks


def freeze():
    assert not OUT.exists(),'One audit; no overwrite'
    checks=check_source();OUT.mkdir(parents=True)
    (OUT/'protocol-before-run.md').write_bytes(DOC.read_bytes())
    write(OUT/'protocol.json',dict(id='ba-score-factor-collapse-20260920',
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes=checks,gap_audit_sha256=sha(GAP),threshold=read(SOURCE/'protocol.json')['threshold'],
        doc_sha256=sha(OUT/'protocol-before-run.md'),code_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('audit_calibration_footprint.py')),
        test_sha256=sha(Path(__file__).with_name('test_score_factor_collapse.py')),
        source_protocol_sha256=sha(SOURCE/'protocol.json'),gap_frames=5,gaps=4,
        backend='TASK_NOT_GPU_SUITABLE: scalar JSON and factor arithmetic',automatic_successor=False))
    print('FROZEN',sha(OUT/'protocol.json'))


def run():
    start=time.perf_counter();p=read(OUT/'protocol.json')
    assert sha(Path(__file__))==p['code_sha256'] and sha(GAP)==p['gap_audit_sha256']
    assert sha(OUT/'protocol-before-run.md')==p['doc_sha256']
    assert check_source()==p['source_hashes']
    predictions={r['id']:r for r in read(SOURCE/'predictions.json')}
    rows={r['id']:r for r in read(SOURCE/'frame-results.json')}
    gaps=read(GAP)['internal_gaps'];threshold=p['threshold']

    def fixed(frame,z):
        pred=predictions[frame]
        anchor=next((a for a in pred['anchors'] if a['zone']==z),None)
        score=next((s for s in pred['zone_scores'] if s['zone']==z),None)
        if anchor is None:
            assert score is None
            return None
        assert score is not None
        if score['depth']:
            assert math.isclose(score['joint'],score['depth']*score['angular_given_depth'],abs_tol=1e-15)
        else:
            assert score['joint']==0 and score['angular_given_depth'] is None
        return dict(zone=z,d=score['depth'],o=score['angular_given_depth'],s=score['joint'],
            threshold_ratio=score['joint']/threshold,interval_m=anchor['interval_m'],
            possible=anchor['possible'],definite=anchor['definite'])

    def dominant(frame):
        pred=predictions[frame];possible={a['zone'] for a in pred['anchors'] if a['possible']}
        ranking=sorted((s for s in pred['zone_scores'] if s['zone'] in possible),key=lambda s:(-s['joint'],s['zone']))
        assert ranking and ranking[0]['joint']==pred['predictions']['calibrated']['score']
        return fixed(frame,ranking[0]['zone'])

    windows=[];attributions=[];count=0
    for gap in gaps:
        ids=[r['id'] for r in gap['frames']];first,last=ids[0],ids[-1]
        assert rows[first]['predictions']['calibrated']['alert'] and rows[last]['predictions']['calibrated']['alert']
        assert len({rows[i]['clip_id'] for i in ids})==1
        assert [rows[i]['frame_in_clip'] for i in ids]==list(range(rows[first]['frame_in_clip'],rows[last]['frame_in_clip']+1))
        for frame in ids:
            assert predictions[frame]['predictions']==rows[frame]['predictions']
            assert not rows[frame]['truth'],'These gaps must not be called missed true obstacles'
        dom={i:dominant(i) for i in ids};zones=sorted({x['zone'] for x in dom.values()})
        windows.append(dict(clip_id=gap['clip_id'],frames=[dict(id=i,time_s=rows[i]['time_s'],
            alert=rows[i]['predictions']['calibrated']['alert'],truth=False,dominant=dom[i]) for i in ids],
            fixed_zone_tracks={str(z):[dict(id=i,factors=fixed(i,z)) for i in ids] for z in zones}))
        count+=len(ids)
        for i in ids[1:-1]:
            assert not rows[i]['predictions']['calibrated']['alert']
            z=dom[i]['zone']
            attributions.append(dict(id=i,before=first,after=last,dominant_zones=[dom[first]['zone'],z,dom[last]['zone']],
                gap_factors=dom[i],dominant_CF=attribution(dom[first],dom[i],dom[last],threshold),
                fixed_gap_zone_CF=attribution(fixed(first,z),dom[i],fixed(last,z),threshold)))
    assert len(windows)==4 and len(attributions)==5 and count==13
    result=dict(status='COMPLETE_DIAGNOSTIC_ONLY',frames_in_windows=count,gap_frames=5,threshold=threshold,
        windows=windows,attributions=attributions,
        dominant_categories=dict(Counter(x['dominant_CF']['category'] for x in attributions)),
        fixed_gap_zone_categories=dict(Counter(x['fixed_gap_zone_CF'].get('category',x['fixed_gap_zone_CF']['status']) for x in attributions)),
        unchanged_predictions_sha256=sha(SOURCE/'predictions.json'),elapsed_s=time.perf_counter()-start,
        output_changes=0,new_algorithm=False,automatic_successor=False,protocol_sha256=sha(OUT/'protocol.json'))
    assert check_source()==p['source_hashes']
    write(OUT/'results.json',result)
    write(OUT/'result-seal.json',dict(status='COMPLETE',hashes={n:sha(OUT/n) for n in ('protocol.json','results.json')}))
    print(json.dumps({k:result[k] for k in ('status','dominant_categories','fixed_gap_zone_categories','output_changes')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('freeze','run'))
    args=parser.parse_args();assert OUT.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    globals()[args.phase]()
