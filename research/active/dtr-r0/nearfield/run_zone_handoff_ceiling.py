"""One frozen observable-feature pass, followed by consumed evaluator diagnosis."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from audit_calibration_footprint import read, sha
from ba_camera_corridor_metrics import evaluate_rows
from tof_corridor_calibration import score_frame, decide
from zone_handoff_ceiling import features

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/ba-core-transfer-20260920'
GAPS = ROOT/'artifacts.local/work/ba-calibration-footprint-audit-20260920/audit.json'
OUT = ROOT/'artifacts.local/work/ba-zone-handoff-ceiling-20260920'
DOC = Path(__file__).with_name('ZONE_HANDOFF_CEILING_20260920.md')
CODE = ('zone_handoff_ceiling.py','run_zone_handoff_ceiling.py','test_zone_handoff_ceiling.py',
        'tof_corridor_calibration.py','ba_camera_corridor.py','ba_camera_corridor_metrics.py',
        'audit_calibration_footprint.py')


def write(path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def verify_sources():
    for name in ('observation-seal.json','prediction-seal.json','evaluation-seal.json'):
        seal = read(SOURCE/name)
        assert seal['status'] == 'COMPLETE' and seal['frames'] == 432
        assert seal['protocol_sha256'] == sha(SOURCE/'protocol.json')
        for filename, digest in seal['hashes'].items():
            assert sha(SOURCE/filename) == digest, filename


def freeze():
    assert not OUT.exists(), 'One run only'
    verify_sources()
    OUT.mkdir(parents=True)
    (OUT/'protocol-before-run.md').write_bytes(DOC.read_bytes())
    write(OUT/'protocol.json', dict(id='ba-zone-handoff-ceiling-20260920',
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frames=432, gaps=4, gap_frames=5, threshold=read(SOURCE/'protocol.json')['threshold'],
        floor=0, connectivity=4, fp_added_max=5, complete_gaps_min=3, segments_max=37,
        document_sha256=sha(OUT/'protocol-before-run.md'),
        code={n:sha(Path(__file__).with_name(n)) for n in CODE},
        inputs={str(p.relative_to(ROOT).as_posix()):sha(p) for p in
                [SOURCE/n for n in ('protocol.json','observations.json','predictions.json',
                                   'frame-results.json','results.json','prediction-seal.json',
                                   'evaluation-seal.json','observation-seal.json')]+[GAPS]},
        scope='CONSUMED_CONTROLLED432_DIAGNOSTIC',
        backend='TASK_NOT_GPU_SUITABLE: scalar geometry and <=64-node graphs'))
    print('FROZEN', sha(OUT/'protocol.json'))


def verify():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'protocol-before-run.md') == p['document_sha256']
    for name,digest in p['code'].items():
        assert sha(Path(__file__).with_name(name)) == digest, name
    for name,digest in p['inputs'].items():
        assert sha(ROOT/name) == digest, name
    verify_sources()
    return p


def predict():
    p=verify()
    assert not (OUT/'features.json').exists()
    observed=read(SOURCE/'observations.json'); predictions=read(SOURCE/'predictions.json')
    assert len(observed) == len(predictions) == 432
    result=[]; start=time.perf_counter()
    for obs,pred in zip(observed,predictions):
        assert obs['id'] == pred['id']
        assert sha(SOURCE/obs['path']) == obs['sha256'] == pred['observation_sha256']
        with np.load(SOURCE/obs['path'],allow_pickle=False) as data:
            boxes,values=data['boxes'],data['values']
        original=score_frame(boxes,values)
        assert original['anchors'] == pred['anchors'] and original['zone_scores'] == pred['zone_scores']
        assert decide(original,p['threshold']) == pred['predictions']['calibrated']
        f=features(boxes.tolist(),pred['anchors'],pred['zone_scores'])
        assert abs(f['s1'] - original['score']) <= 1e-15
        assert f['S_component'] >= f['s1'] - 1e-15
        assert f['S_all'] >= max(f['S_pair'], f['S_component']) - 1e-15
        hypothetical=bool(original['baseline']['definite_zones'] or
                          (original['baseline']['alert'] and f['S_component'] >= p['threshold']))
        assert hypothetical or not pred['predictions']['calibrated']['alert']
        result.append(dict(id=obs['id'],observation_sha256=obs['sha256'],
                           hypothetical_component_alert=hypothetical,**f))
    write(OUT/'features.json', result)
    write(OUT/'feature-seal.json',dict(status='COMPLETE',frames=432,
        protocol_sha256=sha(OUT/'protocol.json'),features_sha256=sha(OUT/'features.json'),
        elapsed_s=time.perf_counter()-start,python=sys.executable,baseline_parity=432,
        evaluator_labels_read_in_feature_pass=False))
    print('FEATURES_SEALED',sha(OUT/'features.json'))


def evaluate():
    p=verify(); seal=read(OUT/'feature-seal.json')
    assert seal['protocol_sha256']==sha(OUT/'protocol.json') and seal['features_sha256']==sha(OUT/'features.json')
    assert not (OUT/'results.json').exists()
    f=read(OUT/'features.json'); rows=read(SOURCE/'frame-results.json')
    gap_audit=read(GAPS)
    gaps=[[r['id'] for r in g['frames'] if not r['alert']] for g in gap_audit['internal_gaps']]
    gap_ids={i for g in gaps for i in g}
    assert len(gaps)==4 and len(gap_ids)==5
    by_id={};joined=[]
    for row,item in zip(rows,f):
        assert row['id']==item['id']
        r=deepcopy(row); cal=r['predictions']['calibrated']; comp=deepcopy(cal)
        comp.update(alert=item['hypothetical_component_alert'],score=item['S_component'])
        comp['ambiguous']=comp['alert'] and not comp['definite_zones']
        comp['state']='ALERT_SUPPORTED' if comp['definite_zones'] else 'ALERT_AMBIGUOUS' if comp['alert'] else cal['state']
        comp['meaning']='Offline hypothetical pooling diagnostic; not live baseline'
        r['predictions']['component']=comp; joined.append(r); by_id[r['id']]={**item,'row':r}
    metrics=evaluate_rows(joined, arms=('calibrated','component'))
    groups=dict(all_frames=joined, TP=[r for r in joined if r['truth']],
        calibration_FP=[r for r in joined if not r['truth'] and r['predictions']['calibrated']['alert']],
        negative_withheld_UNKNOWN=[r for r in joined if not r['truth'] and not r['predictions']['calibrated']['alert']],
        gap_frames=[r for r in joined if r['id'] in gap_ids])
    table={}
    for name,group in groups.items():
        items=[by_id[r['id']] for r in group]
        table[name]=dict(frames=len(items), top2_available=sum(len(x['top2']['zones'])==2 for x in items),
            adjacent_top2=sum(x['top2']['adjacent'] for x in items),
            depth_consistent_top2=sum(x['top2']['depth_consistent'] for x in items),
            continuous_top2=sum(x['top2']['corridor_continuous'] for x in items),
            same_component_top2=sum(x['top2']['same_component'] for x in items),
            pair_reaches_threshold=sum(x['S_pair']>=p['threshold'] for x in items),
            component_reaches_threshold=sum(x['S_component']>=p['threshold'] for x in items),
            all_zone_sum_reaches_threshold=sum(x['S_all']>=p['threshold'] for x in items))
    gap_results=[dict(ids=g,complete_component=all(by_id[i]['hypothetical_component_alert'] for i in g),
                     complete_unconstrained_sum=all(by_id[i]['S_all']>=p['threshold'] for i in g),
                     frame_features=[{k:v for k,v in by_id[i].items() if k!='row'} for i in g]) for g in gaps]
    added=[r for r in joined if r['predictions']['component']['alert'] and not r['predictions']['calibrated']['alert']]
    assert all(not r['truth'] for r in added), 'Baseline already has complete TP'
    base,candidate=(metrics['arms'][a] for a in ('calibrated','component'))
    new_segments=[s for s in candidate['false_alert_segments'] if not any(
        r['clip_id']==s['clip_id'] and s['start_frame']<=r['frame_in_clip']<=s['end_frame']
        and not r['truth'] and r['predictions']['calibrated']['alert'] for r in joined)]
    first=lambda m:[(e['clip_id'],e['start_frame'],e['end_frame'],e['detected'],e['first_alert_time_s']) for e in m['events']]
    checks=dict(complete_gaps=sum(g['complete_component'] for g in gap_results)>=p['complete_gaps_min'],
        added_fp_within_cap=len(added)<=p['fp_added_max'],no_new_fp_segments=not new_segments,
        fragments_at_most37=candidate['false_alert_segment_count']<=p['segments_max'],
        tp_preserved=all(r['predictions']['component']['alert'] for r in joined if r['truth']),
        event_and_onset_preserved=first(base)==first(candidate))
    result=dict(status='CEILING_PASS_IMPLEMENT_EXACT_CHALLENGER' if all(checks.values()) else 'CEILING_FAIL_STOP_AGGREGATION',
        checks=checks,table=table,gaps=gap_results,metrics=metrics,
        strata={key:{value:evaluate_rows([r for r in joined if r[key]==value],arms=('calibrated','component'))
                     for value in sorted({r[key] for r in joined})} for key in ('layer','layout_relation')},
        added_FP_ids=[r['id'] for r in added],added_bridge_FP_ids=[r['id'] for r in added if r['id'] in gap_ids],
        collateral_FP_ids=[r['id'] for r in added if r['id'] not in gap_ids],new_FP_segments=new_segments,
        training_updates=0,baseline_outputs_changed=False,protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'frame-diagnostics.json',joined);write(OUT/'results.json',result)
    write(OUT/'result-seal.json',dict(status='COMPLETE',hashes={n:sha(OUT/n) for n in
        ('protocol.json','features.json','feature-seal.json','frame-diagnostics.json','results.json')}))
    print(json.dumps(dict(status=result['status'],checks=checks,table=table,
                          added_FP_ids=result['added_FP_ids'],gaps_recovered=sum(g['complete_component'] for g in gap_results))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('freeze','predict','evaluate'))
    args=parser.parse_args()
    assert OUT.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    globals()[args.phase]()
