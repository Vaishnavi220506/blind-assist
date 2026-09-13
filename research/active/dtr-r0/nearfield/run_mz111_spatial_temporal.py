"""Fixed observable multi-method study, evaluator scoring after prediction seal."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import mz108_competitive_association as visual
import mz109_interval_extent as interval
import mz111_spatial_evidence as spatial
import mz111_temporal_geometry as temporal
from run_mz107_four_sensor import ROOT, readrows, sha, truth, write, metrics
from research_backend import BackendCandidate, DeviceObservation, select_backend


def event_metrics(rows, truth_values, predictions):
    """Discrete contiguous truth-positive and false-alert segments, .25s bins."""
    positive = missed = false_segments = 0; delays = []; false_bins = 0
    start = None; alerted = False; previous_false = False; episode = None; previous_time = None
    for row, gt, pred in zip(rows, truth_values, predictions):
        if row['episode_id'] != episode:
            if start is not None and not alerted:
                missed += 1
            start = None; previous_false = False; previous_time = None
        if previous_time is not None:
            assert abs(row['time_s']-previous_time-.25) < 1e-6
        episode = row['episode_id']; previous_time = row['time_s']
        if gt and start is None:
            positive += 1; start = row['time_s']; alerted = False
        if gt and pred and not alerted:
            delays.append(row['time_s']-start); alerted = True
        if not gt and start is not None:
            if not alerted:
                missed += 1
            start = None
        false = bool(pred and not gt)
        false_segments += false and not previous_false; false_bins += false
        previous_false = false
    if start is not None and not alerted:
        missed += 1
    return dict(positive_segments=positive, missed_positive_segments=missed,
                first_alert_delay_s=delays, max_detected_delay_s=max(delays,default=None),
                mean_detected_delay_s=sum(delays)/len(delays) if delays else None,
                false_alert_segments=false_segments, false_alert_bin_duration_s=false_bins*.25)


def methods(rows, nominal):
    result = dict(baseline=[dict(candidate=p['baseline']) for p in nominal], nominal=nominal,
                  interval=[interval.refine(r,p) for r,p in zip(rows,nominal)])
    for name, surface, filtered in [('shell','shell',False),('plane','plane',False),('filtered_plane','plane',True)]:
        result[name] = spatial.predict(rows,nominal,surface,filtered)
    result['temporal'] = temporal.predict(rows,nominal)
    result['combined'] = [dict(candidate=s['candidate'] or (t['candidate'] and not n['candidate']))
                          for s,t,n in zip(result['filtered_plane'],result['temporal'],nominal)]
    for arm, preds in result.items():
        assert len(preds) == len(rows)
        assert all(not n['tof_support'] or p['candidate'] for n,p in zip(nominal,preds)),arm
    return result


def score(rows, es, values):
    gt = np.array([truth(e) for e in es], bool)
    baseline = np.array([p['candidate'] for p in values['baseline']], bool)
    nominal = np.array([p['candidate'] for p in values['nominal']], bool)
    result = {}
    for arm, preds in values.items():
        prediction = np.array([p['candidate'] for p in preds], bool)
        result[arm] = dict(metrics=metrics(gt,prediction), lost_baseline_TP=int((gt&baseline&~prediction).sum()),
            lost_nominal_TP=int((gt&nominal&~prediction).sum()), gained_nominal_TP=int((gt&~nominal&prediction).sum()),
            events=event_metrics(rows,gt,prediction),families={})
        for family in sorted({e['family'] for e in es}):
            mask = np.array([e['family']==family for e in es])
            result[arm]['families'][family] = metrics(gt[mask],prediction[mask])
    return result


def run(out, fresh=None):
    out = out.resolve(); assert out.is_relative_to((ROOT/'artifacts.local').resolve()) and not out.exists()
    out.mkdir(parents=True)
    work = ROOT/'artifacts.local/work'; old_study = work/'mz109-interval-extent-20260913/analysis-v1'
    if fresh:
        sources = dict(fresh_mz112=fresh.resolve())
    else:
        sources = dict(consumed_mz107=work/'mz107-rgb-tof-radar-imu-20260913/capture-v1',
                       consumed_mz108=work/'mz108-competitive-association-20260913/capture-v2',
                       consumed_mz109=work/'mz109-interval-extent-20260913/capture-v1')
    old_seal = json.loads((old_study/'prediction-seal.json').read_text())
    for name,digest in old_seal['code_sha256'].items():
        assert sha(Path(__file__).with_name(name))==digest,name
    packets = {}; predictions = {}; seals = {}; timings = {}
    for panel,capture in sources.items():
        receipt = json.loads((capture/'receipt.json').read_text())
        assert receipt['status']=='PASS'
        for name,digest in receipt['hashes'].items():
            assert sha(capture/name)==digest,name
        manifest=json.loads((capture/'manifest.json').read_text())
        assert manifest['rgb_camera_count']==1 and not manifest['depth_images_produced']
        rows=readrows(capture/'raw.jsonl'); assert len(rows)==receipt['frames']
        assert len({r['id'] for r in rows})==len(rows)
        if fresh:
            assert len(rows)==240
            images={r['id']:cv2.imread(str(capture/r['rgb_path'])) for r in rows}
            assert all(im is not None for im in images.values())
            nominal=visual.predict(rows,lambda r:images[r['id']])
        else:
            key='new_mz109' if panel=='consumed_mz109' else panel
            pp=old_study/key/'predictions.json'; seal=old_seal['panels'][key]
            assert sha(pp)==seal['prediction_sha256'] and sha(capture/'raw.jsonl')==seal['raw_sha256']
            nominal=json.loads(pp.read_text())['nominal']
            assert [r['id'] for r in rows]==[p['id'] for p in nominal]
        if not packets:
            select_backend('scalar-scoring',cpu=BackendCandidate('python-spatial-temporal','cpu',
                lambda: spatial.predict(rows[:1],nominal[:1]),lambda _:DeviceObservation('cpu','host CPU','Python and NumPy scalar geometry')),
                record_path=out/'backend.json',capabilities={'workload':'small per-return geometry and causal state; authenticated RGB proposal cache'})
        start=time.perf_counter();values=methods(rows,nominal);timings[panel]=(time.perf_counter()-start)/len(rows)
        disabled=visual.predict(rows,lambda _:None); controls=methods(rows,disabled)
        assert all(all(p['candidate']==d['baseline'] for p,d in zip(arm,disabled)) for arm in controls.values())
        if not fresh:
            key='new_mz109' if panel=='consumed_mz109' else panel
            old_cf=json.loads((work/'mz110-association-diagnostic-20260913/analysis-v2'/key/'observable-traces.json').read_text())['no_tof_gate']
            assert all(p['candidate']==c['candidate'] for p,c in zip(values['shell'],old_cf))
        target=out/panel;target.mkdir()
        write(target/'predictions.json',values);write(target/'rgb-disabled.json',controls)
        packets[panel]=rows;predictions[panel]=values
        seals[panel]=dict(capture=str(capture),receipt_sha256=sha(capture/'receipt.json'),raw_sha256=sha(capture/'raw.jsonl'),
                          predictions_sha256=sha(target/'predictions.json'),disabled_sha256=sha(target/'rgb-disabled.json'))
    for path in (Path(__file__),Path(spatial.__file__),Path(temporal.__file__),Path(visual.__file__),Path(interval.__file__),
                 Path(__file__).with_name('MZ111_PROTOCOL_20260913.md')):
        shutil.copyfile(path,out/path.name)
    write(out/'prediction-seal.json',dict(status='ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE',panels=seals,
        code_sha256={p.name:sha(p) for p in out.glob('*.py')},protocol_sha256=sha(out/'MZ111_PROTOCOL_20260913.md')))
    results={};allrows=[];alles=[];allvalues={}
    for panel,rows in packets.items():
        es=readrows(sources[panel]/'evaluator.jsonl')
        assert [r['id'] for r in rows]==[e['id'] for e in es]
        results[panel]=score(rows,es,predictions[panel])
        allrows.extend([dict(r,episode_id=panel+'/'+r['episode_id']) for r in rows]);alles.extend(es)
        for arm,values in predictions[panel].items():allvalues.setdefault(arm,[]).extend(values)
    result=dict(status='FIXED_MULTI_METHOD_STUDY_COMPLETE',frames=len(allrows),panels=results,
                overall=score(allrows,alles,allvalues),all_methods_seconds_per_frame=timings,
                scope='FRESH_CONSTRUCTED_DEVELOPMENT' if fresh else 'CONSUMED_DEVELOPMENT')
    write(out/'summary.json',result)
    write(out/'completion.json',dict(status='PASS',output_hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.json'))},resources_started=[]))
    print(json.dumps({arm:{k:v for k,v in r.items() if k!='families'} for arm,r in result['overall'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fresh-capture',type=Path)
    a=p.parse_args();run(a.output,a.fresh_capture)
