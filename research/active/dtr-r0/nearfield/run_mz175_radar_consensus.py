"""Seal public-only Radar consensus before reusing TRAIN-only saved labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

from mz175_radar_consensus import METHOD, predict_frame
from mz136_incumbent import public_observations
from run_mz139_surface_fit import local_dependencies, selected_jsonl

ROOT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
WORK = ROOT/'artifacts.local/work/mz175-radar-consensus-20260916'
PARENT = ROOT/'artifacts.local/work/mz173-causal-geometry-20260916/run-v1'
INC = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/incumbent/fresh-v1'
CAP = ROOT/'artifacts.local/work/mz136-corridor-pair-20260914/source/returned-v1/capture-v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_array(path, key, ids, record_ids=None):
    """Scan JSON object boundaries, decoding only admitted object IDs.

    The sealed parent is a valid JSON top-level object of named record arrays;
    a boundary scan is not used to reinterpret malformed source JSON.
    """
    text = Path(path).read_text(encoding='utf-8')
    match = re.search(r'"'+re.escape(key)+r'"\s*:\s*\[', text)
    assert match, key
    depth = 0
    quoted = escaped = False
    start = None
    result = []
    record_index = 0
    for i in range(match.end(), len(text)):
        ch = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            assert depth >= 0
            if depth == 0:
                fragment = text[start:i+1]
                identity = re.search(r'"id"\s*:\s*"([^"]+)"', fragment)
                if record_ids is None:
                    assert identity, 'Source record lacks identity'
                    record_id = identity[1]
                else:
                    record_id = record_ids[record_index]
                    assert identity is None or identity[1]==record_id
                if record_id in ids:
                    value = json.loads(fragment)
                    value['id'] = record_id
                    result.append(value)
                record_index += 1
        elif ch == ']' and depth == 0:
            break
    assert len(result) == len(ids) and {x['id'] for x in result} == set(ids)
    assert record_ids is None or record_index==len(record_ids)
    return result


def report(rows, annotations, flags, baseline):
    counts = Counter(dict(TP=0, FP=0, FN=0, TN=0))
    events = []
    active = None
    previous_episode = None
    false_segments = 0
    previous_false = False
    new_tp, new_fp, lost_tp = [], [], []
    misses = []
    for row,a,p,b in zip(rows, annotations, flags, baseline):
        y = a['truth']
        counts['TP' if y and p else 'FN' if y else 'FP' if p else 'TN'] += 1
        if row['episode_id'] != previous_episode:
            active = None
            previous_false = False
        if not y:
            active = None
        elif active is None:
            active = dict(episode=row['episode_id'], onset_s=row['time_s'], first_alert_s=None)
            events.append(active)
        if y and p and active['first_alert_s'] is None:
            active['first_alert_s'] = row['time_s']
        now_false = bool(p and not y)
        false_segments += int(now_false and not previous_false)
        previous_false = now_false
        previous_episode = row['episode_id']
        if p and not b:
            (new_tp if y else new_fp).append(row['id'])
        if y and b and not p:
            lost_tp.append(row['id'])
        if not p and a.get('corridor_contributor_samples', 0):
            misses.append(dict(id=row['id'], samples=a['corridor_contributor_samples']))
    counts['UNKNOWN'] = flags.count(False)
    return dict(metrics=dict(counts), precision=counts['TP']/max(1,counts['TP']+counts['FP']),
        recall=counts['TP']/max(1,counts['TP']+counts['FN']), unknown=flags.count(False),
        false_segments=false_segments, events=events, detected_events=sum(e['first_alert_s'] is not None for e in events),
        new_tp=new_tp, new_fp=new_fp, lost_tp=lost_tp,
        nonalert_with_native_corridor_contributors=misses)


def summarize(rows, predictions, annotations):
    baseline = [p['baseline'] for p in predictions]
    arms = dict(baseline=baseline, **{arm:[p['arms'][arm]['candidate'] for p in predictions] for arm in ('any','all')})
    groups = {name:[i for i,a in enumerate(annotations) if a['family']==name]
              for name in sorted({a['family'] for a in annotations})}
    groups['pressure'] = [i for i,a in enumerate(annotations) if a['family']=='shallow_boundary_stress']
    groups['ordinary'] = [i for i,a in enumerate(annotations) if a['family']!='shallow_boundary_stress']
    reports = {}
    pairs = [(i+j,i+6+j) for i in range(0,len(rows),12) for j in range(6)]
    for i,j in pairs:
        assert rows[i]['time_s']==rows[j]['time_s'] and annotations[i]['family']==annotations[j]['family']
        # Pair identity is admission metadata, never an inference input.
        left = re.sub(r'_(in|out|enter|exit)$','',rows[i]['episode_id'])
        right = re.sub(r'_(in|out|enter|exit)$','',rows[j]['episode_id'])
        assert left==right and rows[i]['episode_id']!=rows[j]['episode_id']
    for arm,ff in arms.items():
        r = report(rows,annotations,ff,baseline)
        r['groups'] = {g:report([rows[i] for i in ix],[annotations[i] for i in ix],
                             [ff[i] for i in ix],[baseline[i] for i in ix]) for g,ix in groups.items()}
        r['pairs'] = dict(total=len(pairs), both_correct=sum(ff[i]==annotations[i]['truth'] and ff[j]==annotations[j]['truth'] for i,j in pairs))
        r['native_returned_samples'] = sum(a.get('returned_contributor_samples',0) for a in annotations)
        r['native_corridor_samples'] = sum(a.get('corridor_contributor_samples',0) for a in annotations)
        reports[arm] = r
    primary,base = reports['all'],reports['baseline']
    event_changes = [dict(episode=a['episode'], onset_s=a['onset_s'], baseline=b['first_alert_s'], candidate=a['first_alert_s'])
                     for a,b in zip(primary['events'],base['events'])]
    assert len(primary['events'])==len(base['events'])
    checks = dict(additional_true_frame=bool(primary['new_tp']), no_added_false_frame=not primary['new_fp'],
        no_old_true_frame_loss=not primary['lost_tp'], all_old_flags_retained=all(not b or p for b,p in zip(baseline,arms['all'])),
        old_event_timing_retained=all(e['candidate'] is not None and e['candidate']<=e['baseline'] for e in event_changes if e['baseline'] is not None),
        no_new_native_supported_nonalert={x['id'] for x in primary['nonalert_with_native_corridor_contributors']} <= {x['id'] for x in base['nonalert_with_native_corridor_contributors']},
        family_fp_nonincrease=all(primary['groups'][g]['metrics']['FP']<=base['groups'][g]['metrics']['FP'] for g in groups))
    return dict(arms=reports, checks=checks, passed=all(checks.values()), event_changes=event_changes,
        decision='MZ175_CONSUMED_CONSENSUS_RECALL_COMPONENT' if all(checks.values()) else 'MZ175_CONDITIONAL_CONSENSUS_GAIN_NOT_MET',
        all_any_different_frames=[rows[i]['id'] for i in range(len(rows)) if arms['all'][i]!=arms['any'][i]],
        eligible_ambiguous_returns=sum(len(p['evidence']) for p in predictions),
        branch_frames={arm:sum(p['additional'][arm] for p in predictions) for arm in ('all','any')},
        default_changed=False, radar_native_lineage='NOT_EVALUABLE',
        authority='CONSUMED_ORIGINAL_TRAIN192_NOT_FRESH_OR_DEVICE', original_dev_test_decoded=False)


def run(output):
    started = time.perf_counter()
    output = output.resolve()
    assert output.is_relative_to(WORK.resolve()) and not output.exists()
    output.mkdir(parents=True)
    inputs = {}
    def checked(path, expected=None, decode=False):
        digest = sha(path)
        assert expected is None or digest==expected, str(path)
        inputs[str(path.resolve())] = digest
        return read(path) if decode else digest
    try:
        done = checked(PARENT/'completion.json',decode=True)
        assert done['status']=='PASS'
        seal = checked(PARENT/'prediction-seal.json',done['prediction_seal_sha256'],True)
        freeze = checked(PARENT/'freeze.json',seal['freeze_sha256'],True)
        ids = freeze['ids']
        assert len(ids)==len(set(ids))==192
        checked(PARENT/'summary.json',done['summary_sha256'])
        checked(PARENT/'geometry-cases.json',done['cases_sha256'])
        inc_done = checked(INC/'completion.json',decode=True)
        assert inc_done['status']=='PASS'
        inc_seal = checked(INC/'prediction-seal.json',inc_done['prediction_seal_sha256'],True)
        checked(INC/'nominal/predictions.json',inc_seal['predictions_sha256']['nominal'])
        for name in ('raw.jsonl','receipt.json'):
            path = CAP/name
            expected = [h for p,h in inc_seal['inputs'].items() if Path(p).resolve()==path.resolve()]
            assert len(expected)==1
            checked(path,expected[0])
        obs_seal = checked(INC/'observation-seal.json',inc_seal['observation_seal_sha256'],True)
        checked(INC/'nominal/raw.jsonl',obs_seal['hashes']['nominal'])
        for name in ('mz111_spatial_evidence.py','mz107_rgb_association.py','mz136_incumbent.py'):
            path = CODE/name
            expected = [h for p,h in inc_seal['source_hashes'].items() if Path(p).name==name]
            assert len(expected)==1
            checked(path,expected[0])
        rows = public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids)))
        assert rows==selected_jsonl(INC/'nominal/raw.jsonl',set(ids))
        assert [r['id'] for r in rows]==ids
        assert set(Counter(r['episode_id'] for r in rows).values())=={6}
        old = selected_array(INC/'nominal/predictions.json','predictions',ids)
        # Original corrected records are positional, from the sealed incumbent's
        # zip(rows, baseline). Scan only ID text in public JSONL for that join.
        record_ids = [re.search(r'"id"\s*:\s*"([^"]+)"',line)[1]
                      for line in (INC/'nominal/raw.jsonl').read_text().splitlines()]
        assert len(record_ids)==len(set(record_ids))==inc_done['frames']
        corrected = selected_array(INC/'nominal/predictions.json','corrected',ids,record_ids)
        assert [p['id'] for p in old]==[p['id'] for p in corrected]==ids
        sources = local_dependencies(__file__)
        for name in ('MZ175_PROTOCOL_20260916.md','test_mz175_radar_consensus.py'):
            sources[str(CODE/name)] = sha(CODE/name)
        for path in sources:
            p=Path(path); dest=output/'source-snapshot'/p.relative_to(ROOT.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        write(output/'freeze.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,sources=sources,
            method=METHOD,ids=ids,compute_cap_s=120,previous_goal_turn='PROGRESS_DELIVERY_COMPLETED',
            authority='CONSUMED_TRAIN_ONLY_SAVED_LABELS_AFTER_PREDICTION_SEAL'))
        before=time.perf_counter()
        predictions=[predict_frame(r,c,b) for r,c,b in zip(rows,corrected,old)]
        inference_s=time.perf_counter()-before
        write(output/'predictions.json',predictions)
        write(output/'prediction-seal.json',dict(freeze_sha256=sha(output/'freeze.json'),predictions_sha256=sha(output/'predictions.json'),
            inference_seconds=inference_s,authority='SEALED_BEFORE_REUSED_LABEL_DECODE'))
        write(output/'backend.json',dict(device='cpu',reason='TASK_NOT_GPU_SUITABLE',
            operation='SMALL_IRREGULAR_BOOLEAN_AND_NUMPY_PLANE_GEOMETRY',python=sys.executable,inference_seconds=inference_s))
        write(output/'evaluation-start.json',dict(prediction_seal_sha256=sha(output/'prediction-seal.json')))
        previous=read(PARENT/'summary.json')
        native=previous['arms']['baseline']['native']['cases']
        families=read(PARENT/'geometry-cases.json')
        assert [n['id'] for n in native]==[f['id'] for f in families]==ids
        annotations=[]
        for n,f,p in zip(native,families,predictions):
            assert p['baseline']==n['baseline']==n['candidate']
            annotations.append(dict(n,family=f['family']))
        summary=summarize(rows,predictions,annotations)
        assert summary['arms']['baseline']['metrics']==previous['arms']['baseline']['metrics']
        write(output/'annotations.json',annotations)
        write(output/'row-metadata.json',[{k:r[k] for k in ('id','episode_id','time_s')} for r in rows])
        summary['inference_seconds']=inference_s
        write(output/'summary.json',summary)
        assert all(sha(p)==h for p,h in (sources|inputs).items())
        assert time.perf_counter()-started < 120
        write(output/'completion.json',dict(status='PASS',seconds=time.perf_counter()-started,decision=summary['decision'],
            hashes={n:sha(output/n) for n in ('prediction-seal.json','summary.json','annotations.json','row-metadata.json','backend.json')},
            sources_inputs_unchanged=True,resource_state='NO_PERSISTENT_PROCESS_OR_ALLOCATION'))
        print(json.dumps(dict(decision=summary['decision'],checks=summary['checks'],
            arms={a:r['metrics'] for a,r in summary['arms'].items()},inference_seconds=inference_s)),flush=True)
    except Exception as exc:
        write(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
