"""Frozen observable joint-assignment comparison; evaluator after prediction seal."""
import argparse
import json
from pathlib import Path
import shutil
import time
import cv2
import mz108_competitive_association as visual
import mz114_joint_association as joint
from run_mz107_four_sensor import ROOT, readrows, sha, truth, write
from run_mz111_spatial_temporal import score
from run_mz113_dynamic_flow import evaluate as inherited, exit_metrics


def evaluate(rows, nominal, images):
    arms, costs = inherited(rows, nominal, images)
    for label, temporal in [('joint_current', False), ('joint_temporal', True)]:
        start = time.perf_counter()
        spatial = joint.predict(rows, nominal, lambda r: images.get(r['id']), use_temporal=temporal)
        costs[label] = (time.perf_counter()-start)/len(rows)
        arms[label+'_spatial'] = spatial
        arms[label] = [dict(candidate=bool(s['candidate'] or (f['candidate'] and not n['candidate'])))
                       for s, f, n in zip(spatial, arms['flow'], nominal)]
    for arm, predictions in arms.items():
        assert len(predictions) == len(rows), arm
        assert all(not n['tof_support'] or p['candidate'] for n, p in zip(nominal, predictions)), arm
    return arms, costs


def run(output):
    output = output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    output.mkdir(parents=True)
    base = ROOT/'artifacts.local/work/mz113-dynamic-flow-20260913'
    packets = {}; seals = {}; costs = {}
    for prior in ('consumed-v1', 'fresh-v1'):
        original_seal = json.loads((base/prior/'prediction-seal.json').read_text())
        for name, item in original_seal['panels'].items():
            source = Path(item['capture']); panel = name.replace('fresh_', 'consumed_')
            cached = base/prior/name/'predictions.json'
            assert sha(cached) == item['predictions_sha256']
            assert sha(source/'raw.jsonl') == item['raw_sha256']
            assert sha(source/'receipt.json') == item['receipt_sha256']
            receipt = json.loads((source/'receipt.json').read_text()); assert receipt['status'] == 'PASS'
            for file, digest in receipt['hashes'].items(): assert sha(source/file) == digest, file
            rows = readrows(source/'raw.jsonl')
            assert len(rows) == receipt['frames'] and len({r['id'] for r in rows}) == len(rows)
            cache = json.loads(cached.read_text()); nominal = cache['nominal']
            assert len(rows) == len(nominal)
            assert all('id' not in n or n['id'] == r['id'] for n, r in zip(nominal, rows))
            images = {r['id']: cv2.imread(str(source/r['rgb_path'])) for r in rows}
            assert all(image is not None for image in images.values())
            arms, costs[panel] = evaluate(rows, nominal, images)
            assert all(a['candidate'] == b['candidate'] for a, b in zip(arms['combined_flow'], cache['combined_flow']))
            disabled = visual.predict(rows, lambda _: None)
            controls, _ = evaluate(rows, disabled, {})
            assert all(all(p['candidate'] == n['baseline'] for p, n in zip(preds, disabled)) for preds in controls.values())
            target = output/panel; target.mkdir()
            write(target/'predictions.json', arms); write(target/'rgb-disabled.json', controls)
            seals[panel] = dict(capture=str(source), raw_sha256=sha(source/'raw.jsonl'),
                receipt_sha256=sha(source/'receipt.json'), predictions_sha256=sha(target/'predictions.json'),
                disabled_sha256=sha(target/'rgb-disabled.json'))
            packets[panel] = (rows, arms, source)
    assert sum(len(v[0]) for v in packets.values()) == 768
    code = ['mz114_joint_association.py', 'run_mz114_joint_association.py', 'mz113_flow_persistence.py',
            'run_mz113_dynamic_flow.py', 'mz111_spatial_evidence.py', 'mz111_temporal_geometry.py',
            'run_mz111_spatial_temporal.py', 'mz108_competitive_association.py', 'mz109_interval_extent.py',
            'mz107_rgb_association.py', 'run_mz107_four_sensor.py']
    for name in code: shutil.copyfile(Path(__file__).with_name(name), output/name)
    protocol = 'MZ114_PROTOCOL_20260913.md'; shutil.copyfile(Path(__file__).with_name(protocol), output/protocol)
    write(output/'prediction-seal.json', dict(status='ALL_ARMS_SEALED_BEFORE_EVALUATOR_PARSE', panels=seals,
        code_sha256={name: sha(output/name) for name in code}, protocol_sha256=sha(output/protocol)))
    summaries = {}; all_rows = []; all_es = []; all_arms = {}
    for panel, (rows, arms, source) in packets.items():
        es = readrows(source/'evaluator.jsonl'); assert [r['id'] for r in rows] == [e['id'] for e in es]
        labels = [truth(e) for e in es]; summaries[panel] = score(rows, es, arms)
        for arm, predictions in arms.items():
            summaries[panel][arm]['exit_metrics'] = exit_metrics(rows, labels, predictions)
            summaries[panel][arm]['vs_incumbent'] = dict(
                gained_TP=sum(gt and p['candidate'] and not q['candidate'] for gt,p,q in zip(labels,predictions,arms['combined_flow'])),
                lost_TP=sum(gt and not p['candidate'] and q['candidate'] for gt,p,q in zip(labels,predictions,arms['combined_flow'])),
                added_FP=sum(not gt and p['candidate'] and not q['candidate'] for gt,p,q in zip(labels,predictions,arms['combined_flow'])),
                removed_FP=sum(not gt and not p['candidate'] and q['candidate'] for gt,p,q in zip(labels,predictions,arms['combined_flow'])))
        all_rows.extend(dict(r, episode_id=panel+'/'+r['episode_id']) for r in rows); all_es.extend(es)
        for arm, predictions in arms.items(): all_arms.setdefault(arm, []).extend(predictions)
    overall = score(all_rows, all_es, all_arms)
    for arm, predictions in all_arms.items(): overall[arm]['exit_metrics'] = exit_metrics(all_rows, [truth(e) for e in all_es], predictions)
    result = dict(status='FIXED_JOINT_ASSIGNMENT_CONSUMED_STUDY_COMPLETE', frames=768,
        primary='joint_temporal', incumbent='combined_flow', scope='CONSUMED_DEVELOPMENT',
        overall=overall, panels=summaries, seconds_per_frame=costs,
        backend=dict(device='CPU', framework='OpenCV '+cv2.__version__, reason='GPU_BACKEND_UNAVAILABLE',
                     assignment_workload='at most four returns, exact finite combinatorial enumeration',
                     cuda_devices=cv2.cuda.getCudaEnabledDeviceCount(), sparse_lk_binding=hasattr(cv2.cuda,'SparsePyrLKOpticalFlow_create')))
    write(output/'summary.json', result)
    write(output/'completion.json', dict(status='PASS', output_hashes={str(p.relative_to(output)): sha(p) for p in output.rglob('*.json')}, resources_started=[]))
    print(json.dumps({arm: {key: overall[arm][key] for key in ('metrics','events','exit_metrics','lost_nominal_TP')}
                      for arm in ('combined_flow','joint_current','joint_temporal')}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.output)
