"""One sealed 576-frame complete-sequence comparison; no fit or tuning."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import run_core_transfer as source
from full_event_spec_20260920 import specification, check_spec, bounds, classify, PROFILE
from multireturn_pilot_20260920 import simulate_two, T
from tof_fov45_core import boxes45
from tof_corridor_calibration import score_frame, decide
from ba_camera_corridor import sample_native

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/ba-full-event-transfer-20260920'
ARMS = ('strongest_hold', 'closest_hold')
OLD = ('ba-core-transfer-20260920', 'ba-core-workpoint-transfer-20260920', 'ba-core-hold-validation-20260920')
CODE = ('run_full_event_20260920.py', 'full_event_spec_20260920.py', 'full_event_metrics_20260920.py',
        'full_event_capture_20260920.py', 'launch_full_event_20260920.py', 'ue_capture_readiness.py',
        'multireturn_pilot_20260920.py', 'tof_fov45_core.py', 'tof_corridor_calibration.py',
        'ba_camera_corridor.py', 'ba_camera_corridor_spec.py', 'core_transfer_spec.py',
        'run_core_transfer.py', 'evaluate_ba_camera_corridor.py', 'audit_existing_strata_20260920.py',
        'tof_lateral_core.py', 'ba_camera_corridor_metrics.py', 'audit_core_workpoint_20260920.py')
read, sha, require, write = source.read, source.sha, source.require, source.write_new


def apply_hold(current, previous, time_s):
    held = bool(previous is not None and abs(time_s-previous[0]-.2) < 1e-6 and previous[1])
    return bool(current or held), bool(held and not current), (time_s, bool(current))


def freeze(out):
    require(not out.exists(), 'No overwrite or scientific retry')
    spec = specification()
    old_paths = [ROOT/'artifacts.local/work'/name/'spec.json' for name in OLD]
    checked = check_spec(spec, [read(p) for p in old_paths])
    prior = ROOT/'artifacts.local/work/ba-multireturn-pilot-20260920/protocol.json'
    frozen = read(prior)
    for name in ('multireturn_pilot_20260920.py', 'tof_fov45_core.py', 'tof_corridor_calibration.py', 'ba_camera_corridor.py'):
        require(sha(Path(__file__).with_name(name)) == frozen['code'][name], 'Prior mechanism changed: '+name)
    assert T == .4071309640537889
    a, _, state = apply_hold(True, None, 0.)
    b, held, state = apply_hold(False, state, .2)
    c, _, _ = apply_hold(False, state, .4)
    assert a and b and held and not c
    assert not apply_hold(False, (0., True), .4)[0]
    out.mkdir(parents=True)
    write(out/'spec.json', spec)
    write(out/'preflight.json', checked)
    (out/'protocol-before-run.md').write_bytes(Path(__file__).with_name('FULL_EVENT_PROTOCOL_20260920.md').read_bytes())
    dependencies = old_paths + [prior, ROOT/'tools/street_process_lifecycle.py', ROOT/'tools/run_obstacle_research.py']
    # Capture launcher imports lifecycle from the Unreal research route on this checkout.
    dependencies = [p if p.exists() else ROOT/'research/active/dtr-r0/unreal'/p.name for p in dependencies]
    write(out/'protocol.json', dict(id='ba-full-event-transfer-20260920',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frames=576, clips=24, frames_per_clip=24, dt_s=.2, capture_timeout_s=3600,
        threshold=T, arms=ARMS, profile=PROFILE,
        spec_sha256=sha(out/'spec.json'), protocol_text_sha256=sha(out/'protocol-before-run.md'),
        code_hashes={n:sha(Path(__file__).with_name(n)) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
        placement='UE graphics capture; CPU scalar geometry TASK_NOT_GPU_SUITABLE',
        stop='One capture and complete evaluation; no tuning, training or successor'))
    print(json.dumps(dict(status='FROZEN', preflight=checked)))


def check_capture(out):
    p = source.verify(out)
    cap = out/'capture'
    receipt, launch = read(cap/'receipt.json'), read(cap/'launch-receipt.json')
    require(receipt['status'] == 'PASS' and receipt['frame_count'] == p['frames'], 'Complete source required')
    require(receipt['source_unchanged'] and receipt['task_actors_released'] and read(cap/'process-release.json')['released'], 'Source/resource audit')
    require(receipt['protocol_sha256'] == launch['protocol_sha256'] == sha(out/'protocol.json'), 'Capture protocol')
    require(receipt['spec_sha256'] == launch['spec_sha256'] == p['spec_sha256'], 'Capture spec')
    require(receipt['script_sha256'] == launch['capture_script_sha256'] == p['code_hashes']['full_event_capture_20260920.py'], 'Capture script')
    require(launch['launcher_sha256'] == p['code_hashes']['launch_full_event_20260920.py'], 'Launcher identity')
    require(receipt['readiness_helper_sha256'] == launch['readiness_helper_sha256'] == p['code_hashes']['ue_capture_readiness.py'], 'Readiness helper')
    spec = read(out/'spec.json')
    require(receipt['map_sha256_before'] == receipt['map_sha256_after'] == spec['expected_map_sha256'], 'Map changed')
    require(len(receipt['view_readiness']) == p['frames'], 'Readiness coverage')
    for i, r in enumerate(receipt['view_readiness']):
        require(r['sample_index'] == i, 'Readiness ordering')
        source.validate_readiness(r)
    plugin = Path(launch['plugin_path'])
    require(sha(plugin) == launch['plugin_sha256'] and sha(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll') == launch['plugin_binary_sha256'], 'Plugin changed')
    for path, digest in launch['assets'].items():
        require(sha(path) == digest, 'Asset changed')
    manifest, geometry = read(cap/'observations/manifest.json'), read(cap/'evaluator/geometry.json')
    require(len(spec['cases']) == len(manifest['frames']) == len(geometry) == p['frames'], 'Source coverage')
    return spec, manifest, geometry


def seal(out, name, files):
    write(out/name, dict(status='COMPLETE', frames=576, protocol_sha256=sha(out/'protocol.json'),
                        hashes={n:sha(out/n) for n in files}))


def check_seal(out, name):
    s = read(out/name)
    require(s['status'] == 'COMPLETE' and s['frames'] == 576 and s['protocol_sha256'] == sha(out/'protocol.json'), 'Seal identity')
    for n, h in s['hashes'].items():
        require(sha(out/n) == h, 'Sealed file changed: '+n)


def materialize(out):
    spec, manifest, geometry = check_capture(out)
    require(not (out/'materialization-start.json').exists(), 'One materialization only')
    write(out/'materialization-start.json', dict(scope='Hypothetical sensor construction; ownership only in source admission', labels_used_for_returns=False))
    public, identities, private, admission = defaultdict(list), [], [], []
    for i, (case, row, geo) in enumerate(zip(spec['cases'], manifest['frames'], geometry)):
        source.source_check(case, geo, row, i)
        native_path = out/'capture/evaluator'/geo['native_path']
        rgb_path = out/'capture/observations'/row['rgb_path']
        require(sha(native_path) == geo['native_sha256'] and sha(rgb_path) == row['rgb_sha256'] == geo['rgb_sha256'], 'Input binding')
        native = np.load(native_path, allow_pickle=False)
        require(native.shape == (360,640), 'Native shape')
        target, corridor = source.masks(native, case, geo)
        admission.append(dict(id=f'f{i:04d}', target_visible_pixels=int(target.sum()), competing_corridor_pixels=int((corridor & ~target).sum())))
        seed = 'full-event-transfer-v1/'+case['name']
        packet, lineage = simulate_two(sample_native(native), seed, boxes45())
        for k, value in packet.items():
            public[k].append(value)
        identities.append(dict(id=f'f{i:04d}', **{k:row[k] for k in ('clip_id','frame_in_clip','time_s')}))
        private.append(dict(id=f'f{i:04d}', seed=seed, native_sha256=geo['native_sha256'], zones=lineage))
    passed = all(r['target_visible_pixels'] > 0 and r['competing_corridor_pixels'] < 4 for r in admission)
    write(out/'source-admission.json', dict(status='PASS' if passed else 'NOT_EVALUABLE', checks=admission))
    require(passed, 'No admission exclusions, source is NOT_EVALUABLE')
    np.savez_compressed(out/'observations.npz', boxes=boxes45(), **{k:np.stack(v) for k,v in public.items()})
    write(out/'identities.json', identities)
    write(out/'private-lineage.json', private)
    seal(out, 'observation-seal.json', ('observations.npz','identities.json','private-lineage.json','source-admission.json',
                                      'capture/evaluator/geometry.json','capture/observations/manifest.json'))
    print('OBSERVATIONS_SEALED: 576 paired packets')


def predict(out):
    source.verify(out)
    check_seal(out, 'observation-seal.json')
    write(out/'prediction-start.json', dict(evaluator_access=False, models=0, python=sys.executable,
        placement='TASK_NOT_GPU_SUITABLE', current_flags_for_causal_attribution_only=True))
    rows, previous = [], {}
    with np.load(out/'observations.npz', allow_pickle=False) as data:
        for i, identity in enumerate(read(out/'identities.json')):
            values = data['ranges'][i]
            choose_second = np.isfinite(values[:,1]) & (~np.isfinite(values[:,0]) | (values[:,1] < values[:,0]))
            nearest = np.where(choose_second, values[:,1], values[:,0])
            flags, current, unknown, held_only, triggers, scores = {}, {}, {}, {}, {}, {}
            for arm, vector, selected in (('strongest_hold', values[:,0], np.zeros(64, int)),
                                          ('closest_hold', nearest, choose_second.astype(int))):
                scored = score_frame(data['boxes'], vector)
                decision = decide(scored, T)
                current[arm], unknown[arm], scores[arm] = decision['alert'], decision['unknown'], decision['score']
                key = identity['clip_id'], arm
                flags[arm], held_only[arm], previous[key] = apply_hold(current[arm], previous.get(key), identity['time_s'])
                factors = {s['zone']:s for s in scored['zone_scores']}
                triggers[arm] = [dict(zone=a['zone'], slot=int(selected[a['zone']])) for a in scored['anchors']
                    if a['definite'] or (a['possible'] and factors[a['zone']]['joint'] >= T)]
                assert current[arm] == bool(triggers[arm])
            rows.append({**identity, 'flags':flags, 'current':current, 'current_unknown':unknown,
                         'held_only':held_only, 'triggers':triggers, 'scores':scores})
    write(out/'predictions.json', rows)
    seal(out, 'prediction-seal.json', ('predictions.json','prediction-start.json','observation-seal.json'))
    print('PREDICTIONS_SEALED: 576 frames, two arms')


def evaluate(out):
    from full_event_metrics_20260920 import evaluate as event_metrics
    source.verify(out)
    check_seal(out, 'observation-seal.json')
    check_seal(out, 'prediction-seal.json')
    require(not (out/'results.json').exists(), 'One evaluation only')
    spec, manifest, geometry = check_capture(out)
    rows = read(out/'predictions.json')
    private = read(out/'private-lineage.json')
    require(len(rows) == len(private) == len(spec['cases']) == 576, 'Complete evaluation')
    for i, (r, case, geo, lin) in enumerate(zip(rows, spec['cases'], geometry, private)):
        require(r['id'] == lin['id'] == f'f{i:04d}', 'Frame identity')
        require(all(r[k] == case[k] for k in ('clip_id','frame_in_clip','time_s')), 'Timeline binding')
        truth = source.primitive_truth(dict(case=case,geometry=geo), PROFILE)
        declared = classify(*bounds(case))
        require(truth['truth'] == declared['truth'] and truth['boundary'] == declared['boundary'], 'Native full bounds truth mismatch')
        r.update(**declared, **{k:case[k] for k in ('layer','layout_relation','background','type_id','phase')})
        native_path = out/'capture/evaluator'/geo['native_path']
        require(sha(native_path) == geo['native_sha256'] == lin['native_sha256'], 'Native unchanged')
        target, corridor = source.masks(np.load(native_path, allow_pickle=False), case, geo)
        support = sample_native(target & corridor).ravel()
        r['native_trigger_contributors'] = {arm:sum(int(support[lin['zones'][t['zone']]['indices'][t['slot']]].sum())
            for t in r['triggers'][arm]) for arm in ARMS}
        r['native_second_trigger_contributors'] = sum(int(support[lin['zones'][t['zone']]['indices'][1]].sum())
            for t in r['triggers']['closest_hold'] if t['slot'] == 1)
    result = event_metrics(rows)
    masks = dict(overall=lambda r:True, Core=lambda r:r['layout_relation'] != 'BOUNDARY',
                 Boundary=lambda r:r['layout_relation'] == 'BOUNDARY')
    changes = {name: {kind:[r['id'] for r in rows if mask(r) and r['truth'] == truth
                    and r['flags']['strongest_hold'] == before and r['flags']['closest_hold'] != before]
                for kind,truth,before in (('TP_lost',True,True),('FN_recovered',True,False),
                    ('FP_added',False,False),('FP_removed',False,True))} for name,mask in masks.items()}
    baseline, candidate = (result['Core']['arms'][a] for a in ARMS)
    before = {e['clip_id']:e for e in baseline['events']}
    after = {e['clip_id']:e for e in candidate['events']}
    require(set(before) == set(after) and len(before) == 8, 'Eight Core events')
    delayed = [k for k,b in before.items() if b['detected'] and (not after[k]['detected']
        or after[k]['first_in_event_alert_delay_s'] > b['first_in_event_alert_delay_s']+1e-8)]
    earlier = [k for k,b in before.items() if after[k]['detected'] and (not b['detected']
        or after[k]['first_in_event_alert_delay_s'] < b['first_in_event_alert_delay_s']-1e-8)]
    fewer_gaps = [k for k,b in before.items() if after[k]['internal_interruption_count'] < b['internal_interruption_count']]
    native_current_recovery = [r['id'] for r in rows if masks['Core'](r) and r['truth']
        and r['current']['closest_hold'] and not r['current']['strongest_hold']
        and r['native_second_trigger_contributors'] > 0]
    unknown_new_current = [r['id'] for r in rows if masks['Core'](r) and r['truth']
        and r['current']['closest_hold'] and not r['current']['strongest_hold']
        and r['native_second_trigger_contributors'] == 0]
    release_worse = []
    for k,b in before.items():
        c = after[k]
        if c['release_right_censored'] or b['release_right_censored'] or any(
            c[f] is None or b[f] is None or c[f] > b[f]+1e-8 for f in
            ('postexit_carryover_sampled_s','first_silent_relative_to_exit_s')):
            release_worse.append(k)
    checks = dict(nonempty_comparison=baseline['event_count'] == candidate['event_count'] == 8,
        useful_gain=bool(native_current_recovery or earlier or fewer_gaps),
        no_lost_core_TP=not changes['Core']['TP_lost'],
        no_lost_core_event=candidate['detected_events'] >= baseline['detected_events'],
        no_delayed_core_onset=not delayed,
        no_more_core_FP=candidate['frames']['FP'] <= baseline['frames']['FP'],
        no_more_core_false_segments=candidate['false_alert_segment_count'] <= baseline['false_alert_segment_count'],
        no_more_core_false_sampled_s=candidate['false_alert_sampled_s'] <= baseline['false_alert_sampled_s']+1e-8,
        no_more_core_internal_interruptions=all(after[k]['internal_interruption_count'] <= b['internal_interruption_count'] for k,b in before.items()),
        no_worse_core_release=not release_worse)
    result.update(paired_changes=changes, checks=checks, passed=all(checks.values()),
        gate_details=dict(native_backed_current_recovery=native_current_recovery,
            current_recovery_without_native_trigger=unknown_new_current, earlier_core_onsets=earlier,
            delayed_core_onsets=delayed, fewer_core_gaps=fewer_gaps, worse_or_censored_core_release=release_worse),
        scope='ONE_NEW_CONTROLLED_COMPLETE_SEQUENCE_TRANSFER_NOT_HARDWARE_OR_AUDIO', no_retuning=True)
    write(out/'frame-results.json', rows)
    write(out/'results.json', result)
    seal(out, 'evaluation-seal.json', ('results.json','frame-results.json','prediction-seal.json'))
    print(json.dumps(dict(status='EVALUATION_SEALED',passed=result['passed'],checks=checks,
        core={a:result['Core']['arms'][a]['frames'] for a in ARMS})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('freeze','materialize','predict','evaluate'))
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    globals()[args.stage](args.out)
    print(f'{args.stage}: {time.perf_counter()-started:.3f}s')
