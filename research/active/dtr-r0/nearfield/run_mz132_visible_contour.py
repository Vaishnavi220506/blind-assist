"""Single frozen paired oracle-frontend diagnostic; requires admitted UE masks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz126_central_tof import read, write, sha
from mz124_measurement_geometry import metrics
from mz132_contour_adapter import anonymous_components, predict
from mz132_visible_contour_capture import verify_geometry
from research_backend import BackendCandidate, DeviceObservation, select_backend

MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'
BRIEF = Path(__file__).with_name('MZ132_VISIBLE_CONTOUR_BRIEF_20260914.md')


def run(capture, output):
    assert not output.exists() and output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    verification = read(capture/'verification.json')
    assert verification['status'] in ('GEOMETRY_PASS_RGB_IDENTICAL','GEOMETRY_PASS_RGB_NOT_IDENTICAL'), 'No ideal-frontend score from unadmitted renderer output'
    manifest = read(capture/'manifest.json')
    receipt = read(capture/'receipt.json')
    assert receipt['status'] == 'PASS' and manifest['sensor_calls'] == 0
    launch = read(capture/'launch.json')
    source_receipt = read(SOURCE/'capture-v1/receipt.json')
    assert launch['frozen_spec_sha256'] == source_receipt['spec_sha256']
    assert launch['source_sha256'] == source_receipt['source_sha256']
    assert sha(capture/'mz115_zonal_capture.py') == launch['instrumented_sha256'] == receipt['source_sha256']
    assert sha(capture/'spec.json') == launch['spec_sha256'] == receipt['spec_sha256']
    assert read(capture/'spec.json') == read(SOURCE/'capture-v1/spec.json')
    assert receipt['engine_version'] == source_receipt['engine_version']
    smoke = read(capture/'smoke-geometry-admission.json')
    assert sha(capture/'smoke-geometry-admission.json') == launch['smoke_admission_sha256']
    assert smoke['status'] == 'GEOMETRY_SMOKE_PASS'
    assert smoke['cross_host_masks_exact'] and smoke['all_geometry_readbacks_pass']
    raw = SOURCE/'capture-v1/raw.jsonl'
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    assert len(rows) == len(manifest['frames']) == 288
    assert [r['id'] for r in rows] == [r['id'] for r in manifest['frames']]
    assert [r['id'] for r in rows] == [r['id'] for r in verification['frames']]
    cache = read(CORRECTION/'predictions.json')
    old_radar = read(MZ129/'radar-v1/predictions.json')
    baseline = read(MZ129/'replay-v1r2/predictions.json')['radar']
    paths = [raw, CORRECTION/'predictions.json', MZ129/'radar-v1/predictions.json',
             MZ129/'replay-v1r2/predictions.json', capture/'verification.json',
             capture/'manifest.json', capture/'receipt.json', capture/'launch.json']
    paths += [capture/'mz115_zonal_capture.py',capture/'spec.json']
    paths += [capture/'smoke-geometry-admission.json',SOURCE/'capture-v1/evaluator.jsonl']
    source_spec = read(SOURCE/'capture-v1/spec.json')
    scene_frames = {f['id']:f for f in source_spec['frames']}
    assert sha(SOURCE/'capture-v1/evaluator.jsonl') == source_receipt['hashes']['evaluator.jsonl']
    source_evaluations = {e['id']:e for e in map(json.loads,(SOURCE/'capture-v1/evaluator.jsonl').read_text().splitlines())}
    correction_seal = read(CORRECTION/'prediction-seal.json')
    assert sha(raw) == correction_seal['raw_sha256']
    assert sha(CORRECTION/'predictions.json') == correction_seal['predictions_sha256']
    paths.append(CORRECTION/'prediction-seal.json')
    for directory in (MZ129/'radar-v1',MZ129/'replay-v1r2'):
        prior = read(directory/'prediction-seal.json')
        assert sha(directory/'predictions.json') == prior['predictions_sha256']
        for key in ('input_hashes','dependencies','source_hashes'):
            for path,digest in prior.get(key,{}).items():
                assert sha(ROOT/path) == digest,path
        paths.append(directory/'prediction-seal.json')
    # Reauthenticate every categorical raster and original RGB; the stored
    # decoded instance-id PNG is not trusted independently of its color pass.
    components = []
    rgb_root = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/rgb-v1'
    original_receipt = read(SOURCE/'capture-v1/receipt.json')
    for frame, admission in zip(manifest['frames'],verification['frames']):
        geometry = verify_geometry(frame,scene_frames[frame['id']],source_spec,source_evaluations[frame['id']])
        assert geometry['passed'] and geometry == admission['geometry']
        rgb_path = capture/frame['paths']['rgb']
        mask_path = capture/frame['paths']['instance_rgb']
        original = rgb_root/frame['paths']['rgb']
        assert sha(original) == original_receipt['hashes'][frame['paths']['rgb']]
        assert sha(rgb_path) == receipt['hashes'][frame['paths']['rgb']]
        assert sha(mask_path) == receipt['hashes'][frame['paths']['instance_rgb']]
        rgb = cv2.imread(str(rgb_path)); old_rgb = cv2.imread(str(original))
        assert rgb is not None and old_rgb is not None and rgb.shape == old_rgb.shape == (360,640,3)
        assert bool(np.array_equal(rgb,old_rgb)) == admission['rgb_equal']
        colors = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
        labels = np.zeros(colors.shape[:2], np.uint8)
        recognized = np.all(colors == 0, axis=-1)
        for index, color in enumerate(frame['palette'],1):
            selected = np.all(colors == color, axis=-1)
            labels[selected] = index
            recognized |= selected
        assert recognized.all() and labels.shape == (360,640) and np.any(labels)
        components.append(anonymous_components(labels))
        paths.extend([rgb_path,mask_path,original])
    output.mkdir(parents=True)
    select_backend('scalar-scoring',cpu=BackendCandidate('anonymous-contour-interval-readout','cpu',
        lambda: bool(components),lambda _:DeviceObservation('cpu','host CPU','Python/NumPy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE',record_path=output/'backend.json')
    anonymous_path = output/'anonymous-candidates.json'
    write(anonymous_path, [dict(id=r['id'],components=c) for r,c in zip(rows,components)])
    # Reload only the whitelisted 2D export for the prediction boundary.
    components = [r['components'] for r in read(anonymous_path)]
    arms = dict(baseline=baseline); details = {}; elapsed = {}
    for name, use_masks in (('oracle_boxes',False),('oracle_masks',True)):
        started = time.perf_counter()
        arms[name], details[name] = predict(rows,cache,old_radar,components,use_masks)
        elapsed[name] = time.perf_counter()-started
    write(output/'predictions.json',arms)
    write(output/'support-details.json',details)
    sources = [p for p in Path(__file__).parent.glob('*.py') if p.name in {
        'run_mz132_visible_contour.py','mz132_contour_adapter.py','mz132_visible_contour_capture.py',
        'mz115_spatial_allocation.py','mz111_spatial_evidence.py','mz116_radar_resolution_guard.py',
        'mz128_zone_weighting.py','mz130_local_masks.py','mz130_local_support.py'}]
    paths += sources+[BRIEF]
    write(output/'prediction-seal.json',dict(input_hashes={str(p):sha(p) for p in paths},
        predictions_sha256=sha(output/'predictions.json'), support_sha256=sha(output/'support-details.json'),
        anonymous_sha256=sha(anonymous_path), seconds=elapsed,
        backend='CPU: TASK_NOT_GPU_SUITABLE for scalar interval association/readout',
        authority='IDEAL_VISIBLE_FRONTEND_CONSUMED_DIAGNOSTIC_PREDICTIONS_SAVED_BEFORE_SCORE'))
    score(output,rows,arms,details)


def score(output, rows, arms, details):
    from mz115_allocation_audit import native_truth, project_point, contains, possible, native_bounds
    evaluator_path = SOURCE/'capture-v1/evaluator.jsonl'
    evaluator = [json.loads(s) for s in evaluator_path.read_text().splitlines()]
    provenance_path = SOURCE/'capture-v1/provenance.jsonl'
    provenance = [json.loads(s) for s in provenance_path.read_text().splitlines()]
    old_radar = read(MZ129/'radar-v1/predictions.json')
    report = read(SOURCE/'analysis-v1/frame-report.json')
    assert [r['id'] for r in rows] == [r['id'] for r in evaluator] == [r['id'] for r in report]
    assert [r['id'] for r in rows] == [r['id'] for r in provenance]
    truth = [native_truth(e) for e in evaluator]
    assert truth == [r['truth'] for r in report]
    before = [r['candidate'] for r in arms['baseline']]
    old_events = events(rows,truth,before)
    result = dict(authority='IDEAL_FRONTEND_CONSUMED_SIMULATION_NOT_DEPLOYABLE_RESULT', frames=288, arms={})
    for name,predictions in arms.items():
        flags = [r['candidate'] for r in predictions]
        value = metrics(rows,truth,flags); value['events'] = events(rows,truth,flags)
        value['lost_TP_ids'] = [r['id'] for r,t,b,a in zip(rows,truth,before,flags) if t and b and not a]
        value['new_FP_ids'] = [r['id'] for r,t,b,a in zip(rows,truth,before,flags) if not t and not b and a]
        value['event_times_preserved_or_earlier'] = len(value['events']) == len(old_events) and all(
            a['episode'] == b['episode'] and a['onset_s'] == b['onset_s'] and
            a['first_alert_s'] is not None and a['first_alert_s'] <= b['first_alert_s']
            for a,b in zip(value['events'],old_events))
        value['families'] = {}
        for family in sorted({r['family'] for r in report}):
            ix = [i for i,r in enumerate(report) if r['family'] == family]
            value['families'][family] = metrics([rows[i] for i in ix],[truth[i] for i in ix],[flags[i] for i in ix])
        result['arms'][name] = value
        if name == 'baseline':
            continue
        native = []; impact = []
        for row,ev,pr,old_rb,label,old,pred,detail in zip(rows,evaluator,provenance,old_radar,report,arms['baseline'],predictions,details[name]):
            actors = {row['episode_id']+'/'+a['name']:a for a in ev['native_bounds']}
            for ret in detail['tof']['returns']:
                if not ret['matched']:
                    continue
                zone = next(z for z in ev['zonal_tof_native'] if z['zone_id'] == ret['zone_id'])
                lineage = next((l for l in zone['returned_lineage'] if l['target_index'] == ret['slot']),None)
                assert lineage and lineage['hit_indices'], 'Missing native lineage for matched return'
                for index in lineage['hit_indices']:
                    hit = zone['private_rays'][index]; point = hit['hit_point_m']
                    pixel = project_point(point,ev['camera'],row['rgb_intrinsics'])
                    assert pixel is not None, 'Projection unavailable; do not treat as retained'
                    body = [a-b for a,b in zip(point,ev['body_origin_m'])]
                    actor = actors.get(hit.get('actor_id'))
                    native.append(dict(id=row['id'],sensor='TOF',zone_id=ret['zone_id'],slot=ret['slot'],sample=index,
                        pixel=pixel, previously_contained=contains(ret['old_roi'],pixel),
                        retained=any(contains(t,pixel) for t in ret['tiles']),
                        outside_rgb=not (0<=pixel[0]<=640 and 0<=pixel[1]<=360),
                        corridor_point=possible([(v,v) for v in body]),
                        hazardous_actor=bool(actor and possible(native_bounds(actor,ev['body_origin_m'])))))
            previous_returns = {r['slot']:r for r in old_rb['corrected_current_evidence']}
            for ret in detail['radar']['returns']:
                previous = previous_returns[ret['slot']]
                if ret['proposal'] is None or previous['proposal'] is None:
                    continue
                origin = pr['radar_slots'][ret['slot']]
                actor = actors.get(origin.get('actor_id')) if origin else None
                assert origin and 'kind' in origin, 'Radar provenance unavailable'
                if origin['kind'] == 'real_actor':
                    assert actor is not None, 'Real Radar actor lookup unavailable'
                else:
                    assert origin['kind'] == 'persistent_ghost', 'Unknown Radar source kind is not negative evidence'
                    # Ghosts have no physical surface point; label this source
                    # type explicitly rather than manufacturing a native hit.
                    native.append(dict(id=row['id'],sensor='RADAR_NONPHYSICAL',slot=ret['slot'],
                        source_kind=origin['kind'],previously_contained=False,
                        retained=False,corridor_point=False,hazardous_actor=False,outside_rgb=False))
                    continue
                camera = [ev['camera'][k] for k in ('x','y','z')]
                delta = [c-o for c,o in zip(actor['center_m'],camera)]
                scale = origin['pre_noise_range_m']/math.hypot(delta[0],delta[1])
                point = [c+scale*d for c,d in zip(camera,delta)]
                pixel = project_point(point,ev['camera'],row['rgb_intrinsics'])
                assert pixel is not None
                body = [a-b for a,b in zip(point,ev['body_origin_m'])]
                native.append(dict(id=row['id'],sensor='RADAR_REPRESENTATIVE_PROXY',slot=ret['slot'],
                    pixel=pixel,previously_contained=contains(previous['box'],pixel),
                    retained=any(contains(t,pixel) for t in ret['angular_tiles']),
                    corridor_point=possible([(v,v) for v in body]),
                    hazardous_actor=possible(native_bounds(actor,ev['body_origin_m'])),
                    outside_rgb=not (0<=pixel[0]<=640 and 0<=pixel[1]<=360)))
            if label['truth'] or not old['candidate']:
                continue
            returns = detail['tof']['returns']
            triggering = [r for r in returns if r['old_possible'] and r['weight']>0]
            rb = detail['radar']
            impact.append(dict(id=row['id'],family=label['family'],
                new_tof_associations=sum(r['new_association'] for r in returns),
                new_triggering_tof_associations=sum(r['new_association'] for r in triggering),
                matched_triggering_returns=sum(r['matched'] for r in triggering),
                removed_triggering_possible_bits=sum(not r['possible'] for r in triggering),
                remaining_positive_weight_returns=sum(r['possible'] and r['weight']>0 for r in returns),
                new_radar_associations=sum(r['new_association'] for r in rb['returns']),
                removed_radar_supports=sum(r['old_support'] and not r['support'] for r in rb['returns']),
                old_score=old['score'],new_score=pred['score'],certain_coarse=pred['certain_coarse'],
                current_radar=any(r['support'] for r in rb['returns']),raw_radar=rb['raw_center_support'],
                carry=rb['inherited_carry_support'],guards=bool(rb['guard_events']),alert_removed=not pred['candidate']))
        dropped = [r for r in native if r['previously_contained'] and not r['retained']]
        value['native'] = dict(audited_samples=len(native), newly_dropped_samples=len(dropped),
            newly_dropped_corridor_samples=sum(r['corridor_point'] for r in dropped),
            newly_dropped_hazard_actor_samples=sum(r['hazardous_actor'] for r in dropped),
            newly_dropped_outside_rgb_samples=sum(r['outside_rgb'] for r in dropped))
        value['native']['by_sensor'] = {sensor:dict(audited=sum(r['sensor']==sensor for r in native),
            dropped=sum(r['sensor']==sensor for r in dropped),
            dropped_corridor=sum(r['sensor']==sensor and r['corridor_point'] for r in dropped))
            for sensor in sorted({r['sensor'] for r in native})}
        value['fp_support_impact'] = {key:sum(bool(r[key]) for r in impact) for key in (
            'new_tof_associations','new_triggering_tof_associations','matched_triggering_returns',
            'removed_triggering_possible_bits','new_radar_associations','removed_radar_supports','alert_removed')}
        value['fp_support_impact_units'] = 'frames_with_condition'
        value['association_totals'] = dict(
            matched_tof_returns=sum(r['matched'] for d in details[name] for r in d['tof']['returns']),
            new_tof_associations=sum(r['new_association'] for d in details[name] for r in d['tof']['returns']),
            new_radar_associations=sum(r['new_association'] for d in details[name] for r in d['radar']['returns']))
        value['fp_impact_families'] = {family:{key:sum(bool(r[key]) for r in impact if r['family']==family)
            for key in value['fp_support_impact']} for family in value['families']}
        value['go_criterion'] = bool(value['families']['substantial_body']['FP']<=15 and
            not value['lost_TP_ids'] and not value['new_FP_ids'] and value['event_times_preserved_or_earlier'] and
            not value['native']['newly_dropped_corridor_samples'])
        write(output/(name+'-native-retention.json'),native)
        write(output/(name+'-fp-trigger-impact.json'),impact)
    write(output/'summary.json',result)
    seal = read(output/'prediction-seal.json')
    for path,digest in seal['input_hashes'].items():
        assert sha(Path(path)) == digest,path
    assert sha(output/'predictions.json') == seal['predictions_sha256']
    assert sha(output/'support-details.json') == seal['support_sha256']
    write(output/'completion.json',dict(status='PASS',summary_sha256=sha(output/'summary.json'),
        evaluator_sha256=sha(evaluator_path),provenance_sha256=sha(provenance_path),
        resources='No persistent prediction process or allocation'))
    print(json.dumps({n:{k:v for k,v in m.items() if k in ('TP','FP','FN','go_criterion','native','fp_support_impact')} for n,m in result['arms'].items()},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.capture,args.output)
