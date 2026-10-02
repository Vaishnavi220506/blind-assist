"""Frozen existing-phone category-hint diagnostic, consumed Development only.

Inference observes RGB only. Evaluator-only visible instance boxes and query
geometry measure coverage; they never filter predictions or supply categories.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import ndimage

import cnh_rgb_visible_query_gate as P
import cnh_rgb_visible_query as V
import cnh_rgb_range_anchor as A

ROOT = P.ROOT
OUT = ROOT/'artifacts.local/work/cnh-rgb-semantic-hints-20261002'
RUN_ID = 'CNH_RGB_SEMANTIC_HINTS_20261002'
MODEL = ROOT/'app/src/main/assets/yolo11n_fp16_320.tflite'
LABELS = ROOT/'app/src/main/assets/coco_labels.txt'
# COCO indices refer to the exact phone labels file, not sparse COCO category IDs.
MAPPING = {4: 59, 5: 56, 6: 57, 24: 72, 25: 62, 31: 0, 33: 61, 34: 71}


def prepare():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Preserve the original frozen plan')
    manifest = {r['id']: r for r in P.read(P.SOURCE/'manifest.json')}
    old = P.read(P.OUT/'PLAN.json')
    rows = []
    for source in old['inputs']:
        m = manifest[source['id']]
        depth = ROOT/source['depth_path']
        semantic = depth.with_name(depth.name.replace('depth_meters', 'semantic'))
        instance = depth.with_name(depth.name.replace('depth_meters', 'semantic_instance'))
        rgb = P.DATA/m['rgb']
        assert P.sha(rgb) == m['rgb_sha256']
        assert P.sha(depth) == source['depth_sha256']
        rows.append(dict(id=source['id'], scene=source['scene'], family=source['family'],
                         split=source['split'], camera_matrix=source['camera_matrix'],
                         rgb_path=str(rgb), rgb_sha256=m['rgb_sha256'],
                         depth_path=str(depth), depth_sha256=source['depth_sha256'],
                         semantic_path=str(semantic), semantic_sha256=P.sha(semantic),
                         instance_path=str(instance), instance_sha256=P.sha(instance)))
    assert len(rows) == 343
    prereg = (f'| 2026-10-02 | {RUN_ID} | PRE_RUN; existing phone YOLO11n fp16 320, '
              'conf>=.35 / same-class NMS IoU>.45; fixed 343 consumed Development frames '
              '(193 cal / 150 eval), no fitting; exact NYU40→COCO8; full-image visible '
              'instance >=16 pixels and class-aware confidence-greedy IoU>=.5 match before '
              'query stratification; preserve unsupported/unknown and every prediction; '
              'HEAD/BODY visible 0.6–2.1m coverage only, not clearance/alarm gain | NOT_RUN | '
              'Diagnostic category coverage; no threshold tuning, no automatic promotion | '
              '`artifacts.local/work/cnh-rgb-semantic-hints-20261002/REPORT.md` |')
    text = P.RUNS.read_text(encoding='utf8')
    assert RUN_ID not in text
    P.RUNS.write_text(text.rstrip()+'\n'+prereg+'\n', encoding='utf8')
    (OUT/'prerun-row.txt').write_text(prereg+'\n', encoding='utf8')
    P.save(OUT/'PLAN.json', dict(run_id=RUN_ID, frozen_utc=datetime.now(timezone.utc).isoformat(),
        role='Consumed Development, no confirmation; retained old family cal/eval split, no training/fitting',
        exact_mapping=MAPPING, mapping_note='books/book granularity ambiguous; table/desk are not necessarily dining table; all remain unsupported',
        model_sha256=P.sha(MODEL), labels_sha256=P.sha(LABELS),
        taxonomy_sha256=P.sha(OUT/'nyu40-labels.csv'), inputs=rows,
        matching='all full-image predictions score descending, same class max IoU >= .5, one-to-one; no query/GT filtering of predictions',
        reference='visible instance >=16 pixels, full-image bbox including semantic-invalid pixels; any invalid or mixed semantic IDs are UNKNOWN; instance 0 is valid',
        queries='same camera-relative 45x45 FOV, HEAD y[-.2,.42] BODY [.42,.9], Z[.6,2.1), |X|<.3 contact, |X|<=.4 expanded; >=16 query pixels',
        missing='unmatched predictions retained, semantic-valid box fraction and class-agnostic overlaps reported; not all are false positives',
        interpretation='descriptive mapped-object recall plus all-object coverage ceiling and full prediction burden; no predeclared pass score or model selection',
        source_sha256={str(Path(p).relative_to(ROOT)): P.sha(p) for p in (
            __file__, P.__file__, V.__file__, A.__file__,
            Path(__file__).with_name('cnh_rgb_semantic_detector.py'))},
        preregistration=prereg))
    P.save(OUT/'observations.json', [dict(id=r['id'], rgb_path=r['rgb_path'], rgb_sha256=r['rgb_sha256']) for r in rows])
    print(json.dumps(dict(prepared=len(rows), exact_mapping=MAPPING)))


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    size = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    area = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection/area if area > 0 else 0.


def match(objects, predictions):
    """Only exact taxonomy and full-image geometry enter matching."""
    taken = set()
    for j in sorted(range(len(predictions)), key=lambda k: (-predictions[k]['score'], k)):
        p = predictions[j]
        options = [(iou(p['box'], o['box']), i) for i, o in enumerate(objects)
                   if i not in taken and o['coco_class'] == p['class_id']]
        value, index = max(options, default=(0., -1), key=lambda x: (x[0], -x[1]))
        p['matched_object_index'] = index if value >= .5 else None
        p['match_iou'] = value if value >= .5 else None
        if value >= .5:
            taken.add(index)
            objects[index]['prediction_index'] = j
    return taken


def hdf(path):
    with h5py.File(path) as f:
        return f['dataset'][:]


def evaluate_frame(row, predictions):
    radial = hdf(row['depth_path'])
    sem = hdf(row['semantic_path']).astype(np.int32)
    inst = hdf(row['instance_path']).astype(np.int32)
    assert radial.shape == sem.shape == inst.shape == (768, 1024)
    g = A.whole_geometry(V.ray_geometry(radial.shape, row['camera_matrix']))
    z = radial/g['radial_factor']
    valid_depth = np.isfinite(z) & (z > 0)
    x, y = np.abs(z*g['fx']), z*g['fy']
    sem_valid = (sem >= 1) & (sem <= 40)
    mask = inst >= 0
    slices = ndimage.find_objects(np.where(mask, inst+1, 0))
    objects, small = [], []
    query_masks = []
    queries = []
    truth = A.visible_truth(z, g)
    for k, q in enumerate(g['queries']):
        near = valid_depth & g['fov_mask'] & (z >= .6) & (z < 2.1) & (y >= q['y_low']) & (y <= q['y_high'])
        contact, expanded = near & (x < .3), near & (x <= .4)
        query_masks.append((contact, expanded))
        semantic_support = []
        for c in range(1, 41):
            cmask = sem == c
            semantic_support.append(dict(nyu40=c, contact_pixels=int((contact & cmask).sum()),
                expanded_pixels=int((expanded & cmask).sum()),
                contact_without_instance=int((contact & cmask & (inst < 0)).sum())))
        queries.append(dict(name=q['name'], contact_pixels=int(contact.sum()),
            expanded_pixels=int(expanded.sum()), semantic_support=semantic_support,
            unannotated_contact_pixels=int((contact & ~sem_valid).sum()),
            native_clipped=bool(g['query_native_clipped'][k]),
            coverage=float(truth['coverage'][k]), abstain=bool(truth['abstain'][k])))
    for identity, sl in enumerate(slices):
        if sl is None:
            continue
        local = mask[sl] & (inst[sl] == identity)
        count = int(local.sum())
        ys, xs = sl
        classes, counts = np.unique(sem[sl][local], return_counts=True)
        klass = int(classes[0]) if len(classes) == 1 and 1 <= classes[0] <= 40 else None
        support = [dict(contact_pixels=int((c[sl] & local).sum()), expanded_pixels=int((e[sl] & local).sum()))
                   for c, e in query_masks]
        obj = dict(instance_id=identity, nyu40=klass, semantic_ids=classes.tolist(),
                   semantic_counts=counts.tolist(), pixels=count,
                   semantic_valid_pixels=int((sem_valid[sl] & local).sum()),
                   coco_class=MAPPING.get(klass), box=[xs.start, ys.start, xs.stop, ys.stop],
                   query_support=support, prediction_index=None)
        (objects if count >= 16 else small).append(obj)
    match(objects, predictions)
    h, w = sem.shape
    for p in predictions:
        x0, y0, x1, y1 = p['box']
        xx0, yy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        xx1, yy1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        region = sem_valid[yy0:yy1, xx0:xx1]
        p['semantic_valid_box_fraction'] = float(region.mean()) if region.size else None
        overlap, index = max(((iou(p['box'], o['box']), i) for i, o in enumerate(objects)), default=(0., -1))
        p['class_agnostic_best_iou'] = overlap
        p['class_agnostic_best_object_index'] = index if index >= 0 else None
    return dict(id=row['id'], scene=row['scene'], family=row['family'], split=row['split'],
                objects=objects, small_instances=small, predictions=predictions, queries=queries)


def summarize(frames):
    counters = Counter(frames=len(frames))
    classes = {str(c): Counter() for c in range(1, 41)}
    for f in frames:
        counters['predictions'] += len(f['predictions'])
        counters['matched_predictions'] += sum(p['matched_object_index'] is not None for p in f['predictions'])
        counters['unmatched_predictions'] += sum(p['matched_object_index'] is None for p in f['predictions'])
        counters['unmatched_low_annotation'] += sum(p['matched_object_index'] is None and (p['semantic_valid_box_fraction'] or 0) < .8 for p in f['predictions'])
        counters['small_instance_views'] += len(f['small_instances'])
        counters['small_instance_query_support_1_to_15'] += sum(0 < q['contact_pixels'] < 16 for o in f['small_instances'] for q in o['query_support'])
        for o in f['objects']:
            mapped, hit = o['coco_class'] is not None, o['prediction_index'] is not None
            contact = any(q['contact_pixels'] >= 16 for q in o['query_support'])
            expanded = any(q['expanded_pixels'] >= 16 for q in o['query_support'])
            for name, selected in [('all', True), ('contact', contact), ('expanded', expanded)]:
                if selected:
                    counters[name+'_object_views'] += 1
                    counters[name+'_mapped_object_views'] += mapped
                    counters[name+'_matched_object_views'] += hit
                    counters[name+'_mixed_unknown_object_views'] += o['nyu40'] is None
                    if o['nyu40'] is not None:
                        classes[str(o['nyu40'])][name+'_views'] += 1
                        classes[str(o['nyu40'])][name+'_hits'] += hit
            counters['contact_object_query_views'] += sum(q['contact_pixels'] >= 16 for q in o['query_support'])
            counters['contact_matched_object_query_views'] += sum(q['contact_pixels'] >= 16 for q in o['query_support']) * hit
            counters['object_query_support_1_to_15'] += sum(0 < q['contact_pixels'] < 16 for q in o['query_support'])
            counters['contact_object_views_with_known_geometry'] += any(q['contact_pixels'] >= 16 and not f['queries'][j]['abstain'] for j, q in enumerate(o['query_support']))
            counters['contact_object_views_with_abstain_geometry'] += any(q['contact_pixels'] >= 16 and f['queries'][j]['abstain'] for j, q in enumerate(o['query_support']))
        for q in f['queries']:
            counters['queries'] += 1
            counters['query_geometry_abstain'] += q['abstain']
            counters['query_native_clipped'] += q['native_clipped']
            counters['queries_with_contact_support'] += q['contact_pixels'] >= 16
            counters['unannotated_contact_pixels'] += q['unannotated_contact_pixels']
            for c in q['semantic_support']:
                cc = classes[str(c['nyu40'])]
                cc['contact_pixels'] += c['contact_pixels']
                cc['contact_queries'] += c['contact_pixels'] >= 16
                cc['contact_without_instance_pixels'] += c['contact_without_instance']
    counters['unique_scene_instances'] = len({(f['scene'], o['instance_id']) for f in frames for o in f['objects']})
    return dict(counts=dict(counters), by_nyu40={k: dict(v) for k, v in classes.items()})


def evaluate():
    if any((OUT/p).exists() for p in ('frame-ledger.json', 'result.json')):
        raise FileExistsError('Preserve existing evaluated outputs')
    plan = P.read(OUT/'PLAN.json')
    for path, expected in plan['source_sha256'].items():
        assert P.sha(ROOT/path) == expected, path
    assert P.sha(MODEL) == plan['model_sha256'] and P.sha(LABELS) == plan['labels_sha256']
    assert P.sha(OUT/'nyu40-labels.csv') == plan['taxonomy_sha256']
    receipt = P.read(OUT/'predictions/inference-result.json')
    assert receipt['frames'] == len(plan['inputs']) == 343
    assert receipt['provenance']['model_sha256'] == plan['model_sha256']
    assert receipt['provenance']['labels_sha256'] == plan['labels_sha256']
    assert receipt['observations_sha256'] == P.sha(OUT/'observations.json')
    frames = []
    started = time.perf_counter()
    for index, row in enumerate(plan['inputs']):
        for name in ('depth', 'semantic', 'instance'):
            assert P.sha(row[name+'_path']) == row[name+'_sha256']
        source = OUT/'predictions'/receipt['outputs'][row['id']]['receipt']
        assert P.sha(source) == receipt['outputs'][row['id']]['receipt_sha256']
        p = P.read(source)
        assert p['id'] == row['id'] and p['rgb_sha256'] == row['rgb_sha256']
        for detection in p['detections']:
            detection['box'] = detection['xyxy']
        frames.append(evaluate_frame(row, p['detections']))
        if (index+1) % 50 == 0:
            print(json.dumps(dict(evaluated=index+1)), flush=True)
    P.save(OUT/'frame-ledger.json', frames)
    summary = {s: summarize([f for f in frames if s == 'all' or f['split'] == s]) for s in ('cal', 'eval', 'all')}
    P.save(OUT/'result.json', dict(run_id=RUN_ID, role=plan['role'], groups=summary,
        evaluation_seconds=time.perf_counter()-started, evaluator_sha256=P.sha(__file__),
        observation_inference='existing phone model, RGB only, no threshold fit',
        limitations=['Offline port, not device parity', 'Visible bounding boxes, not precise shape or clearance',
            'Unsupported and unannotated classes preclude full precision or class-agnostic coverage claims',
            'Frame and query views are correlated; counts are descriptive, not independent trials']))
    print(json.dumps({s: v['counts'] for s, v in summary.items()}))


def selftest():
    objects = [dict(box=[0, 0, 10, 10], coco_class=56, prediction_index=None),
               dict(box=[20, 0, 30, 10], coco_class=None, prediction_index=None)]
    preds = [dict(box=[0, 0, 10, 10], class_id=56, score=.9),
             dict(box=[0, 0, 10, 10], class_id=56, score=.8),
             dict(box=[20, 0, 30, 10], class_id=56, score=.7)]
    assert match(objects, preds) == {0}
    assert [p['matched_object_index'] for p in preds] == [0, None, None]
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0
    assert iou([0, 0, 10, 10], [0, 0, 5, 10]) == .5
    print('PASS: one-to-one duplicates, unsupported class, boundary IoU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'evaluate', 'selftest'))
    globals()[parser.parse_args().action]()
