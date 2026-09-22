"""Fixed source construction; appearance pairing is audited before inference."""
from pathlib import Path
from collections import defaultdict
import time
import cv2
import numpy as np
from query_occupancy_data import (stage_path,read,write,sha,new_stage_directory,
    observation_tokens,camera_bounds,geometric_labels,visible_labels,QUERIES,EDGES)


def audit_pairs(out):
    ids=read(out/'observations/identities.json')
    rgb=np.load(out/'observations/rgb.npy',mmap_mode='r')
    tof=np.load(out/'observations/tof.npy',mmap_mode='r')
    labels=dict(np.load(out/'labels/evaluation.npz',allow_pickle=False))
    assert np.array_equal(labels['indices'],np.arange(len(ids)))
    groups=defaultdict(dict)
    for i,row in enumerate(ids):
        assert row['appearance'] not in groups[row['appearance_pair_id']]
        groups[row['appearance_pair_id']][row['appearance']]=i
    pairs=[]
    for pair,rows in sorted(groups.items()):
        assert set(rows)=={'base','changed'}
        a,b=rows['base'],rows['changed']
        delta=np.abs(rgb[a].astype(np.int16)-rgb[b].astype(np.int16))
        pairs.append(dict(pair_id=pair,base_index=a,changed_index=b,
            tof_equal=bool(np.array_equal(tof[a],tof[b])),
            classes_equal=bool(np.array_equal(labels['classes'][a],labels['classes'][b])),
            valid_equal=bool(np.array_equal(labels['valid'][a],labels['valid'][b])),
            rgb_changed=bool(delta.any()),rgb_mean_absolute_delta_normalized=float(delta.mean()/255)))
    summary=dict(pairs=len(pairs),tof_equal=sum(p['tof_equal'] for p in pairs),
        classes_equal=sum(p['classes_equal'] for p in pairs),valid_equal=sum(p['valid_equal'] for p in pairs),
        rgb_changed=sum(p['rgb_changed'] for p in pairs),
        mean_rgb_delta=float(np.mean([p['rgb_mean_absolute_delta_normalized'] for p in pairs])),
        all_query_labels_valid=bool(labels['valid'].all()))
    summary['admissible']=bool(len(ids)==576 and len(pairs)==288 and all(
        p['tof_equal'] and p['classes_equal'] and p['valid_equal'] and p['rgb_changed'] for p in pairs)
        and summary['mean_rgb_delta']>=1/255 and summary['all_query_labels_valid'])
    return dict(summary=summary,pairs=pairs,scope='Visual appearance under fixed hypothetical ToF, not hardware reflectance response')


def materialize(root):
    from ba_camera_corridor import sample_native
    from tof_fov45_core import boxes45, simulate
    from tof_corridor_calibration import score_frame, decide
    root = Path(root)
    cap = stage_path(root, 'capture')
    spec = read(root/'plan/spec.json')
    receipt = read(cap/'receipt.json')
    launch = read(cap/'launch-receipt.json')
    assert receipt['status'] == 'PASS' and read(cap/'process-release.json')['released']
    assert receipt['source_unchanged'] and receipt['task_actors_released']
    assert receipt['spec_sha256'] == launch['spec_sha256'] == sha(root/'plan/spec.json')
    assert receipt['protocol_sha256'] == launch['protocol_sha256'] == sha(root/'plan/protocol.json')
    manifests = read(cap/'observations/manifest.json')['frames']
    geometry = read(cap/'evaluator/geometry.json')
    cases = spec['cases']
    assert len(cases) == len(manifests) == len(geometry) == spec['frames']
    out = stage_path(root, 'prepared')
    new_stage_directory(out)
    (out/'observations').mkdir()
    (out/'labels').mkdir()
    boxes = boxes45()
    n = len(cases)
    rgb = np.lib.format.open_memmap(out/'observations/rgb.npy', mode='w+', dtype=np.uint8, shape=(n, 3, 180, 320))
    tof = np.lib.format.open_memmap(out/'observations/tof.npy', mode='w+', dtype=np.float32, shape=(n, 64, 6))
    buffers = {s: [] for s in ('train', 'dev', 'evaluation')}
    labels = {s: dict(classes=[], distances=[], mask=[], coverage=[], valid=[]) for s in buffers}
    identities, audits, start = [], [], time.perf_counter()
    for i, (case, row, geo) in enumerate(zip(cases, manifests, geometry)):
        assert case['name'] == row['id'] == geo['id'] and row['sample_index'] == geo['sample_index'] == i
        assert case['camera'] == geo['declared_camera']
        assert max(abs(geo['actual_camera_location_m'][j]-case['camera'][k]) for j, k in enumerate(('x','y','z'))) < .002
        for planned, actual in zip(case['objects'], geo['objects']):
            assert planned['name'] == actual['name']
            assert np.allclose(planned['center_m'], actual['render_bounds_center_m'], atol=.002)
            assert np.allclose(planned['size_m'], 2*np.asarray(actual['render_bounds_extent_m']), atol=.002)
        rp = cap/'observations'/row['rgb_path']
        dp = cap/'evaluator'/geo['native_path']
        assert sha(rp) == row['rgb_sha256'] == geo['rgb_sha256'] and sha(dp) == geo['native_sha256']
        image = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        assert image.shape == (360, 640, 3)
        rgb[i] = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        depth = np.load(dp, allow_pickle=False)
        values, _ = simulate(sample_native(depth), 'local-transfer/'+case['sensor_noise_key'], boxes)
        tof[i] = observation_tokens(values, boxes)
        baseline = decide(score_frame(boxes, values), .4071309640537889)
        bounds = camera_bounds(geo['objects'], case['camera'])
        classes, distances = geometric_labels(bounds)
        visible = visible_labels(depth, bounds)
        admissible = np.asarray(visible['unexplained']) == 0
        split = case['split']
        buffers[split].append(i)
        for key, value in dict(classes=classes, distances=distances, mask=visible['mask'],
                               coverage=visible['coverage'], valid=admissible).items():
            labels[split][key].append(value)
        identities.append(dict(index=i, id=case['name'], split=split, clip_id=case['clip_id'],
            frame_in_clip=case['frame_in_clip'], time_s=case['time_s'],
            base_group_id=case['base_group_id'], type_id=case['type_id'], layer=case['layer'],
            appearance=case['appearance'],appearance_pair_id=case['appearance_pair_id'],
            geometry_pair_id=case['geometry_pair_id'],
            layout_relation=case['layout_relation'], rgb_sha256=row['rgb_sha256'], baseline=baseline))
        audits.append(dict(index=i, source_native_sha256=geo['native_sha256'],
            visible_pixels=visible['count'], unexplained_pixels=visible['unexplained'],
            visible_nearest=visible['visible_nearest'], query_label_valid=admissible.tolist()))
        if i % 144 == 0:
            print('MATERIALIZE', i, '/', n, flush=True)
    rgb.flush(); tof.flush()
    del rgb, tof
    for split in labels:
        arrays = {k: np.asarray(v, np.float32 if k in ('mask', 'coverage', 'distances') else np.int64 if k == 'classes' else bool)
                  for k, v in labels[split].items()}
        np.savez_compressed(out/f'labels/{split}.npz', indices=np.asarray(buffers[split]), **arrays)
    write(out/'observations/identities.json', identities)
    write(out/'labels/visibility-audit.json', audits)
    pair_audit = audit_pairs(out)
    write(out/'labels/pair-audit.json', pair_audit)
    manifest = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    report = dict(status='PASS', frames=n, pair_audit=pair_audit['summary'], counts={k:len(v) for k,v in buffers.items()},
                  queries=QUERIES.tolist(), bin_edges_m=EDGES.tolist(),
                  source_spec_sha256=sha(root/'plan/spec.json'), hashes=manifest,
                  invalid_queries=sum(sum(not v for v in a['query_label_valid']) for a in audits),
                  elapsed_s=time.perf_counter()-start, backend='TASK_NOT_GPU_SUITABLE')
    write(out/'materialization.json', report)
    return report
