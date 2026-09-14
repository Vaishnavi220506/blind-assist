"""Authenticate the frozen MZ129 comparator and inventory contour source inputs.

Engineering prerequisite only: no oracle prediction, training or outcome tuning.
"""
import argparse
from collections import Counter
from pathlib import Path
import json

from mz129_extent_correction import ROOT, SOURCE, CORRECTION, events
from mz126_central_tof import read, write, sha, serial, tof
from mz128_zone_weighting import column_weights, FOUR, THRESHOLD, readout
from mz124_measurement_geometry import metrics

MZ129 = ROOT/'artifacts.local/work/mz129-extent-correction-20260914'


def run(output):
    assert not output.exists()
    assert output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw = SOURCE/'capture-v1/raw.jsonl'
    manifest_path = SOURCE/'capture-v1/manifest.json'
    receipt_path = SOURCE/'capture-v1/receipt.json'
    cp = CORRECTION/'predictions.json'
    bp = MZ129/'replay-v1r2/predictions.json'
    rp = MZ129/'radar-v1/predictions.json'
    inputs = [raw, manifest_path, receipt_path, cp, bp, rp,
              SOURCE/'capture-v1/spec.json', SOURCE/'analysis-v1/frame-report.json']
    for directory in (MZ129/'replay-v1r2', MZ129/'radar-v1'):
        seal_path = directory/'prediction-seal.json'
        inputs.append(seal_path)
        seal = read(seal_path)
        assert sha(directory/'predictions.json') == seal['predictions_sha256']
        for key in ('input_hashes', 'dependencies', 'source_hashes'):
            for path, digest in seal.get(key, {}).items():
                assert sha(ROOT/path) == digest, path
    cs_path = CORRECTION/'prediction-seal.json'
    inputs.append(cs_path)
    cs = read(cs_path)
    assert sha(raw) == cs['raw_sha256'] and sha(cp) == cs['predictions_sha256']
    rows = [json.loads(s) for s in raw.read_text().splitlines()]
    cache, baseline, radar = read(cp), read(bp)['radar'], read(rp)
    manifest, receipt = read(manifest_path), read(receipt_path)
    assert receipt['status'] == 'PASS'
    assert sha(SOURCE/'capture-v1/spec.json') == receipt['spec_sha256']
    assert len(rows) == len(cache) == len(baseline) == len(radar) == 288
    assert [r['id'] for r in rows] == [m['id'] for m in manifest['frames']]
    assert [r['id'] for r in rows] == [p['id'] for p in baseline] == [p['id'] for p in radar]
    replayed = []
    for row, cached, old, rb in zip(rows, cache, baseline, radar):
        image = ROOT/'artifacts.local/work/mz125-observable-correction-20260913/rgb-v1'/row['rgb_path']
        assert sha(image) == receipt['hashes'][row['rgb_path']] == cs['rgb_sha256'][row['id']]
        inputs.append(image)
        assert serial(tof.allocate(row, cached)) == cached['spatial_evidence']
        value = readout(cached, column_weights(row, FOUR))
        flag = bool(value['score'] >= THRESHOLD or value['certain_coarse'] or rb['candidate'])
        assert value['score'] == old['score'] and value['certain_coarse'] == old['certain_coarse']
        assert flag == old['candidate']
        replayed.append(dict(id=row['id'], candidate=flag))
    output.mkdir(parents=True)
    write(output/'baseline-reproduced.json', replayed)
    report = read(SOURCE/'analysis-v1/frame-report.json')
    assert [r['id'] for r in report] == [r['id'] for r in rows]
    truth = [r['truth'] for r in report]
    flags = [r['candidate'] for r in replayed]
    result = dict(status='BASELINE_REPRODUCED_SOURCE_CONTOURS_ABSENT',
        authority='ENGINEERING_READINESS_NOT_ORACLE_METHOD_RESULT',
        baseline=metrics(rows, truth, flags), events=events(rows, truth, flags),
        manifest_path_keys=dict(Counter(k for m in manifest['frames'] for k in m['paths'])),
        capture_depth_images_produced=manifest['depth_images_produced'],
        verified_rgb_frames=len(rows), families={},
        associator_interface='MZ115 assign_groups and MZ111 current_returns accept boxes only',
        candidate_metrics=None, training_authorized_by_result=False,
        source_requirement='UE visible instance pass, complete scene including background/floor; anonymous 2D export only',
        backend='CPU: TASK_NOT_GPU_SUITABLE, scalar comparator and integrity audit')
    for family in sorted({r['family'] for r in report}):
        ix = [i for i, r in enumerate(report) if r['family'] == family]
        result['families'][family] = metrics([rows[i] for i in ix], [truth[i] for i in ix], [flags[i] for i in ix])
    assert (result['baseline']['TP'], result['baseline']['FP'], result['baseline']['FN']) == (139, 93, 5)
    write(output/'readiness.json', result)
    inputs.extend([Path(__file__), Path(tof.__file__)])
    write(output/'seal.json', dict(input_hashes={str(p):sha(p) for p in inputs},
        baseline_sha256=sha(output/'baseline-reproduced.json'), readiness_sha256=sha(output/'readiness.json')))
    write(output/'completion.json', dict(status='PASS', oracle_run=False,
        resources='No persistent processes or allocation', seal_sha256=sha(output/'seal.json')))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
