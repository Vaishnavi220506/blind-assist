"""Fixed corrected-box Radar readout; preserves raw support and complete old carry.

Observable-only consumed Development arm. Original cached boxes still own the
unchanged optical-flow tracks. Corrected boxes affect only current plane/range
geometry and its additive resolution guard. No labels or native provenance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import mz111_spatial_evidence as radar
import mz113_flow_persistence as flow
import mz115_spatial_allocation as zonal
from mz116_radar_resolution_guard import protect


ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def nominal_from_cache(rows, cached):
    """MZ115 nominal has empty legacy ToF, so its alert is raw Radar exactly."""
    return [dict(proposals=copy.deepcopy(p['proposals']),
        integrated_yaw_deg=p['integrated_yaw_deg'], tof_support=False,
        candidate=zonal.raw_radar(r, p['integrated_yaw_deg'], 3.6),
        baseline=zonal.raw_radar(r, p['integrated_yaw_deg'], 3.6))
        for r, p in zip(rows, cached)]


def predict(rows, baseline, corrected, image_loader):
    """Return {'predictions': [...], 'audit': {...}} without reading evaluator data.

    Each prediction supplies common_radar, guard_events and candidate for the
    Radar-only branch. Integrators OR these with their independently frozen ToF
    readout. Flow records include overlapping/hidden carry, not an OR residual.
    """
    assert len(rows) == len(baseline) == len(corrected)
    for r, old, new in zip(rows, baseline, corrected):
        for p in (old, new):
            if 'id' in p:
                assert p['id'] == r['id']
        assert old['integrated_yaw_deg'] == new['integrated_yaw_deg']
        assert old['common_radar'] == new['common_radar']
        assert old['guard_events'] == new['guard_events']
    empty = [zonal.legacy_empty_tof(r) for r in rows]
    old_nominal = nominal_from_cache(rows, baseline)
    new_nominal = nominal_from_cache(rows, corrected)
    old_current = radar.predict(empty, old_nominal, surface='plane', filter_range=True)
    new_current = radar.predict(empty, new_nominal, surface='plane', filter_range=True)
    old_flow = flow.predict(empty, old_nominal, image_loader)
    predictions = []
    for index, (r, old, new, nom, cur_old, cur_new, past) in enumerate(zip(
            rows, baseline, corrected, old_nominal, old_current, new_current, old_flow)):
        raw = nom['candidate']; yaw = nom['integrated_yaw_deg']
        carried = []
        for event in past['diagnostics']['propagated']:
            item = copy.deepcopy(event)
            xyz = zonal.radar_xyz(nom['proposals'][event['proposal']],
                                  event['range_m'], r, yaw, False)
            item['mz115_shell_possible'] = zonal.possible(xyz)
            item['xyz'] = xyz
            measured = rows[event['measurement_index']]
            assert measured['episode_id'] == r['episode_id']
            assert 0 < event['age_s'] <= flow.MAX_AGE_S
            assert event['age_s'] == r['time_s']-measured['time_s']
            item['measurement_id'] = measured['id']
            carried.append(item)
        carry = any(p['mz115_shell_possible'] for p in carried)
        # This verifies the exact saved union, including hidden propagated
        # records recovered by the original flow function, rather than guessing
        # carry from saved_common AND NOT current.
        original_common = bool(cur_old['candidate'] or (carry and not raw))
        assert original_common == old['common_radar'], r['id']
        original_guard = protect(r, {'candidate': False}, cur_old, yaw)['guard_events']
        assert original_guard == old['guard_events'], r['id']
        corrected_guard = protect(r, {'candidate': False}, cur_new, yaw)['guard_events']
        guard_events = [dict(copy.deepcopy(e), proposal_namespace='original')
                        for e in original_guard]
        guard_events += [dict(copy.deepcopy(e), proposal_namespace='corrected')
                         for e in corrected_guard]
        common = bool(raw or cur_new['candidate'] or carry)
        flag = bool(common or guard_events)
        assert not raw or flag
        assert not carry or flag
        assert not original_guard or flag
        predictions.append(dict(id=r['id'], candidate=flag,
            candidate_state='ALERT' if flag else 'UNKNOWN', common_radar=common,
            guard_events=guard_events, integrated_yaw_deg=yaw,
            original_common_radar=original_common,
            original_guard_support=bool(original_guard),
            original_current_radar=cur_old['candidate'],
            corrected_current_radar=cur_new['candidate'], raw_center_support=raw,
            inherited_carry_support=carry, inherited_carry_events=carried,
            carry_overlapped_original_current=bool(carry and cur_old['candidate']),
            original_current_evidence=cur_old['spatial_evidence'],
            corrected_current_evidence=cur_new['spatial_evidence']))
    audit = dict(frames=len(rows), original_common_reproduced=len(rows),
        raw_support_frames=sum(p['raw_center_support'] for p in predictions),
        complete_carry_support_frames=sum(p['inherited_carry_support'] for p in predictions),
        hidden_carry_frames=sum(p['carry_overlapped_original_current'] for p in predictions),
        propagated_records=sum(len(p['inherited_carry_events']) for p in predictions),
        explicit_raw_additions_to_original_common=[p['id'] for p in predictions
            if p['raw_center_support'] and not p['original_common_radar']],
        explicit_raw_additions_to_original_common_or_guard=[p['id'] for p in predictions
            if p['raw_center_support'] and not (p['original_common_radar'] or p['original_guard_support'])],
        policy='OR_CORRECTED_CURRENT_PLANE_RAW_CENTER_COMPLETE_ORIGINAL_CARRY_OLD_AND_NEW_GUARDS',
        max_carry_age_s=flow.MAX_AGE_S,
        authority='OBSERVABLE_ONLY_UNSCORED_BRANCH_PREDICTIONS',
        limits='Saved-common equality validates the observable union, not bit-identical historical OpenCV internal flow. All recomputed original propagated events are retained, including hidden overlaps; original boxes and frozen flow parameters are unchanged.')
    return dict(predictions=predictions, audit=audit)


def run(source, correction, output, image_root=None):
    source, correction, output = source.resolve(), correction.resolve(), output.resolve()
    image_root = (image_root or correction.parent/'rgb-v1').resolve()
    assert not output.exists()
    assert output.resolve().is_relative_to((ROOT/'artifacts.local').resolve())
    raw_path = source/'capture-v1/raw.jsonl'
    old_path = source/'analysis-v1/baseline-predictions.json'
    new_path = correction/'predictions.json'
    correction_seal_path = correction/'prediction-seal.json'
    seal = read(correction_seal_path)
    assert sha(raw_path) == seal['raw_sha256']
    assert sha(old_path) == seal['baseline_sha256']
    assert sha(new_path) == seal['predictions_sha256']
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert [r['id'] for r in rows] == list(seal['rgb_sha256'])
    inputs = [raw_path, old_path, new_path, correction_seal_path]
    for r in rows:
        image_path = image_root/r['rgb_path']
        assert sha(image_path) == seal['rgb_sha256'][r['id']]
        inputs.append(image_path)
    input_hashes = {str(p):sha(p) for p in inputs}
    def image_loader(row):
        image = cv2.imread(str(image_root/row['rgb_path']))
        assert image is not None, row['id']
        return image
    started = time.perf_counter()
    result = predict(rows, read(old_path), read(new_path), image_loader)
    result['audit'].update(seconds=time.perf_counter()-started,
        python=sys.executable, numpy=np.__version__, opencv=cv2.__version__,
        backend='CPU: GPU_BACKEND_UNAVAILABLE for frozen OpenCV LK flow and scalar geometry')
    output.mkdir(parents=True)
    write(output/'predictions.json', result['predictions'])
    write(output/'audit.json', result['audit'])
    sources = {str(Path(m.__file__).relative_to(ROOT)):sha(m.__file__)
        for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).suffix == '.py' and Path(m.__file__).is_relative_to(ROOT)
        and 'nearfield' in Path(m.__file__).parts}
    write(output/'prediction-seal.json', dict(input_hashes=input_hashes,
        raw_sha256=sha(raw_path), baseline_sha256=sha(old_path),
        correction_sha256=sha(new_path),
        source_hashes=sources, predictions_sha256=sha(output/'predictions.json'),
        audit_sha256=sha(output/'audit.json'), authority='SAVED_BEFORE_ANY_THIS_RUN_LABEL_PARSE'))
    assert input_hashes == {str(p):sha(p) for p in inputs}
    write(output/'completion.json', dict(status='PASS', inputs_unchanged=True,
        resources='No persistent worker, GPU allocation or process',
        prediction_seal_sha256=sha(output/'prediction-seal.json')))
    print(json.dumps(result['audit'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'correction', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    args = parser.parse_args()
    run(args.source, args.correction, args.output, args.image_root)
