"""Read-only saved-evidence entry audit; no temporal classifier or oracle fit."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[5]
    artifacts = (repo / 'artifacts.local').resolve()
    work = artifacts / 'work'
    paths = {
        'cases': work / 'corridor-representation-20260918/cases.json',
        'predictions': work / 'corridor-representation-20260918/predictions.json',
        'models': work / 'corridor-representation-20260918/model-seal.json',
        'raw': work / 'corridor-public-single-20260917/source/returned-v1/capture-v1/raw.jsonl',
    }
    cases = json.loads(paths['cases'].read_text())
    predictions = json.loads(paths['predictions'].read_text())
    models = json.loads(paths['models'].read_text())['models']
    raw = [json.loads(line) for line in paths['raw'].read_text().splitlines() if line.strip()]
    ids = [r['id'] for r in cases]
    assert len(ids) == len(set(ids)) == 288
    assert ids == [r['id'] for r in predictions] == [r['id'] for r in raw]
    episodes = {}
    for c, p, r in zip(cases, predictions, raw):
        assert c['ablation'] == p
        assert c['episode_id'] == r['episode_id'] and c['time_s'] == r['time_s']
        for arm in ('raw', 'multi'):
            assert p['flags'][arm] == (p['scores'][arm] >= models[arm]['threshold'])
        history = episodes.setdefault(r['episode_id'], [])
        assert not history or history[-1][0]['time_s'] < c['time_s']
        history.append((c, r))
    def counts(arm):
        rows = [c for c in cases if c['stratum'] != 'boundary']
        return dict(TP=sum(c['truth'] and c['ablation']['flags'][arm] for c in rows),
                    FP=sum(not c['truth'] and c['ablation']['flags'][arm] for c in rows),
                    FN=sum(c['truth'] and not c['ablation']['flags'][arm] for c in rows))
    frontier = []
    for c in cases:
        if c['stratum'] == 'boundary' or not c['ablation']['flags']['raw'] or c['ablation']['flags']['multi']:
            continue
        history = [dict(id=p['id'], time_s=p['time_s'],
                        saved_native_corridor_witness=p['sampled_witness'],
                        usable_tof_returns=p['usable_tof_returns'],
                        radar_range_m=r['radar_range_m'], radar_angle=r['radar_angle'],
                        radar_velocity=r['radar_velocity'], radar_valid=r['radar_valid'],
                        imu_valid=r['imu_valid'], delta_yaw=r['delta_yaw'])
                   for p, r in episodes[c['episode_id']]
                   if 0 <= c['time_s'] - p['time_s'] <= 1.0]
        frontier.append(dict(id=c['id'], truth=c['truth'], history=history,
                             history_has_native_tof_corridor_witness=any(h['saved_native_corridor_witness'] for h in history)))
    removed = [c['id'] for c in cases if c['stratum'] != 'boundary' and not c['truth']
               and c['ablation']['flags']['multi'] and not c['ablation']['flags']['raw']]
    result = dict(status='ENTRY_AUDIT_PASS_NOT_A_TEMPORAL_CEILING_RESULT',
                  backend='CPU_TASK_NOT_GPU_SUITABLE_JSON_ACCOUNTING',
                  counts={a:counts(a) for a in ('multi', 'raw')}, frontier=frontier,
                  raw_removed_astar_fp=removed,
                  hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                  limits=['Consumed Development saved-label audit; not a fresh native reconstruction.',
                          'No association, transport, temporal separability or free-space proof.',
                          'No model, threshold, source, or App change.'])
    assert result['counts'] == {'multi':dict(TP=96,FP=3,FN=12),'raw':dict(TP=99,FP=5,FN=9)}
    assert sum(f['truth'] for f in frontier) == 3 and len(frontier) == 6 and len(removed) == 1
    out = args.output.resolve()
    if not out.is_relative_to(artifacts) or out == artifacts or out.exists():
        raise ValueError('Output must be a fresh artifact file')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','counts')}))


if __name__ == '__main__':
    main()
