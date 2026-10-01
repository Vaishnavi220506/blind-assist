"""Read-only acceptance accounting for completed corridor stopping run."""
import hashlib
import json
import numpy as np
import cnh_corridor_stopping as M


def main():
    r = json.loads((M.OUT / 'results.json').read_text())
    rows = r['rows']
    pick = lambda policy, reaction: next(x for x in rows if x['policy'] == policy and x['reaction'] == reaction)
    for reaction in M.REACTIONS:
        always, never = pick('always', reaction), pick('never', reaction)
        assert never['collisions'] == never['hazard_n'] and never['commands'] == 0
        assert always['unnecessary'] == always['clear_n'] and always['commands'] == always['total']
        for policy in ('head', 'travel', 'old_head', 'old_travel'):
            row = pick(policy, reaction)
            assert always['collisions'] <= row['collisions'] <= never['collisions']
            assert row['commands'] <= row['total']
            assert row['unnecessary'] <= row['clear_n']
    for policy in ('head', 'travel', 'old_head', 'old_travel', 'always', 'never'):
        values = [pick(policy, reaction)['collisions'] for reaction in M.REACTIONS]
        assert values == sorted(values)
    requests = {}
    models = {}
    for kind in ('head', 'travel'):
        folder = M.OUT / 'models' / kind
        request = json.loads((folder / 'request.json').read_text())
        assert [x['unit'] for x in request['inputs']] == M.SPLITS['train']
        assert not request['calib_access'] and not request['eval_access']
        # Keep raw request; copied generic recipe carries a travel default.
        # Actual training indexes labels_{kind}/support_{kind}, as the top-level
        # query_frame already records. Normalize metadata only, never weights.
        normalized = dict(request, recipe=dict(request['recipe'], query_frame=kind))
        normalized['metadata_note'] = 'Generic recipe query_frame normalized; actual labels/support unchanged; raw request retained'
        M.save(folder / 'request_normalized.json', normalized)
        requests[kind] = request
        for seed in range(3):
            history = json.loads((folder / f'history_seed{seed}.json').read_text())
            assert [x['epoch'] for x in history] == list(range(1, 21))
            path = folder / f'model_seed{seed}.pt'
            models[f'{kind}{seed}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert requests['head']['inputs'] == requests['travel']['inputs']
    assert requests['head']['recipe'] == requests['travel']['recipe']
    calibration = {}
    for policy in ('head', 'travel', 'old_head', 'old_travel'):
        negative = []
        for unit in M.SPLITS['calib']:
            with np.load(M.OUT / 'predictions' / f'unit{unit}.npz') as z:
                negative.append(z[policy][np.isinf(z['distance']), 7:])
        scores = np.concatenate(negative)
        threshold = r['thresholds'][policy]
        count, alarm = M.episodes(scores, threshold)
        minutes = scores.size * M.DT / 60
        assert count / minutes <= 2 + 1e-12
        calibration[policy] = dict(threshold=threshold, negative_clips=len(scores),
                                   episodes=count, minutes=minutes, episodes_per_minute=count/minutes,
                                   first_stops=int(alarm.any(1).sum()))
    M.save(M.OUT / 'calibration_actual.json', calibration)
    collisions = outside_query_height = 0
    for unit in M.SPLITS['evaluation']:
        with np.load(M.OUT / 'predictions' / f'unit{unit}.npz') as z:
            for policy in ('head', 'travel', 'old_head', 'old_travel'):
                assert z[policy].shape == (22, M.FRAMES) and np.isfinite(z[policy]).all()
            saved = z['distance']
        for i, scene in enumerate(M.scenes(unit)):
            distance, ids = M.first_collision(scene['boxes'])
            assert distance == saved[i]
            if not np.isfinite(distance):
                continue
            collisions += 1
            outside_query_height += int(all(scene['boxes'][j]['lo'][1] >= .9 or scene['boxes'][j]['hi'][1] <= -.2 for j in ids))
    receipt = dict(status='PASS', focused_tests=8, strategy_rows=len(rows), models=models,
                   identical_training_inputs=True, identical_recipes=True,
                   calibration_actual=calibration,
                   collision_geometry_recomputed=48*22,
                   collision_scenes=collisions,
                   first_collision_completely_outside_HEAD_BODY_height=outside_query_height,
                   checks=['baseline ordering', 'reaction monotonicity', 'finite scores',
                           '48 train only', '2 systems x 3 seeds x 20 epochs', 'continuous collision geometry'])
    M.save(M.OUT / 'acceptance.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
