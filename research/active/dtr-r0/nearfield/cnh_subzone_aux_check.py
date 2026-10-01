"""Read-only acceptance checks for the consumed auxiliary-supervision run."""
import hashlib
import json
import numpy as np
import torch
import cnh_subzone_aux as A


def check():
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    request = read(A.OUT/'models/request.json')
    baseline = read(A.BASE/'models/training_request.json')
    assert request['features'] == baseline['inputs']
    assert request['recipe'] == baseline['recipe']
    assert request['source_sha256'] == digest(A.HERE/'cnh_subzone_aux.py')
    for name, sha in read(A.OUT/'target_request.json')['identity'].items():
        assert digest(A.OUT/'source'/name) == sha
    max_sum_error = 0.
    for feature, target in zip(request['features'], request['targets'], strict=True):
        unit = feature['unit']
        fp = A.BASE/'features/train'/f'unit{unit}.npz'
        tp = A.OUT/'targets'/f'unit{unit}.npz'
        assert digest(fp) == feature['sha256']
        assert digest(tp) == target['sha256']
        with np.load(fp) as f, np.load(tp) as t:
            assert np.array_equal(f['scene'], t['scene'])
            assert np.array_equal(f['frame'], t['frame'])
            assert t['target'].shape == (352, 4, 8, 8, 17)
            assert np.isfinite(t['target']).all() and (t['target'] >= 0).all()
            err = float(abs(t['target'].astype(np.float32).sum(-1)-1).max())
            max_sum_error = max(max_sum_error, err)
            assert err < .001  # float16 storage; training renormalizes in float32.
    for model in read(A.OUT/'models/receipt.json')['models']:
        seed = model['seed']
        p = A.OUT/'models'/f'model_seed{seed}.pt'
        assert digest(p) == model['sha256']
        export = torch.load(p, map_location='cpu', weights_only=True)
        full = torch.load(A.OUT/'models'/f'train_only_seed{seed}.pt', map_location='cpu', weights_only=True)
        plain = A.L.Readout()
        plain.load_state_dict(export, strict=True)
        for k, v in export.items():
            torch.testing.assert_close(v, full['readout.'+k], rtol=0, atol=0)
        assert len(read(A.OUT/'models'/f'history_seed{seed}.json')) == 20
        assert model['deployed_parameters'] == 44833
    result = read(A.OUT/'results.json')
    ledger = [json.loads(s) for s in (A.OUT/'sample_ledger.jsonl').read_text().splitlines()]
    assert len(ledger) == 2112
    assert len({(r['unit'], r['config'], r['group']) for r in ledger}) == 2112
    for row in result['rows']:
        qi = ['HEAD', 'BODY'].index(row['group'])
        subset = row['subset']
        selected = []
        for sample in ledger:
            if sample['group'] != row['group']:
                continue
            own = sample['target_group'] == qi
            near = abs(sample['margin']) <= .05
            if row['policy'] == 'ignore5cm' and own and near:
                continue
            if subset in ('general', 'mixed_surface') and sample['family'] != subset:
                continue
            if subset == 'inside_0_5cm' and not (own and near and sample['margin'] < 0):
                continue
            if subset == 'outside_0_5cm' and not (own and near and sample['margin'] > 0):
                continue
            selected.append(sample)
        y = np.array([s['label'] for s in selected], bool)
        pred = np.array([s[row['arm'].lower()] for s in selected], bool)
        for k, v in A.D.old.binary_metrics(y, pred).items():
            assert row[k] == v or (v is not None and abs(row[k]-v) < 1e-12), (row, k, v)
    for qi, group in enumerate(['HEAD', 'BODY']):
        scores, labels = [], []
        for unit in A.D.SPLITS['calib']:
            with np.load(A.OUT/'predictions'/f'unit{unit}.npz') as z:
                scores.extend(z['scores'][:, qi])
            with np.load(A.BASE/'predictions'/f'unit{unit}.npz') as z:
                labels.extend(z['labels'][:, qi])
        score = np.array(scores); label = np.array(labels, bool)
        assert A.D.old.threshold(score[~label]) == result['thresholds'][group]
        selected = [s for s in ledger if s['group'] == group]
        boot = A.paired_cluster_ber(
            np.array([s['label'] for s in selected], bool),
            np.array([s['base'] for s in selected], bool),
            np.array([s['aux'] for s in selected], bool),
            np.array([s['unit'] for s in selected]))
        assert boot == next(c['bootstrap'] for c in result['comparisons']
                            if c['group'] == group and c['policy'] == 'strict' and c['subset'] == 'all')
    receipt = dict(status='PASS', train_units=96, exported_models=3, epochs_per_seed=20,
                   ledger_rows=2112, recounted_metric_rows=len(result['rows']),
                   maximum_float16_target_sum_error=max_sum_error,
                   result_sha256=digest(A.OUT/'results.json'))
    (A.OUT/'acceptance.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    check()
