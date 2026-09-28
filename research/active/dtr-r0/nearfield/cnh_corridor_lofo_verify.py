"""Independent reduction/role acceptance of saved LOFO evidence; no model fit."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-corridor-lofo-20260929'


def run():
    r=json.loads((OUT/'results.json').read_text())
    request=json.loads((OUT/'request.json').read_text())
    ledger=[json.loads(x) for x in (OUT/'sample_ledger.jsonl').read_text().splitlines()]
    assert len(ledger)==2112
    assert len({(x['unit'],x['config'],x['group']) for x in ledger})==2112
    for path,expected in {**request['inputs'],**request['sources']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,('identity',path)
    histories=0
    for family in request['families']:
        for seed in range(3):
            h=json.loads((OUT/family/f'seed{seed}_history.json').read_text())
            receipt=json.loads((OUT/family/f'seed{seed}_receipt.json').read_text())
            assert len(h)==20 and [x['epoch'] for x in h]==list(range(1,21))
            assert len({x['samples'] for x in h})==1
            assert hashlib.sha256((OUT/family/f'model_seed{seed}.pt').read_bytes()).hexdigest()==receipt['sha256']
            assert receipt['peak_cuda_reserved_bytes']<=1.2*1024**3
            histories+=1
        for u in request['splits']['calib']:
            with np.load(OUT/family/'predictions'/f'unit{u}.npz') as z:
                assert not (z['family']==family).any()
                assert len(z['family'])==(18 if family=='general' else 16)
        for u in request['splits']['evaluation']:
            with np.load(OUT/family/'predictions'/f'unit{u}.npz') as z:
                assert (z['family']==family).all()
                assert len(z['family'])==(4 if family=='general' else 6)
    def subset(family,g,policy):
        return [x for x in ledger if x['group']==g and (family=='pooled_crossfold' or x['family']==family)
            and (policy=='strict' or not(x['target_group']==['HEAD','BODY'].index(g) and abs(x['margin'])<=.05))]
    for row in r['rows']:
        xs=subset(row['family'],row['group'],row['policy'])
        y=np.array([x['label'] for x in xs],bool);pred=np.array([x['predictions'][row['arm']] for x in xs],bool)
        assert row['positive']==int(y.sum()) and row['negative']==int((~y).sum())
        assert row['tp']==int((y&pred).sum()) and row['fp']==int((~y&pred).sum())
        assert row['fn']==int((y&~pred).sum()) and row['tn']==int((~y&~pred).sum())
    for c in r['comparisons']:
        xs=subset(c['family'],c['group'],c['policy']);ids=sorted({x['unit'] for x in xs});assert len(ids)==48
        cnt=np.array([[sum(x['label']==v for x in xs if x['unit']==u) for v in [1,0]]+
            [sum(x['label']==v and x['predictions'][a]!=v for x in xs if x['unit']==u) for a in [c['baseline'],'LOFO'] for v in [1,0]] for u in ids])
        pooled=cnt.sum(0);delta=.5*((pooled[4]-pooled[2])/pooled[0]+(pooled[5]-pooled[3])/pooled[1])
        draw=np.random.default_rng(20260929).integers(0,48,(5000,48));boot=cnt[draw].sum(1)
        valid=(boot[:,0]>0)&(boot[:,1]>0);b=boot[valid]
        differences=.5*((b[:,4]-b[:,2])/b[:,0]+(b[:,5]-b[:,3])/b[:,1])
        ci=np.quantile(differences,[.025,.975])
        assert abs(c['bootstrap']['point']-delta)<1e-12
        np.testing.assert_allclose(c['bootstrap']['ci'],ci,atol=1e-12)
        gate=bool(delta<=-.05 and ci[1]<0 and (1-pooled[4]/pooled[0])>=(1-pooled[2]/pooled[0])-.02)
        assert gate==c['numerical_gate']
        repaired=sum(x['predictions'][c['baseline']]!=x['label'] and x['predictions']['LOFO']==x['label'] for x in xs)
        added=sum(x['predictions'][c['baseline']]==x['label'] and x['predictions']['LOFO']!=x['label'] for x in xs)
        assert c['repaired']==repaired and c['new_errors']==added
    result=dict(status='PASS',ledger_rows=len(ledger),metric_rows=len(r['rows']),comparisons=len(r['comparisons']),
        seed_histories=histories,epochs_per_seed=20,identities_verified=len(request['inputs'])+len(request['sources']),
        checks=['calibration excludes heldout family','evaluation contains only heldout family','same unit retained in bootstrap including one-class clusters',
            'all denominators and confusion cells reconstructed','independent pooled cluster bootstrap and numerical gate recomputed','paired repairs and new errors reconstructed','source/input/model hashes and training budget'])
    (OUT/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':run()
