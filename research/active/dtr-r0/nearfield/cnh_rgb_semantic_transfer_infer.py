"""Path/count adapter for the unchanged RGB-only dense semantic recipe."""
from pathlib import Path
import time
import numpy as np
from PIL import Image
import cnh_rgb_dense_semantics_infer as I

OUT=I.ROOT/'artifacts.local/work/cnh-rgb-semantic-transfer-20261002'


def main():
    plan=I.read(OUT/'PLAN.json')
    for path in (Path(__file__),Path(I.__file__)):
        assert I.sha(path)==plan['source_sha256'][str(path.relative_to(I.ROOT))]
    observations=I.read(OUT/'observations.json')
    assert I.sha(OUT/'observations.json')==plan['observations_sha256']
    assert len(observations)==len({r['id'] for r in observations})==96
    assert all(set(r)=={'id','rgb_path','rgb_sha256'} for r in observations)
    pred=OUT/'predictions';pred.mkdir(exist_ok=True)
    assert not (pred/'inference-result.json').exists()
    torch,processor,model,meta=I.load()
    expected=I.read(I.OUT/'model-receipt.json')
    assert meta['files']==expected['files'] and meta['processor']==expected['processor']
    outputs={};started=time.perf_counter()
    for index,row in enumerate(observations):
        dest=pred/(row['id']+'.npz');receipt=dest.with_suffix('.json')
        assert I.sha(row['rgb_path'])==row['rgb_sha256']
        if dest.exists() or receipt.exists():
            assert dest.exists() and receipt.exists(),'Inspect partial outputs before recovery'
            record=I.read(receipt)
            assert record['source_sha256']==I.sha(I.__file__) and record['adapter_sha256']==I.sha(__file__)
            assert record['rgb_sha256']==row['rgb_sha256'] and record['npz_sha256']==I.sha(dest)
            assert record['model_sha256']==I.MODEL_SHA
        else:
            with Image.open(row['rgb_path']) as f:image=f.convert('RGB')
            assert image.size==(1024,768)
            labels,seconds=I.predict(image,torch,processor,model)
            with dest.open('xb') as f:np.savez_compressed(f,labels=labels)
            record=dict(id=row['id'],rgb_sha256=row['rgb_sha256'],npz_sha256=I.sha(dest),model_sha256=I.MODEL_SHA,
                source_sha256=I.sha(I.__file__),adapter_sha256=I.sha(__file__),seconds=seconds,
                shape=list(labels.shape),class_histogram=np.bincount(labels.ravel(),minlength=150).tolist())
            I.save(receipt,record)
        outputs[row['id']]=dict(record,receipt_sha256=I.sha(receipt))
        if (index+1)%24==0:print(f'{index+1}/96',flush=True)
    I.save(pred/'inference-result.json',dict(status='COMPLETE',outputs=outputs,model=meta,
        elapsed_seconds=time.perf_counter()-started,predict_seconds=sum(r['seconds'] for r in outputs.values())))


if __name__=='__main__':main()
