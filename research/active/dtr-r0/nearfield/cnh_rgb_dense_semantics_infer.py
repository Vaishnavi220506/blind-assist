"""Frozen full-RGB SegFormer B0 inference; no labels, depth or query input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-rgb-dense-semantics-20261002'
REVISION='489d5cd81a0b59fab9b7ea758d3548ebe99677da'
MODEL=ROOT/'artifacts.local/models/nvidia--segformer-b0-finetuned-ade-512-512/snapshots'/REVISION
MODEL_SHA='6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def checkpoint_key(key):
    # Exact naming-only migration in installed Transformers5 Segformer conversion_mapping.
    for pattern,replacement in ((r'encoder\.patch_embeddings\.(\d+)\.',r'stages.\1.patch_embeddings.'),
        (r'encoder\.block\.(\d+)\.',r'stages.\1.blocks.'),(r'encoder\.layer_norm\.(\d+)',r'stages.\1.layer_norm')):
        key=re.sub(pattern,replacement,key)
    for old,new in (('attention.self.query','attention.q_proj'),('attention.self.key','attention.k_proj'),
        ('attention.self.value','attention.v_proj'),('attention.self.sr','attention.sequence_reduction.sequence_reduction'),
        ('attention.self.layer_norm','attention.sequence_reduction.layer_norm'),('attention.output.dense','attention.o_proj'),
        ('mlp.dense1','mlp.fc1'),('mlp.dense2','mlp.fc2'),('layer_norm_1','layernorm_before'),
        ('layer_norm_2','layernorm_after'),('decode_head.linear_c','decode_head.linear_projections')):
        key=key.replace(old,new)
    return key

def load():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HOME']=str(ROOT/'artifacts.local/cache/huggingface-dense-semantics')
    import torch
    import transformers
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    from safetensors.torch import load_file
    assert sha(MODEL/'model.safetensors')==MODEL_SHA
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA required; do not silently change backend'
    processor=SegformerImageProcessor.from_pretrained(str(MODEL),local_files_only=True)
    model,info=SegformerForSemanticSegmentation.from_pretrained(str(MODEL),local_files_only=True,
        use_safetensors=True,output_loading_info=True)
    for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
        assert not info.get(key), (key,info.get(key))
    checkpoint=load_file(str(MODEL/'model.safetensors'))
    state=model.state_dict()
    renamed={checkpoint_key(k):v for k,v in checkpoint.items()}
    assert len(renamed)==len(checkpoint) and set(state)==set(renamed),'Bijective checkpoint-to-runtime names required'
    assert all(torch.equal(state[k].cpu(),renamed[k]) for k in state)
    count=len(state);del checkpoint,state
    assert model.config.num_labels==150
    model=model.eval().to('cuda',dtype=torch.float32)
    meta=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),dtype='float32',tensor_count_verified=count,
        loading_info={k:list(v) if isinstance(v,set) else v for k,v in info.items()},
        processor=json.loads(json.dumps(processor.to_dict())),revision=REVISION,
        checkpoint_conversion='Naming-only installed Transformers5 migration; every tensor exact-equal',
        attention_implementation=model.config._attn_implementation,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in MODEL.iterdir() if p.is_file()},
        full_image=True,interpolation='bilinear native logits, align_corners=False, then argmax150',
        image_decode='Pillow RGB; no EXIF transpose or ICC conversion',batch_size=1)
    return torch,processor,model,meta

def predict(image,torch,processor,model):
    tensor=processor(images=image,return_tensors='pt')['pixel_values'].to('cuda',dtype=torch.float32)
    assert tuple(tensor.shape)==(1,3,512,512)
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        logits=model(pixel_values=tensor).logits
        assert tuple(logits.shape)==(1,150,128,128) and torch.isfinite(logits).all()
        native=torch.nn.functional.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False)
        labels=native.argmax(dim=1)[0].to(dtype=torch.uint8).cpu().numpy()
    torch.cuda.synchronize()
    return labels,time.perf_counter()-start

def preflight():
    from PIL import Image
    torch,processor,model,meta=load()
    synthetic=Image.new('RGB',(1024,768),color=(125,125,125))
    labels,elapsed=predict(synthetic,torch,processor,model)
    assert labels.shape==(768,1024) and labels.dtype==np.uint8 and labels.max()<150
    meta.update(synthetic_shape=list(labels.shape),synthetic_seconds=elapsed,
        synthetic_only=True,license_path=str(OUT/'MODEL_LICENSE.txt'))
    save(OUT/'model-receipt.json',meta)
    print(json.dumps(dict(status='PASS_CHECKPOINT_AND_SYNTHETIC_CUDA',tensors=meta['tensor_count_verified'],device=meta['device'])))

def infer():
    from PIL import Image
    pred=OUT/'predictions';pred.mkdir(exist_ok=True)
    assert not (pred/'inference-result.json').exists(),'Preserve completed inference'
    plan=read(OUT/'PLAN.json')
    own=str(Path(__file__).relative_to(ROOT))
    assert sha(Path(__file__))==plan['source_sha256'][own]
    observations=read(OUT/'observations.json')
    assert len(observations)==343 and len({r['id'] for r in observations})==343
    assert all(set(r)=={'id','rgb_path','rgb_sha256'} for r in observations)
    torch,processor,model,meta=load()
    expected=read(OUT/'model-receipt.json')
    assert meta['files']==expected['files'] and meta['processor']==expected['processor']
    outputs={};started=time.perf_counter()
    for index,row in enumerate(observations):
        dest=pred/(row['id']+'.npz');receipt=dest.with_suffix('.json')
        assert sha(row['rgb_path'])==row['rgb_sha256']
        if dest.exists() or receipt.exists():
            assert dest.exists() and receipt.exists(),'Partial frame output requires inspection'
            record=read(receipt)
            assert record['source_sha256']==sha(Path(__file__)) and record['rgb_sha256']==row['rgb_sha256']
            assert record['npz_sha256']==sha(dest) and record['model_sha256']==MODEL_SHA
        else:
            with Image.open(row['rgb_path']) as f:image=f.convert('RGB')
            assert image.size==(1024,768)
            labels,seconds=predict(image,torch,processor,model)
            with dest.open('xb') as f:np.savez_compressed(f,labels=labels)
            record=dict(id=row['id'],rgb_sha256=row['rgb_sha256'],npz_sha256=sha(dest),
                model_sha256=MODEL_SHA,source_sha256=sha(Path(__file__)),seconds=seconds,
                shape=list(labels.shape),class_histogram=np.bincount(labels.ravel(),minlength=150).tolist())
            save(receipt,record)
        outputs[row['id']]=dict(record,receipt_sha256=sha(receipt))
        if (index+1)%25==0 or index+1==len(observations):print(f'{index+1}/{len(observations)}',flush=True)
    save(pred/'inference-result.json',dict(status='COMPLETE',outputs=outputs,model=meta,
        elapsed_seconds=time.perf_counter()-started,predict_seconds=sum(r['seconds'] for r in outputs.values())))
    print('Completed343 frozen full-RGB predictions')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','infer'))
    globals()[parser.parse_args().action]()
