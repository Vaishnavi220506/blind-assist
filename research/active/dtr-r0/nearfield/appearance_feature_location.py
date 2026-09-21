"""Saved RGB feature mediation; no image editing, fitting or source recapture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def transplant(reference, changed, region):
    if reference.shape != changed.shape or reference.ndim != 4 or reference.shape[1:] != (220,8,8):
        raise ValueError('Need matching public [N,220,8,8] inputs')
    if region not in ('center','outer'):
        raise ValueError('Unknown partition')
    result = reference.copy()
    if region == 'center':
        result[:,:216,:,2:6] = changed[:,:216,:,2:6]
    else:
        result[:,:216,:,:2] = changed[:,:216,:,:2]
        result[:,:216,:,6:] = changed[:,:216,:,6:]
    return result


def allocate(z00, z11, z10, z01):
    return dict(center=.5*((z10-z00)+(z11-z01)),
                outer=.5*((z01-z00)+(z11-z10)),
                interaction=z11-z10-z01+z00)


def run(source, source_code, out, protocol):
    assert not out.exists()
    started = time.perf_counter()
    sys.path.insert(0,str(source_code))
    import torch
    import spatial_bce_model as model
    out.mkdir(parents=True)
    old_protocol = read(source/'protocol.json')
    for rel,digest in old_protocol['code_hashes'].items():
        assert sha(source_code.parents[3]/rel) == digest, rel
    for name in ('observation','prediction','evaluation'):
        for rel, digest in read(source/(name+'-seal.json'))['hashes'].items():
            assert sha(source/rel) == digest, rel
    feature_receipt = read(source/'features/feature_receipt.json')
    assert sha(source/'features/rgb_features.npy') == feature_receipt['cache_sha256']
    for name in ('B','N'):
        assert sha(source/'inputs'/(name+'.pt')) == old_protocol['input_hashes'][name+'.pt']
    assert torch.__version__ == '2.9.1+cu128' and torch.cuda.is_available()
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3060 Laptop GPU'
    for name in ('B','N'):
        backend = read(source/(name+'-backend.json'))
        assert backend['selected_device_type'] == 'cuda'
    write(out/'protocol.json',dict(protocol_text=protocol.read_text(encoding='utf-8'),
        protocol_sha256=sha(protocol),code_sha256=sha(__file__),source=str(source),
        source_protocol_sha256=sha(source/'protocol.json'),feature_sha256=feature_receipt['cache_sha256'],
        model_sha256={n:sha(source/'inputs'/(n+'.pt')) for n in ('B','N')},
        backend='cuda',device=torch.cuda.get_device_name(),framework=torch.__version__,
        backend_basis='REUSED_SAME_HEAD_BATCH64_PRIOR_MEASURED_GPU_SELECTION'))
    ids = read(source/'identities.json')
    original = read(source/'predictions.json')
    # Pair identities are public IDs, not geometry/truth fields.
    versions = {}
    for i,row in enumerate(ids):
        key,condition = row['id'].rsplit('__',1)
        versions.setdefault(key,{})[condition] = i
    assert len(versions) == 288 and all(set(v)=={'reference','repeat','target','background'} for v in versions.values())
    inputs = model.build_inputs(np.load(source/'features/rgb_features.npy'),np.load(source/'ranges.npy'))
    model._precision()
    result=[]; replay={}
    for name,cutoff in old_protocol['cuts'].items():
        head=model.load_head(source/'inputs'/(name+'.pt'),'cuda')
        all_logits=model.predict(head,inputs).astype(float)
        old_logits=np.array([p['logits'][name] for p in original])
        error=float(np.max(np.abs(all_logits-old_logits)))
        assert error<=2e-5 and np.array_equal(all_logits>=cutoff,old_logits>=cutoff)
        replay[name]=dict(max_logit_error=error,decisions_identical=True)
        for condition in ('repeat','target','background'):
            keys=sorted(versions)
            ri=[versions[k]['reference'] for k in keys];ci=[versions[k][condition] for k in keys]
            reference,changed=inputs[ri],inputs[ci]
            assert np.array_equal(reference[:,216:],changed[:,216:])
            center=model.predict(head,transplant(reference,changed,'center')).astype(float)
            outer=model.predict(head,transplant(reference,changed,'outer')).astype(float)
            for j,key in enumerate(keys):
                r,c=ri[j],ci[j]
                z=dict(reference=float(all_logits[r]),changed=float(all_logits[c]),center=float(center[j]),outer=float(outer[j]))
                baseline=original[r]['flags']['A_current']
                attribution=allocate(z['reference'],z['changed'],z['center'],z['outer'])
                assert abs(attribution['center']+attribution['outer']-(z['changed']-z['reference']))<1e-10
                result.append(dict(source_id=key,index=c,reference_index=r,condition=condition,model=name,
                    logits=z,allocation=attribution,flags={k:bool(baseline or v>=cutoff) for k,v in z.items()}))
    write(out/'predictions.json',result)
    write(out/'prediction-seal.json',dict(time_ns=time.time_ns(),predictions_sha256=sha(out/'predictions.json'),replay=replay))
    # Consumed source descriptors are joined only after public interventions seal.
    spec=read(source/'spec.json')['cases']
    admission=read(source/'pair-admission.json')['rows']
    summaries={}
    for name in ('B','N'):
        summaries[name]={}
        for condition in ('repeat','target','background'):
            rows=[r for r in result if r['model']==name and r['condition']==condition]
            def describe(items):
                return dict(n=len(items),
                    mean_absolute_allocation={k:float(np.mean([abs(r['allocation'][k]) for r in items])) for k in ('center','outer','interaction')},
                    signed_mean_allocation={k:float(np.mean([r['allocation'][k] for r in items])) for k in ('center','outer')},
                    current_flips={k:sum(r['flags'][k]!=r['flags']['reference'] for r in items) for k in ('changed','center','outer')},
                    failed_native_pairs=sum(not admission[r['index']]['depth_equal'] for r in items))
            summaries[name][condition]=dict(all=describe(rows),families={family:describe([r for r in rows if spec[r['index']]['type_id']==family])
                for family in sorted({c['type_id'] for c in spec})})
    write(out/'summary.json',dict(status='INPUT_LOCATION_DIAGNOSTIC_ONLY',results=summaries,
          elapsed_seconds=time.perf_counter()-started,source_causal_status=read(source/'result.json')['status']))
    write(out/'evaluation-seal.json',dict(time_ns=time.time_ns(),summary_sha256=sha(out/'summary.json')))
    print(json.dumps(dict(status='COMPLETE',elapsed_seconds=time.perf_counter()-started,replay=replay)))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('source','source-code','output','protocol'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.source,a.source_code,a.output,a.protocol)
