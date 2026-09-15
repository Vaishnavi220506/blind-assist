"""Post-result, no-update TRAIN-only BatchNorm diagnostic; never a rescue arm."""
import argparse
import json
from pathlib import Path
import gc
import cv2
import numpy as np
import torch
from mz140_depthor import make_input
from mz141_depthor_training import load_trainable, training_prediction, supervision, loss_contribution
from run_mz141_depthor_training import BASE, CAP, module_digest
from run_mz139_surface_fit import read, selected_jsonl
from run_mz107_four_sensor import sha, write
from mz136_incumbent import public_observations
from evaluate_mz140_depthor import rasterize_native_bounds


def run(source):
    source=source.resolve()
    assert read(source/'completion.json')['status']=='PASS'
    out=source.parent/'normalization-diagnostic.json'
    assert not out.exists()
    freeze=read(source/'freeze.json')
    families=sorted({m['family'] for m in freeze['metadata'].values()})
    ids=[sorted(key for key,m in freeze['metadata'].items()
                if m['family']==family and m['scene_group'].endswith('_scene0'))[2] for family in families]
    assert all(freeze['metadata'][key]['partition']=='fit' for key in ids)
    rows={r['id']:r for r in public_observations(selected_jsonl(CAP/'raw.jsonl',set(ids)))}
    native={r['id']:r for r in selected_jsonl(CAP/'evaluator.jsonl',set(ids))}
    torch.set_num_threads(4)
    results=[]
    for arm in ('pixel','surface'):
        checkpoint=source/f'checkpoints/{arm}-epoch8.pt'
        checkpoint_sha=sha(checkpoint)
        model,_=load_trainable(BASE/'upstream',BASE/'depthor-zju-small.pt')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True)
        del payload
        state_sha=module_digest(model,'')
        assert state_sha==read(source/f'{arm}-fit-receipt.json')['final_state_sha256']
        for key in ids:
            row=rows[key]
            labels=supervision(rasterize_native_bounds(row,native[key],{}))
            data,_=make_input(row,cv2.imread(str(CAP/row['rgb_path'])))
            for mode in ('eval','current_frame_bn_only'):
                model.eval()
                norms=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
                for module in norms:
                    module.track_running_stats=(mode=='eval')
                    module.train(mode!='eval')
                with torch.no_grad():
                    prediction=training_prediction(model,data)
                    raw=prediction.cpu().numpy()
                    depth=np.clip(raw,.001,10.)
                    loss=float(loss_contribution(prediction,labels,arm,labels['audit']['known_pixels'],1))
                target=native[key]
                reference=rasterize_native_bounds(row,target,{})
                mask=reference['target_visible'] & np.isfinite(reference['depth'])
                result=dict(arm=arm,id=key,family=freeze['metadata'][key]['family'],mode=mode,
                    target_mae_m=float(np.abs(depth[mask]-reference['depth'][mask]).mean()),
                    target_agreement_fraction=float((np.abs(depth[mask]-reference['depth'][mask])<=.12).mean()),
                    objective_per_frame=loss,raw_min_m=float(raw.min()),raw_max_m=float(raw.max()),
                    raw_nonfinite_pixels=int((~np.isfinite(raw)).sum()),bn_modules=len(norms))
                if mode=='eval':
                    saved=read(source/f'{arm}-predictions.json')[key]
                    result['saved_eval_max_difference_m']=float(np.abs(depth-np.load(saved['path'])).max())
                    assert result['saved_eval_max_difference_m']<=1e-6
                results.append(result)
        assert state_sha==module_digest(model,'') and checkpoint_sha==sha(checkpoint)
        del model,data,prediction
        gc.collect();torch.cuda.empty_cache()
    write(out,dict(status='PASS',source_sha256=sha(Path(__file__)),training_run_summary_sha256=sha(source/'summary.json'),
        ids=ids,selection='THIRD_LEXICAL_FRAME_OF_EACH_FIT_SCENE0_FIXED_BEFORE_DIAGNOSTIC',
        authority='POSTHOC_TRAIN4_NO_UPDATE_DIAGNOSIS_NOT_ADMISSION_OR_RETRAINING',
        no_optimizer_updates=True,checkpoint_and_all_tensor_state_unchanged=True,
        mode='ALL_MODEL_EVAL_EXCEPT_BN_TRAIN_WITH_TRACK_RUNNING_STATS_FALSE; DROPPATH_OFF_BOTH',results=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run)
