"""Make one runnable artifact bundle and bind it before confirmation capture."""
from pathlib import Path
import shutil
from prepare_public_positive import ROOT,HERE,WORK,read,write,sha
from run_mz139_surface_fit import local_dependencies

HOME=WORK/'corridor-public-single-20260917'


def main():
    out=HOME/'bundle'
    assert not (HOME/'recipe-freeze.json').exists()
    assert not (out/'config.json').exists()
    head=HOME/'head';control=HOME/'a-control'
    done=read(head/'completion.json');assert done['status']=='PASS'
    assert sha(head/'model.pt')==done['model_sha256']
    assert sha(head/'selection.json')==done['selection_sha256']
    assert read(control/'completion.json')['status']=='PASS'
    a=WORK/'corridor-depth-e1-20260917/run-v2/A-model.pkl'
    assert sha(a)=='d1e406a9793c3717f363b2e4698c91753580ed2d4dafe81d9820ab521b499cef'
    out.mkdir(exist_ok=True)
    for name,p in [('A.pkl',a),('positive.pt',head/'model.pt'),('A-retrained.pkl',control/'retrainedA.pkl')]:
        if (out/name).exists():
            assert sha(out/name)==sha(p), 'Existing partial bundle differs'
        else:
            shutil.copyfile(p,out/name)
    threshold=read(control/'threshold.json')
    control_tau=threshold['threshold']
    config=dict(A_threshold=.3917890013717321,
        positive_threshold=read(head/'selection.json')['chosen']['threshold'],control_threshold=control_tau,
        model_hashes={name:sha(out/name) for name in ['A.pkl','positive.pt','A-retrained.pkl']},
        method='ONE_FROZEN_A_OR_ONE_PUBLIC_RETURN_BCE_HEAD',parameters=1537,
        no_scene_router=True,unknown_retains_A=True)
    write(out/'config.json',config)
    sources=local_dependencies(HERE/'single_positive_inference.py')
    for p in [HERE/'PUBLIC_SINGLE_PROTOCOL_20260917.md',Path(__file__),HERE/'train_single_positive.py',
              HERE/'prepare_single_a.py',HERE/'train_single_a.py']:
        sources[str(p)]=sha(p)
    write(HOME/'recipe-freeze.json',dict(status='FROZEN_BEFORE_NEW_CAPTURE',config=config,
        bundle_files={str(p):sha(p) for p in out.iterdir()},sources=sources,
        training_completion_sha256=sha(head/'completion.json'),control_completion_sha256=sha(control/'completion.json'),
        source_seed=186017,frames=288,scope='PROSPECTIVE_SAME_SIMULATOR_CONFIGURATION_CONFIRMATION'))
    print(read(HOME/'recipe-freeze.json')['config'])


if __name__=='__main__':main()
