"""One fixed-model data-condition experiment; explicit stage boundaries."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from run_spatial_bce import read, write, sha, seal as _seal, check_seal, T, OLD, CHECKPOINT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'artifacts.local/work/ba-data-coverage-20260921'
SOURCE = ROOT/'artifacts.local/work/ba-spatial-bce-20260920'
TRANSFER = ROOT/'artifacts.local/work/ba-spatial-complement-transfer-20260921'
R_SOURCE = ROOT/'artifacts.local/work/ba-corridor-relative-20260921'
CODE = ('data_coverage_spec.py','data_coverage_capture.py','launch_data_coverage.py',
    'launch_spatial_bce.py','ue_capture_readiness.py','spatial_bce_spec.py',
    'core_transfer_spec.py','run_core_transfer.py','run_spatial_bce.py',
    'spatial_bce_model.py','corridor_relative_model.py','run_corridor_relative.py',
    'full_event_metrics_20260920.py','tof_fov45_core.py','tof_corridor_calibration.py',
    'ba_camera_corridor.py','ba_camera_corridor_spec.py','evaluate_ba_camera_corridor.py',
    'tof_lateral_core.py','ba_camera_corridor_metrics.py')

def seal(*args):
    return _seal(OUT,*args) if len(args)==2 else _seal(*args)

def check(name):
    check_seal(OUT,name)

def verify(out=OUT):
    p=read(out/'protocol.json')
    assert sha(out/'spec.json')==p['spec_sha256']
    assert sha(out/'protocol-before-run.md')==p['protocol_text_sha256']
    for name,digest in p['code_hashes'].items():
        assert sha(HERE/name)==digest,name
    for name,digest in p['input_hashes'].items():
        assert sha(ROOT/name)==digest,name
    from run_spatial_complement_transfer import old_test_unactivated
    old_test_unactivated()
    if (out/'execution-seal.json').exists():
        execution=read(out/'execution-seal.json')
        assert execution['protocol_sha256']==sha(out/'protocol.json')
        for name,digest in execution['hashes'].items():
            assert sha(HERE/name)==digest,name
        for name,digest in execution['additional_input_hashes'].items():
            assert sha(ROOT/name)==digest,name
        assert sha(out/'protocol-clarification.json')==execution['clarification_sha256']
    return p

def freeze():
    from data_coverage_spec import specification,check_spec
    assert not OUT.exists(),'One new cohort only'
    spec=specification()
    old=[ROOT/'artifacts.local/work'/n/'spec.json' for n in OLD]+[SOURCE/'spec.json',TRANSFER/'spec.json']
    admission=check_spec(spec,[read(p) for p in old])
    from run_spatial_complement_transfer import old_test_unactivated
    old_test_unactivated()
    OUT.mkdir(parents=True)
    write(OUT/'spec.json',spec);write(OUT/'preflight.json',admission)
    (OUT/'protocol-before-run.md').write_bytes((HERE/'DATA_COVERAGE_PROTOCOL_20260921.md').read_bytes())
    deps=old+[CHECKPOINT,SOURCE/'fit/head_last.pt',SOURCE/'fit/train_receipt.json',
        SOURCE/'protocol.json',SOURCE/'operating-point.json',R_SOURCE/'head_last.pt',
        R_SOURCE/'protocol.json',ROOT/'tools/research_backend.py',ROOT/'tools/run_obstacle_research.py',
        ROOT/'research/active/dtr-r0/unreal/street_process_lifecycle.py']
    write(OUT/'protocol.json',dict(id=OUT.name,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frames=3456,clips=144,frames_per_clip=24,groups=48,dt_s=.2,capture_timeout_s=3600,
        spec_sha256=sha(OUT/'spec.json'),protocol_text_sha256=sha(OUT/'protocol-before-run.md'),
        code_hashes={n:sha(HERE/n) for n in CODE},
        input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in deps},
        strong_threshold=T,scope='PROSPECTIVE_SAME_GENERATOR_DEVELOPMENT',
        execution_seal_required_before_materialize=True,automatic_successor=False))
    verify();print('FROZEN',admission,flush=True)

def freeze_execution():
    verify()
    names=('run_data_coverage.py','data_coverage_data.py','data_coverage_learning.py',
           'test_data_coverage_spec.py','test_data_coverage_learning.py','data_coverage_regression.py')
    write(OUT/'protocol-clarification.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        scope='Pre-fit implementation clarification; no model outcomes accessed',
        segment_caps='The frozen inherited costs function applies the segment cap to BOTH current and hold. Protocol prose abbreviated this as held segments; both checks remain enforced.',
        frozen_B_threshold_source='Later spatial-complement-transfer protocol high_logit; original operating-point is null',
        evaluation_authority='New layout groups in same procedural simulator Development, not protected final'))
    assert read(TRANSFER/'protocol.json')['high_logit']==7.6612162590026855
    assert read(R_SOURCE/'selection.json')['threshold']==5.128307342529297
    deps=[TRANSFER/'protocol.json',R_SOURCE/'selection.json',TRANSFER/'prediction-seal.json',
          TRANSFER/'features/feature_receipt.json',R_SOURCE/'prediction-seal.json']
    write(OUT/'execution-seal.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(OUT/'protocol.json'),hashes={n:sha(HERE/n) for n in names},
        additional_input_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in deps},
        clarification_sha256=sha(OUT/'protocol-clarification.json')))
    print('EXECUTION_SEALED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('freeze','freeze_execution','materialize','prepare','fit','select','predict','evaluate'))
    stage=parser.parse_args().stage
    if stage in ('freeze','freeze_execution'):
        globals()[stage]()
    else:
        assert (OUT/'execution-seal.json').exists(),'Execution code must be frozen first'
        verify()
        if stage=='materialize':
            from data_coverage_data import materialize
            materialize(OUT)
        else:
            import data_coverage_learning
            getattr(data_coverage_learning,stage)()
