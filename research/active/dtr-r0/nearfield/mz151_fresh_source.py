"""Prospective MZ151 source admission, independent of models and outcomes.

The source design is prepared before candidate evaluation. A separate final
capture freeze must bind the accepted model, thresholds and protocol.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import mz136_paired_source as inherited

ROOT=Path(__file__).resolve().parents[4]
SEED=151016
PRIOR_SEEDS=(136014,146016)
WORK=ROOT/'artifacts.local/work/mz151-expanded-training-20260916'
AUTHORITY='FRESH_SCENE_CONFIGURATION_CONTROLLED_CONFIRMATION_SAME_SIMULATOR_NOT_HARDWARE_OR_NATURAL'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def at_seed(seed):
    previous=inherited.SEED
    try:
        inherited.SEED=seed
        result=inherited.source()
        inherited.check_source(result)
        return result
    finally:
        inherited.SEED=previous

def rename(value):
    if isinstance(value,str):return value.replace('mz136','mz151')
    if isinstance(value,list):return [rename(v) for v in value]
    if isinstance(value,dict):return {k:rename(v) for k,v in value.items()}
    return value

def geometry_signatures(spec):
    groups={g['scene_group']:[] for g in spec['scene_groups']}
    for frame in spec['frames']:
        groups[frame['scene_group']].append(dict(camera=frame['camera'],body_origin_m=frame['body_origin_m'],
            objects=[{k:o[k] for k in ('center_m','size_m')} for o in frame['objects']]))
    return {inherited.digest(v) for v in groups.values()}

def check_source(spec):
    assert spec['seed']==SEED and spec['authority']==AUTHORITY
    assert Counter(f['split'] for f in spec['frames'])=={'confirmation':288}
    assert all(f['id'].startswith('mz151_') for f in spec['frames'])
    assert len({f['episode'] for f in spec['frames']})==48
    copy_spec=copy.deepcopy(spec)
    for key in ('frames','scene_groups','pairs','source_audit'):
        for item in copy_spec[key]:
            assert item['split']=='confirmation'
            item['split']=item.pop('generator_partition')
    original=inherited.check_source(copy_spec)
    assert len(geometry_signatures(spec))==24
    checks=spec['prior_source_disjointness']
    assert [p['prior_seed'] for p in checks]==list(PRIOR_SEEDS)
    assert all(p['background_geometry_signature_overlap']==0 and p['scene_geometry_signature_overlap']==0 for p in checks)
    return dict(original,split_frames={'confirmation':288},source_seed=SEED,
        prior_source_disjointness=checks,geometry_signatures=sorted(geometry_signatures(spec)),
        inherited_partition_is_not_evaluation_split=True,authority=AUTHORITY,
        predictor_access='NO_IDS_FAMILY_GEOMETRY_SEEDS_PARTITIONS_OR_SOURCE_LABELS')

def source():
    fresh=at_seed(SEED)
    background={g['background_geometry_signature'] for g in fresh['scene_groups']}
    geometry=geometry_signatures(fresh);checks=[]
    for seed in PRIOR_SEEDS:
        prior=at_seed(seed)
        bg_overlap=background & {g['background_geometry_signature'] for g in prior['scene_groups']}
        full_overlap=geometry & geometry_signatures(prior)
        assert not bg_overlap and not full_overlap,'Source configuration overlap'
        checks.append(dict(prior_seed=seed,background_geometry_signature_overlap=0,scene_geometry_signature_overlap=0,scene_groups=24))
    result=rename(fresh)
    for key in ('frames','scene_groups','pairs','source_audit'):
        for item in result[key]:
            item['generator_partition']=item['split'];item['split']='confirmation'
    result['authority']=AUTHORITY
    result['source_selection']='ONE_FIXED_SEED_NO_CAPTURE_OR_MODEL_OUTCOME_SELECTION'
    result['prior_source_disjointness']=checks
    result['limitations'].append('New configurations share family priors, renderer and sensor simulator; no natural-scene or physical-sensor claim.')
    check_source(result)
    return result

def prepare(output):
    output=Path(output).resolve()
    assert output.is_relative_to(WORK.resolve()) and not output.exists()
    spec=source();audit=check_source(spec)
    output.mkdir(parents=True)
    for name,value in (('spec.json',spec),('source-audit.json',audit)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    snapshot=output/'source-snapshot';snapshot.mkdir()
    files=[Path(__file__),Path(inherited.__file__)]
    for p in files:shutil.copyfile(p,snapshot/p.name)
    freeze=dict(source_seed=SEED,files={n:sha(output/n) for n in ('spec.json','source-audit.json')},
        source_files={str(p):sha(p) for p in files},authority=AUTHORITY,
        candidate_binding='PENDING_ROOT_MODEL_THRESHOLD_PROTOCOL_SEAL_AND_DEV_GATE_PASS',
        capture_authorized=False,capture_attempt_budget=1,capture_timeout_seconds=1200,
        outcome_access_before_source_design=False,original_mz136_test_records_read=False)
    (output/'source-design-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    result=dict(status='SOURCE_DESIGN_FROZEN_NOT_CAPTURE_AUTHORIZED',output=str(output),spec_sha256=sha(output/'spec.json'),
        source_design_freeze_sha256=sha(output/'source-design-freeze.json'),audit=audit)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
