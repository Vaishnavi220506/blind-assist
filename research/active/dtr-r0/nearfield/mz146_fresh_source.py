"""One prospectively frozen MZ146 confirmation source; no outcomes or inference.

Reuse the unchanged MZ136 geometry generator at seed 146016, then relabel every
record confirmation. Inherited partitions are source-audit metadata only.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import mz136_paired_source as inherited


ROOT = Path(__file__).resolve().parents[4]
SEED = 146016
WORK = ROOT/'artifacts.local/work/mz146-fresh-corridor-confirmation-20260916'
AUTHORITY = 'FRESH_SCENE_CONFIGURATION_CONTROLLED_CONFIRMATION_SAME_SIMULATOR_NOT_HARDWARE_OR_NATURAL'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rename(value):
    if isinstance(value, str):
        return value.replace('mz136', 'mz146')
    if isinstance(value, list):
        return [_rename(x) for x in value]
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    return value


def check_source(spec):
    assert spec['seed'] == SEED and spec['authority'] == AUTHORITY
    assert Counter(f['split'] for f in spec['frames']) == {'confirmation': 288}
    assert all(f['id'].startswith('mz146_') for f in spec['frames'])
    assert len({f['episode'] for f in spec['frames']}) == 48
    audit_copy = copy.deepcopy(spec)
    for key in ('frames', 'scene_groups', 'pairs', 'source_audit'):
        for item in audit_copy[key]:
            assert item['split'] == 'confirmation'
            item['split'] = item.pop('generator_partition')
    original_checks = inherited.check_source(audit_copy)
    return dict(original_checks, split_frames={'confirmation': 288}, source_seed=SEED,
        inherited_partition_is_not_evaluation_split=True, authority=AUTHORITY,
        predictor_access='NO_IDS_FAMILY_GEOMETRY_SEEDS_PARTITIONS_OR_SOURCE_LABELS')


def source():
    previous_seed = inherited.SEED
    try:
        inherited.SEED = SEED
        fresh = inherited.source()
        inherited.check_source(fresh)
    finally:
        inherited.SEED = previous_seed
    old = inherited.source()
    old_groups = {g['background_geometry_signature'] for g in old['scene_groups']}
    fresh_groups = {g['background_geometry_signature'] for g in fresh['scene_groups']}
    assert not old_groups & fresh_groups, 'New source must have disjoint scene configurations'
    result = _rename(fresh)
    for key in ('frames', 'scene_groups', 'pairs', 'source_audit'):
        for item in result[key]:
            item['generator_partition'] = item['split']
            item['split'] = 'confirmation'
    result['authority'] = AUTHORITY
    result['source_selection'] = 'ONE_FIXED_SEED_NO_CAPTURE_OR_MODEL_OUTCOME_SELECTION'
    result['prior_source_disjointness'] = dict(prior_seed=previous_seed,
        background_geometry_signature_overlap=0, scene_groups=24,
        reused_family_priors_and_simulator=True)
    result['limitations'].append('Confirmation scene groups are new configurations, not a new renderer, sensor model or natural distribution.')
    check_source(result)
    assert {f['id'] for f in old['frames']}.isdisjoint(f['id'] for f in result['frames'])
    return result


def freeze(output):
    output = Path(output).resolve()
    assert output.is_relative_to(WORK.resolve()) and not output.exists()
    model_root = ROOT/'artifacts.local/work/mz143-corridor-evidence-20260916/run-v1'
    onset_root = ROOT/'artifacts.local/work/mz145-causal-confirmation-20260916/run-v1'
    model_seal = json.loads((model_root/'model-seal.json').read_text())
    onset = json.loads((onset_root/'onset-seal.json').read_text())
    assert model_seal['selected'] == onset['source_arm'] == 'fused_hgb'
    assert onset['low'] == .020014435971556582 and onset['high'] == .2957935335969224
    assert sha(model_root/'fused_hgb.pkl') == model_seal['models']['fused_hgb']
    assert sha(model_root/'model-seal.json') == onset['model_seal_sha256']
    feature_freeze = json.loads((model_root/'freeze.json').read_text())
    feature_sources = feature_freeze['sources']
    code = Path(__file__).parent
    for name in ('mz143_corridor_features.py', 'mz125_observable_correction.py',
                 'mz136_boundary_geometry.py', 'mz115_spatial_allocation.py'):
        assert sha(code/name) == feature_sources[str(code/name)]
    assert sha(code/'mz145_causal_confirmation.py') == onset['sources'][str(code/'mz145_causal_confirmation.py')]
    frozen_files = [model_root/'fused_hgb.pkl', model_root/'model-seal.json',
        model_root/'freeze.json', onset_root/'onset-seal.json',
        code/'mz143_corridor_features.py', code/'mz145_causal_confirmation.py',
        code/'mz136_paired_source.py', code/'MZ146_PROTOCOL_20260916.md', Path(__file__)]
    spec = source(); audit = check_source(spec)
    output.mkdir(parents=True)
    for name, item in (('spec.json', spec), ('source-audit.json', audit)):
        (output/name).write_text(json.dumps(item, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    snapshot = output/'frozen-inputs'; snapshot.mkdir()
    for path in frozen_files:
        destination = snapshot/path.name
        if destination.exists():
            destination = snapshot/(path.parent.parent.name+'-'+path.name)
        shutil.copyfile(path, destination)
    result = dict(source_seed=SEED, source_sha256=sha(__file__),
        inherited_generator_sha256=sha(inherited.__file__),
        files={name: sha(output/name) for name in ('spec.json', 'source-audit.json')},
        frozen_inputs={str(path): sha(path) for path in frozen_files},
        frozen_predictor_sources=feature_sources,
        candidate=dict(arm='fused_hgb', model_sha256=sha(model_root/'fused_hgb.pkl'),
            low=onset['low'], high=onset['high'], temporal_source_sha256=sha(code/'mz145_causal_confirmation.py')),
        authority=AUTHORITY, capture_attempt_budget=1, capture_timeout_seconds=1200,
        outcome_access_before_freeze=False, no_original_mz136_test_access=True)
    (output/'freeze.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), spec_sha256=sha(output/'spec.json'),
        freeze_sha256=sha(output/'freeze.json'), source_audit=audit), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output)
