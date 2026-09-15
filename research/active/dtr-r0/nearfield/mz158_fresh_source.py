"""Fixed fresh MZ158 source, bound to a root-owned agreement recipe before freeze.

Only source-design geometry is generated here. No captured records, predictor,
training, protected test outcomes or candidate selection enter this module.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import mz136_paired_source as inherited


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work/mz158-crossview-agreement-20260916'
SOURCE = WORK/'source'
SEED = 158016
PRIOR_SEEDS = (136014, 146016)
AUTHORITY = 'FRESH_SCENE_CONFIGURATION_CONTROLLED_CONFIRMATION_SAME_SIMULATOR_NOT_HARDWARE_OR_NATURAL'
CAPTURE_NAMES = ('mz115_zonal_capture.py', 'ue_capture_readiness.py',
    'mz115_zonal_sensors.py', 'mz115_zonal_tof.py', 'mz113_dynamic_sensors.py',
    'mz99_angle_information_capture.py', 'run_mz123_frozen_early.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _at_seed(seed):
    previous = inherited.SEED
    try:
        inherited.SEED = seed
        return inherited.source()
    finally:
        inherited.SEED = previous


def _rename(value):
    if isinstance(value, str):
        return value.replace('mz136', 'mz158')
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, dict):
        return {key: _rename(item) for key, item in value.items()}
    return value


def check_source(spec):
    assert spec['seed'] == SEED and spec['authority'] == AUTHORITY
    assert Counter(f['split'] for f in spec['frames']) == {'confirmation': 288}
    assert all(f['id'].startswith('mz158_') for f in spec['frames'])
    assert len({f['episode'] for f in spec['frames']}) == 48
    audit_copy = copy.deepcopy(spec)
    for key in ('frames', 'scene_groups', 'pairs', 'source_audit'):
        for item in audit_copy[key]:
            assert item['split'] == 'confirmation'
            item['split'] = item.pop('generator_partition')
    original = inherited.check_source(audit_copy)
    disjoint = spec['prior_source_disjointness']
    assert [item['prior_seed'] for item in disjoint] == list(PRIOR_SEEDS)
    fresh = {g['background_geometry_signature'] for g in spec['scene_groups']}
    assert len(fresh) == 24
    for item in disjoint:
        assert item['scene_groups'] == 24
        assert item['background_geometry_signature_overlap'] == 0
        assert len(set(item['prior_geometry_signatures'])) == 24
        assert not fresh.intersection(item['prior_geometry_signatures'])
    return dict(original, split_frames={'confirmation': 288}, source_seed=SEED,
        prior_seeds=list(PRIOR_SEEDS), prior_geometry_overlaps=[0, 0],
        inherited_partition_is_not_evaluation_split=True, authority=AUTHORITY,
        predictor_access='NO_IDS_FAMILY_GEOMETRY_SEEDS_PARTITIONS_OR_SOURCE_LABELS')


def source():
    fresh = _at_seed(SEED)
    signatures = {g['background_geometry_signature'] for g in fresh['scene_groups']}
    prior_checks = []
    for seed in PRIOR_SEEDS:
        prior = _at_seed(seed)
        old = {g['background_geometry_signature'] for g in prior['scene_groups']}
        assert not signatures & old, 'Fixed new configurations overlap prior source'
        prior_checks.append(dict(prior_seed=seed, scene_groups=24,
            background_geometry_signature_overlap=0, prior_geometry_signatures=sorted(old)))
    result = _rename(fresh)
    for key in ('frames', 'scene_groups', 'pairs', 'source_audit'):
        for item in result[key]:
            item['generator_partition'] = item['split']
            item['split'] = 'confirmation'
    result['authority'] = AUTHORITY
    result['source_selection'] = 'ONE_FIXED_SEED_NO_CAPTURE_OR_MODEL_OUTCOME_SELECTION'
    result['prior_source_disjointness'] = prior_checks
    result['limitations'].append(
        'New configurations reuse the same family priors, renderer and sensor simulator; this is not a new natural distribution.')
    check_source(result)
    return result


def freeze(output, recipe_freeze, recipe_freeze_sha256):
    """Root supplies the exact frozen recipe path and digest; never invent one."""
    output = Path(output).resolve()
    if not output.is_relative_to(SOURCE.resolve()) or output.exists():
        raise ValueError('A fresh directory under the task-owned source tree is required')
    if recipe_freeze is None or not re.fullmatch(r'[0-9a-f]{64}', recipe_freeze_sha256 or ''):
        raise ValueError('Root-owned recipe freeze and exact SHA256 are required before design freeze')
    recipe = Path(recipe_freeze).resolve(strict=True)
    if not recipe.is_file() or not recipe.is_relative_to(WORK.resolve()) or recipe.is_relative_to(SOURCE.resolve()):
        raise ValueError('Recipe freeze must be an existing experiment file outside the source-owned tree')
    if sha(recipe) != recipe_freeze_sha256:
        raise ValueError('Root-supplied recipe freeze digest mismatch')
    decoded = json.loads(recipe.read_text(encoding='utf-8-sig'))
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError('Recipe freeze must be a nonempty JSON object')
    code = Path(__file__).resolve().parent
    frozen_paths = [Path(__file__).resolve(), Path(inherited.__file__).resolve(), recipe]
    frozen_paths += [code/name for name in CAPTURE_NAMES]
    frozen_paths += [ROOT/'tools/ue_native_capture.py',
        ROOT/'research/active/dtr-r0/unreal/street_process_lifecycle.py',
        ROOT/'tools/run_city_pcg_capture.py', SOURCE/'launch_capture.py', SOURCE/'prepare_bundle.py']
    hashes = {str(path.resolve()): sha(path) for path in frozen_paths}
    spec = source()
    spec['recipe_binding'] = dict(path=str(recipe), sha256=recipe_freeze_sha256)
    audit = check_source(spec)
    output.mkdir(parents=True)
    write(output/'spec.json', spec)
    write(output/'source-audit.json', audit)
    shutil.copyfile(recipe, output/'recipe-freeze.json')
    snapshot = output/'frozen-inputs'
    snapshot.mkdir()
    snapshot_hashes = {}
    for index, path in enumerate(frozen_paths):
        destination = snapshot/(f'{index:02d}-'+path.name)
        shutil.copyfile(path, destination)
        assert sha(destination) == hashes[str(path.resolve())]
        snapshot_hashes[str(destination.relative_to(output)).replace('\\', '/')] = sha(destination)
    assert hashes == {str(path.resolve()): sha(path) for path in frozen_paths}
    result = dict(source_seed=SEED, source_sha256=sha(__file__),
        inherited_generator_sha256=sha(inherited.__file__),
        files={name: sha(output/name) for name in ('spec.json', 'source-audit.json', 'recipe-freeze.json')},
        frozen_inputs=hashes, source_snapshots=snapshot_hashes,
        recipe_freeze=dict(path=str(recipe), sha256=recipe_freeze_sha256,
            validation_owner='ROOT_RECIPE_MODEL_PROTOCOL_OWNER', copied_file='recipe-freeze.json'),
        authority=AUTHORITY, capture_attempt_budget=1, capture_timeout_seconds=1200,
        dispatch_requires_separate_root_go_ahead=True,
        source_outcome_selection=False, no_original_mz136_test_records_accessed=True,
        prior_comparison_scope='REGENERATED_DESIGN_SIGNATURES_ONLY_NO_CAPTURED_RECORDS')
    write(output/'freeze.json', result)
    return dict(output=str(output), spec_sha256=sha(output/'spec.json'),
        freeze_sha256=sha(output/'freeze.json'), recipe_freeze_sha256=recipe_freeze_sha256, source_audit=audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe-freeze', type=Path, required=True)
    parser.add_argument('--recipe-freeze-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.recipe_freeze, args.recipe_freeze_sha256), indent=2))
