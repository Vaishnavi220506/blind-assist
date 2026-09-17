"""Texture-only factorial interventions; geometry and native-sensor inputs fixed."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mz136_paired_source as inherited

SEED = 189018
AUTHORITY = 'CONTROLLED_TEXTURE_FACTORIAL_DEVELOPMENT_NOT_CAUSAL_IDENTIFICATION_OR_HARDWARE'


def strip(frame, kind):
    f = copy.deepcopy(frame)
    for key in ('id', 'episode', 'scene_group', 'pair_id', 'pair_member', 'pair_variant',
                'background_variant', 'physical_group', 'scene_index'):
        f.pop(key, None)
    if kind == 'lateral':
        f['objects'][0]['center_m'][1] = 0.
    else:
        for obj in f['objects'][1:]:
            for key in ('texture', 'texture_seed', 'texture_grid'):
                obj.pop(key, None)
    return f


def source():
    old = inherited.SEED
    try:
        inherited.SEED = SEED
        spec = inherited.source()
    finally:
        inherited.SEED = old
    original = spec['frames']
    frames, groups, pairs, audits = [], [], [], []
    for fi, family in enumerate(inherited.FAMILIES):
        for k in range(3):
            physical = f'bg189_{family}_scene{k}'
            split = 'train' if k < 2 else 'dev'
            originals = [f for f in original if f['family'] == family and f['scene_group'].endswith(f'_scene{k}')]
            for background in ('plain', 'textured'):
                group = physical+'_'+background
                episodes = []
                for f0 in originals:
                    f = copy.deepcopy(f0)
                    member = f['pair_member']
                    episode = group+'_'+member
                    f.update(id=f"{episode}_{round(f['time_s']/.25):02d}", episode=episode,
                        scene_group=group, physical_group=physical, scene_index=k,
                        pair_id=group, split=split, background_variant=background)
                    for j, obj in enumerate(f['objects'][1:]):
                        obj['texture'] = background == 'textured'
                        obj['texture_seed'] = SEED+fi*1009+k*113+j*31
                        obj['texture_grid'] = [11, 7]
                    frames.append(f)
                    if episode not in episodes:
                        episodes.append(episode)
                groups.append(dict(scene_group=group, physical_group=physical, split=split,
                    category=family, scene_index=k, background_variant=background, episodes=episodes))
                pairs.append(dict(pair_id=group, scene_group=group, physical_group=physical,
                    split=split, category=family, episodes=episodes, intervention='TARGET_LATERAL_CENTERS_ONLY'))
                for episode in episodes:
                    ff = [f for f in frames if f['episode'] == episode]
                    audits.append(dict(episode=episode, scene_group=group, split=split,
                        source_aabb_labels=[any(inherited.intersects(f, o) for o in f['objects']) for f in ff]))
    spec.update(schema='bg189-texture-factorial-v1', authority=AUTHORITY,
        frames=frames, scene_groups=groups, pairs=pairs, source_audit=audits,
        intervention='CONTEXT_TEXTURE_ONLY_NO_COLLISION_NO_REFLECTANCE_CHANGE',
        limitations=spec['limitations']+[
            'Plain versus procedural grayscale albedo tiles only; no wall geometry or clutter intervention.',
            'Renderer texture tiles are non-colliding and unchanged context stays behind target.',
            'Exact public sensor equality and observable RGB change must be audited after capture.',
            'Only four held physical groups: mechanism pilot, not statistical or natural-scene confirmation.'])
    check_source(spec)
    return spec


def pair_indices(spec, subset=None):
    frames = spec['frames'] if subset is None else subset
    lookup = {(f['physical_group'], f['background_variant'], f['pair_member'], f['time_s']): i for i, f in enumerate(frames)}
    rank, invariant = [], []
    for i, f in enumerate(frames):
        if f['background_variant'] == 'plain':
            j = lookup[(f['physical_group'], 'textured', f['pair_member'], f['time_s'])]
            assert strip(f, 'background') == strip(frames[j], 'background')
            invariant.append((i, j))
        positive = any(inherited.intersects(f, o) for o in f['objects'])
        if positive:
            other = {'in':'out', 'out':'in', 'enter':'exit', 'exit':'enter'}[f['pair_member']]
            j = lookup[(f['physical_group'], f['background_variant'], other, f['time_s'])]
            assert strip(f, 'lateral') == strip(frames[j], 'lateral')
            assert not any(inherited.intersects(frames[j], o) for o in frames[j]['objects'])
            rank.append((i, j))
    return rank, invariant


def check_source(spec):
    frames = spec['frames']
    assert spec['seed'] == SEED and spec['authority'] == AUTHORITY
    assert len(frames) == len({f['id'] for f in frames}) == 288
    assert Counter(f['split'] for f in frames) == {'train':192, 'dev':96}
    assert len({f['physical_group'] for f in frames}) == 12
    assert len({f['episode'] for f in frames}) == 48
    for f in frames:
        target = f['objects'][0]
        assert all(not inherited.intersects(f, o) for o in f['objects'][1:])
        assert all(target['center_m'][0]+target['size_m'][0]/2 < o['center_m'][0]-o['size_m'][0]/2 for o in f['objects'][1:])
        assert not inherited.intersects(f, spec['background']) and not inherited.intersects(f, spec['floor'])
    rank, inv = pair_indices(spec)
    assert len(rank) == len(inv) == 144
    for a, b in inv:
        assert frames[a]['objects'][0] == frames[b]['objects'][0]
        assert frames[a]['split'] == frames[b]['split']
    return dict(status='PASS', frames=288, physical_groups=12, train_frames=192, dev_frames=96,
        intrusion_pairs=144, texture_pairs=144, target_geometry_appearance_unchanged=True,
        context_native_geometry_reflectance_unchanged=True, sensor_random_seeds_unchanged=True,
        all_context_behind_target=True, target_observability_scope='SOURCE_DEPTH_ORDER_NOT_RGB_PIXEL_AUDIT')


if __name__ == '__main__':
    print(json.dumps(check_source(source()), indent=2))
