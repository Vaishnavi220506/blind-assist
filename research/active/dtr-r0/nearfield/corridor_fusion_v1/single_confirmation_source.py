"""One fixed targeted S1 confirmation design, without model/outcome access."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mz136_paired_source as inherited

SEED = 186017
DISTANCES = (3.25, 3.55, 3.85, 4.15, 4.45, 4.75)
AUTHORITY = 'FRESH_TARGETED_CONTROLLED_CONFIRMATION_SAME_SIMULATOR_NOT_NATURAL_OR_HARDWARE'


def source():
    previous = inherited.SEED
    try:
        inherited.SEED = SEED
        spec = inherited.source()
    finally:
        inherited.SEED = previous
    spec = json.loads(json.dumps(spec).replace('mz136_', 'singleconfirm_'))
    spec['schema'] = 'single-public-positive-confirmation-v1'
    spec['authority'] = AUTHORITY
    spec['source_selection'] = 'ONE_PREDECLARED_DESIGN_NO_OBSERVATION_OR_OUTCOME_SELECTION'
    for collection in ('frames', 'scene_groups', 'pairs', 'source_audit'):
        for item in spec[collection]:
            item['generator_partition'] = item.pop('split')
            item['split'] = 'confirmation'
    for index, group in enumerate(spec['scene_groups']):
        fi, k = divmod(index, 6)
        distance = DISTANCES[k]
        rng = random.Random(SEED + 17003 * fi + 179 * k)
        # All ranges have openings; some farther ranges also have a full wall.
        # Near full walls would be true corridor obstacles, not negative controls.
        style = ('portal', 'staggered_panels', 'recessed_opening')[(fi+k) % 3]
        if k >= 2 and (fi+k) % 4 == 0:
            style = 'full_wall'
        front = distance + .023
        thickness = rng.uniform(.08, .16)
        height = rng.uniform(3.4, 4.6)
        grid = ([3, 11], [11, 3], [6, 9], [9, 6])[(fi+k) % 4]
        context = []

        def panel(name, x, y, width, h=height, z=None):
            return dict(name=name, center_m=[x+thickness/2, y, h/2 if z is None else z],
                size_m=[thickness, width, h], texture_seed=rng.randrange(2**30),
                texture_grid=list(grid), tof_reflectance_proxy=rng.uniform(.25, .90),
                source_role='targeted_background_'+style)

        if style == 'full_wall':
            context.append(panel('context0', front, rng.uniform(-.15, .15), rng.uniform(5.0, 6.3)))
        else:
            gap = rng.uniform(.40, .48)  # body half width .30 plus camera motion clearance
            widths = [rng.uniform(2.0, 2.8), rng.uniform(2.0, 2.8)]
            for j, sign in enumerate((-1, 1)):
                offset = .22 * j if style == 'staggered_panels' else 0.
                context.append(panel('context'+str(j), front+offset, sign*(gap+widths[j]/2), widths[j]))
            if style == 'portal':
                context.append(panel('context2', front, 0., 2*gap, h=.45, z=2.75))
            elif style == 'recessed_opening':
                context.append(panel('context2', front+1.1, 0., 2.1, h=3.9))
        # A different rear-plane geometry in every configuration.
        context.append(panel('rear_scene', front+rng.uniform(2.1, 3.2), rng.uniform(-.25,.25), rng.uniform(8.0,10.5), h=6.5))
        rho = (.16, .31, .49, .67, .83, .92)[(fi+k) % 6]
        target_grid = ([3, 7], [8, 3], [5, 11])[(fi+2*k) % 3]
        group_frames = [f for f in spec['frames'] if f['scene_group'] == group['scene_group']]
        for f in group_frames:
            target = f['objects'][0]
            if fi == 0:
                target['center_m'][2] = (1.62, 1.72, 1.82, 1.92, 1.76, 1.86)[k]
                target['size_m'][2] = (.18, .22, .26, .20, .24, .28)[k]
                target['size_m'][0] = (.13, .17, .21, .15, .19, .23)[k]
            if fi == 2:
                target['size_m'][1] = (.024, .038, .050, .062, .074, .086)[k]
                target['size_m'][0] = (.045, .065, .085)[k % 3]
                sign = 1 if target['center_m'][1]-f['camera']['y'] >= 0 else -1
                lateral = (.03, .09, .15, .21, .12, .24)[k] if f['pair_member']=='in' else .3+target['size_m'][1]/2+(.12,.18,.24)[k%3]
                target['center_m'][1] = f['camera']['y']+sign*lateral
            target['center_m'][0] = min(target['center_m'][0], front-.45-target['size_m'][0]/2)
            target['tof_reflectance_proxy'] = rho
            target['texture_grid'] = list(target_grid)
            target['texture_seed'] = SEED+index*103
            f['objects'] = [target]+copy.deepcopy(context)
            f.update(wall_distance_m=distance, background_style=style)
        group.update(wall_distance_m=distance, background_style=style,
            target_reflectance=rho, texture_grid=target_grid,
            target_front_m=group_frames[0]['objects'][0]['center_m'][0]-group_frames[0]['objects'][0]['size_m'][0]/2-.023,
            target_size_m=group_frames[0]['objects'][0]['size_m'],
            background_signature=inherited.digest(context),
            background_geometry_signature=inherited.digest([{k:o[k] for k in ('center_m','size_m')} for o in context]))
    for audit in spec['source_audit']:
        audit['source_aabb_labels'] = [any(inherited.intersects(f,o) for o in f['objects']) for f in spec['frames'] if f['episode']==audit['episode']]
    spec['limitations'] += [
        'Wall distance is the front surface relative to the initial body x, not the observable gate q100.',
        'Near backgrounds retain a corridor opening; full walls at <=3.6m would change all-object truth.',
        'Material variation is grayscale procedural texture and simulated reflectance, not measured physical BRDF.',
        'True q100-above-threshold coverage is measured after predictions; no regeneration to force gate activation.',
        'Background styles and distances are targeted stress strata, not a balanced factorial or natural frequency estimate.']
    check_source(spec)
    return spec


def check_source(spec):
    frames, groups = spec['frames'], spec['scene_groups']
    assert spec['seed']==SEED and spec['authority']==AUTHORITY
    assert len(frames)==288 and len(groups)==24 and len(spec['pairs'])==24
    assert len({f['id'] for f in frames})==288
    assert Counter(f['split'] for f in frames)=={'confirmation':288}
    assert len({g['background_geometry_signature'] for g in groups})==24
    truth={}
    for f in frames:
        assert all(not inherited.intersects(f,o) for o in f['objects'][1:])
        assert not inherited.intersects(f,spec['background']) and not inherited.intersects(f,spec['floor'])
        truth[f['id']]=any(inherited.intersects(f,o) for o in f['objects'])
        assert f['objects'][0]['center_m'][0]+f['objects'][0]['size_m'][0]/2 < min(o['center_m'][0]-o['size_m'][0]/2 for o in f['objects'][1:])
    transitions=Counter()
    for pair in spec['pairs']:
        a,b=[[f for f in frames if f['episode']==e] for e in pair['episodes']]
        assert len(a)==len(b)==6
        for x,y in zip(a,b):
            assert truth[x['id']] != truth[y['id']]
            xx,yy=copy.deepcopy(x),copy.deepcopy(y)
            for row in (xx,yy):
                for key in ('id','episode','pair_member','pair_variant'):row.pop(key)
                row['objects'][0]['center_m'][1]=0.
            assert xx==yy
        for episode in (a,b):
            labels=[truth[f['id']] for f in episode]
            transitions['enter']+=sum(not x and y for x,y in zip(labels,labels[1:]))
            transitions['exit']+=sum(x and not y for x,y in zip(labels,labels[1:]))
    assert sum(truth.values())==144 and transitions=={'enter':6,'exit':6}
    for family in inherited.FAMILIES:
        assert sorted(g['wall_distance_m'] for g in groups if g['category']==family)==list(DISTANCES)
    # Source geometry only: compare deterministic prior signatures, never old captured outcomes.
    previous=inherited.SEED
    try:
        for seed in (136014,146016,158016,170016):
            inherited.SEED=seed
            prior=inherited.source()
            assert not {g['background_geometry_signature'] for g in groups}&{g['background_geometry_signature'] for g in prior['scene_groups']}
    finally:
        inherited.SEED=previous
    return dict(status='PASS',frames=288,scene_groups=24,episodes=48,positive_frames=144,negative_frames=144,
        wall_distances_m=list(DISTANCES),styles=dict(Counter(g['background_style'] for g in groups)),
        paired_only_lateral_intervention=True,all_context_in_native_truth=True,prior_geometry_overlap=0,
        transitions=dict(transitions),authority=AUTHORITY)


if __name__=='__main__':
    print(json.dumps(check_source(source()),indent=2))

