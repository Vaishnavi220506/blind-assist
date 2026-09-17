"""Background-counterfactual source designs; no model or captured outcomes."""
import copy
from collections import Counter,defaultdict
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import mz136_paired_source as inherited

SEEDS={'train':180017,'report':181017}
DISTANCES={'train':((3.3,3.8,4.5),(3.6,4.2,4.5)),
           'report':((3.3,4.2),(3.6,4.5),(3.8,4.5))}


def original(seed):
    previous=inherited.SEED
    try:
        inherited.SEED=seed
        return inherited.source()
    finally:inherited.SEED=previous


def background(seed,config,variant,distance):
    """Background RNG/material identity shared across families at each level."""
    rng=random.Random(seed+config*1009+variant*107)
    style=('door' if config%2==0 else 'staggered') if distance<=3.6 else ('segmented' if variant%2==0 else 'full_wall')
    x=distance+.023;thickness=rng.uniform(.09,.15);rho=rng.uniform(.3,.85)
    texture_seed=seed+config*47  # same coarse albedo seed across interventions
    def panel(name,front,side,width,height=3.8,z=1.9):
        return dict(name=name,center_m=[front+thickness/2,side,z],size_m=[thickness,width,height],
            texture_grid=[3,3],texture_seed=texture_seed,tof_reflectance_proxy=rho,source_role='counterfactual_background')
    if style=='full_wall':objects=[panel('context0',x,0.,5.2)]
    else:
        gap=.47;width=2.3
        objects=[panel('context0',x,-gap-width/2,width),
                 panel('context1',x+(.22 if style=='staggered' else 0.),gap+width/2,width)]
        if style=='door':objects.append(panel('context2',x,0.,2*gap,.4,2.7))
        if style=='segmented':objects.append(panel('context2',x+.7,0.,.7,3.8,1.9))
    objects.append(panel('rear_scene',x+2.4+rng.uniform(.1,.4),0.,9.,6.,3.))
    return objects,style


def source(mode='train'):
    if mode not in SEEDS:raise ValueError('mode must be train or report')
    seed=SEEDS[mode];base=original(seed);spec=copy.deepcopy(base)
    frames=[];groups=[];pairs=[];audits=[]
    prefix='intrusion_'+mode
    for fi,family in enumerate(inherited.FAMILIES):
        for config,distances in enumerate(DISTANCES[mode]):
            target_group=f'{prefix}_{family}_target{config}'
            split=('train' if config==0 else 'dev') if mode=='train' else 'confirmation'
            # New seed provides independent target/material/camera configurations.
            template_group=base['scene_groups'][fi*6+config]
            templates=[copy.deepcopy(f) for f in base['frames'] if f['scene_group']==template_group['scene_group']]
            # One target adjustment shared by every counterfactual background.
            for f in templates:
                obj=f['objects'][0];obj['texture_grid']=[3,3]
                obj['center_m'][0]=min(obj['center_m'][0],min(distances)+.023-.45-obj['size_m'][0]/2)
            target_signature=inherited.digest([dict(camera=f['camera'],body_origin_m=f['body_origin_m'],
                target=f['objects'][0],sensor_seed=f['sensor_seed'],tof_sensor_seed=f['tof_sensor_seed']) for f in templates])
            for variant,distance in enumerate(distances):
                context,style=background(seed,config,variant,distance)
                group=f'{target_group}_bg{variant}'
                members=list(dict.fromkeys(f['pair_member'] for f in templates))
                episodes=[group+'_'+member for member in members]
                metadata=dict(target_group=target_group,counterfactual_group=target_group,background_variant=variant,
                    wall_distance_m=distance,background_style=style,target_signature=target_signature,
                    target_reflectance=templates[0]['objects'][0]['tof_reflectance_proxy'],texture_grid=[3,3])
                groups.append(dict(scene_group=group,pair_id=group,split=split,category=family,episodes=episodes,
                    background_signature=inherited.digest(context),background_geometry_signature=inherited.digest([
                        {k:o[k] for k in ('center_m','size_m')} for o in context]),**metadata))
                for f0 in templates:
                    f=copy.deepcopy(f0);member=f['pair_member'];episode=group+'_'+member
                    step=round(f['time_s']/base['dt_s'])
                    f.update(id=f'{episode}_{step:02d}',episode=episode,scene_group=group,pair_id=group,split=split,**metadata)
                    f['objects']=[f['objects'][0]]+copy.deepcopy(context);frames.append(f)
                pairs.append(dict(pair_id=group,scene_group=group,split=split,category=family,episodes=episodes,
                    target_group=target_group,counterfactual_group=target_group,intervention='TARGET_LATERAL_CENTERS_ONLY',
                    shared_shape_texture_camera_sensor_seed=True))
                for member,episode in zip(members,episodes):
                    rr=[f for f in frames if f['episode']==episode]
                    audits.append(dict(episode=episode,scene_group=group,split=split,pair_id=group,pair_member=member,
                        family=family,source_aabb_labels=[any(inherited.intersects(f,o) for o in f['objects']) for f in rr]))
    spec.update(schema='intrusion-background-counterfactual-v1',seed=seed,mode=mode,
        authority='SOURCE_DESIGN_ONLY_NATIVE_ADMISSION_REQUIRED_CONTROLLED_SIMULATION',
        frames=frames,scene_groups=groups,pairs=pairs,source_audit=audits,
        source_selection='PREDECLARED_SEEDS_GEOMETRY_ONLY_NO_MODEL_OR_OUTCOME_ACCESS',
        counterfactual_definition='SAME_TARGET_CAMERA_MATERIAL_SENSOR_SEED_ONLY_CONTEXT_OBJECTS_CHANGE',
        split_unit='target_group: all lateral and background variants kept together')
    spec['limitations'] += [
        'Common random starts are shared; sensor draw alignment can change with background geometry.',
        'Target/camera groups, not frames or background variants, are independent sampling units.',
        'Coarse 3x3 procedural grayscale tiles and reflectance proxies are not physical materials.',
        'Near background openings preserve negative all-object truth; topology and distance are not independently factorial.',
        'Public ToF model has 4m finite range; physical far wall distance is not observable q100.',
        'TRAIN/DEV source and separate report source remain same-generator controlled simulation.']
    check_source(spec,mode)
    return spec


def identity_without_context(f):
    return {k:copy.deepcopy(f[k]) for k in ('time_s','camera','body_origin_m','sensor_seed','tof_sensor_seed',
        'wearer_speed','radar_ghost','pair_member')}|dict(target=copy.deepcopy(f['objects'][0]))


def actor_count(frame):
    # Textured objects: one native cube plus grid tiles; floor/background and
    # camera actors add a small fixed overhead outside this frame-owned count.
    return sum(1+(o['texture_grid'][0]*o['texture_grid'][1] if o.get('texture',True) else 0)
               for o in frame['objects'])


def check_source(spec,mode=None):
    mode=spec['mode'] if mode is None else mode
    assert mode in SEEDS and spec['mode']==mode and spec['seed']==SEEDS[mode]
    fs=spec['frames'];gs=spec['scene_groups'];assert len(fs)==288 and len(gs)==len(spec['pairs'])==24
    assert len({f['id'] for f in fs})==288 and len({f['episode'] for f in fs})==48
    assert Counter(f['split'] for f in fs)==({'train':144,'dev':144} if mode=='train' else {'confirmation':288})
    targets=defaultdict(list);truth={};episodes=defaultdict(list)
    for f in fs:
        targets[f['target_group']].append(f);episodes[f['episode']].append(f)
        assert all(not inherited.intersects(f,o) for o in f['objects'][1:])
        assert not inherited.intersects(f,spec['background']) and not inherited.intersects(f,spec['floor'])
        truth[f['id']]=any(inherited.intersects(f,o) for o in f['objects'])
        assert f['objects'][0]['center_m'][0]+f['objects'][0]['size_m'][0]/2<min(o['center_m'][0]-o['size_m'][0]/2 for o in f['objects'][1:])
    assert sum(truth.values())==144
    for pair in spec['pairs']:
        aa,bb=[episodes[e] for e in pair['episodes']];assert len(aa)==len(bb)==6
        for a,b in zip(aa,bb):
            assert truth[a['id']]!=truth[b['id']]
            x,y=copy.deepcopy(a),copy.deepcopy(b)
            for row in (x,y):
                for key in ('id','episode','pair_member','pair_variant'):row.pop(key)
                row['objects'][0]['center_m'][1]=0.
            assert x==y,'Lateral pair contains another intervention'
    for target,rr in targets.items():
        assert len({f['split'] for f in rr})==1
        groups=defaultdict(list)
        for f in rr:groups[(f['pair_member'],f['time_s'])].append(f)
        for same in groups.values():
            assert len(same)==(3 if mode=='train' else 2)
            assert all(identity_without_context(f)==identity_without_context(same[0]) for f in same)
            assert len({inherited.digest(f['objects'][1:]) for f in same})==len(same)
    # Every family has the exact same background settings and balanced labels.
    reference=None
    for family in inherited.FAMILIES:
        distribution=Counter((f['split'],f['wall_distance_m'],f['background_style'],truth[f['id']]) for f in fs if f['family']==family)
        if reference is None:reference=distribution
        assert distribution==reference
    budget=[actor_count(f) for f in fs];assert max(budget)<=60
    return dict(status='PASS',mode=mode,frames=288,target_groups=len(targets),scene_groups=24,
        split_frames=dict(Counter(f['split'] for f in fs)),positive_frames=144,negative_frames=144,
        frame_pairs=144,background_counterfactual_positions=sum(len(v)//(3 if mode=='train' else 2) for v in targets.values()),
        lateral_pair_only=True,background_only_counterfactuals=True,family_label_background_balanced=True,
        mean_frame_owned_actors=sum(budget)/len(budget),max_frame_owned_actors=max(budget),
        native_truth_status='SOURCE_ONLY_PENDING_NATIVE_CAPTURE_ADMISSION')


if __name__=='__main__':
    import json
    print(json.dumps({mode:check_source(source(mode)) for mode in SEEDS},indent=2))
