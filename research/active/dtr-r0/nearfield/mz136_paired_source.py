"""Prospective controlled scene-group paired source; no model/outcome access.

Geometry labels below are source-design checks, not native captured truth.
The single lateral intervention holds appearance, camera, and RNG seeds fixed.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random

SEED = 136014
DT = .25
STEPS = 6
FAMILIES = ('suspended_head', 'substantial_body', 'near_rod_farwall', 'shallow_boundary_stress')
ROOT = Path(__file__).resolve().parents[4]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def intersects(frame, obj):
    lo = [c-s/2-b for c,s,b in zip(obj['center_m'],obj['size_m'],frame['body_origin_m'])]
    hi = [c+s/2-b for c,s,b in zip(obj['center_m'],obj['size_m'],frame['body_origin_m'])]
    return hi[0] >= .2 and lo[0] <= 3.6 and hi[1] >= -.3 and lo[1] <= .3 and hi[2] >= .4 and lo[2] <= 2.05


def source():
    frames, pairs, groups, audits = [], [], [], []
    for fi,family in enumerate(FAMILIES):
        for k in range(6):
            group = f'mz136_{family}_scene{k}'
            split = 'train' if k < 4 else ('dev' if k == 4 else 'test')
            rng = random.Random(SEED+fi*997+k*37)
            # Each split contains all four distance strata; no farther-only test.
            front = (1.91,2.36,2.81,3.26)[(fi+k)%4]+rng.uniform(-.025,.025)
            if family == 'suspended_head':
                size = [rng.uniform(.12,.21),rng.uniform(.25,.44),rng.uniform(.13,.22)]
                z = rng.uniform(1.73,1.94)
            elif family == 'substantial_body':
                size = [rng.uniform(.21,.32),rng.uniform(.75,1.05),rng.uniform(1.83,2.54)]
                z = size[2]/2
            elif family == 'near_rod_farwall':
                size = [rng.uniform(.041,.071),rng.uniform(.022,.052),rng.uniform(2.10,2.31)]
                z = size[2]/2
            else:
                size = [rng.uniform(.12,.19),rng.uniform(.065,.125),rng.uniform(1.45,1.72)]
                z = rng.uniform(.94,1.13)
            rho = rng.uniform(.14,.85)
            noise_seed, texture_seed = rng.randrange(2**30), rng.randrange(2**30)
            sign = 1 if rng.random() < .5 else -1
            clearance = rng.uniform(.17,.23)
            span = rng.uniform(.018,.033)
            background = dict(name='env0',center_m=[rng.uniform(6.4,8.6),rng.uniform(-.25,.25),rng.uniform(1.6,2.2)],
                size_m=[rng.uniform(.10,.21),rng.uniform(9.0,12.0),rng.uniform(6.0,7.5)],
                texture_seed=rng.randrange(2**30),texture_grid=[7,6],tof_reflectance_proxy=rng.uniform(.35,.75),
                source_role='distant_scene_background')
            context = [background]
            if family == 'near_rod_farwall':
                context.append(dict(name='shape1',center_m=[rng.uniform(3.86,3.99),rng.uniform(-.05,.05),1.81],
                    size_m=[.08,rng.uniform(4.7,5.6),3.62],texture_seed=rng.randrange(2**30),texture_grid=[5,8],
                    tof_reflectance_proxy=rng.uniform(.45,.90),source_role='farwall_reference'))
            motion = dict(vx=rng.uniform(.015,.03),amplitude=rng.uniform(.009,.023),frequency=rng.uniform(.8,1.1),
                          yaw=rng.uniform(.35,.85),phase=rng.uniform(-.2,.2))
            members = ('enter','exit') if family == 'shallow_boundary_stress' else ('in','out')
            episodes = [group+'_'+member for member in members]
            groups.append(dict(scene_group=group,pair_id=group,split=split,category=family,episodes=episodes,
                background_signature=digest(background),background_geometry_signature=digest({key:background[key] for key in ('center_m','size_m')}),
                target_front_m=front,target_size_m=size,target_reflectance=rho))
            for member,episode in zip(members,episodes):
                truths = []
                for step in range(STEPS):
                    t = DT*step
                    cam = dict(x=.023+motion['vx']*t,y=motion['amplitude']*math.sin(motion['frequency']*t+motion['phase']),
                               z=1.7,yaw=motion['yaw']*math.sin(.71*t),pitch=-3.,roll=0.)
                    if family == 'shallow_boundary_stress':
                        ramp = span*(1-2*step/(STEPS-1))
                        side = sign*(.3+size[1]/2+(ramp if member=='enter' else -ramp))
                    else:
                        side = sign*(.06 if member=='in' else .3+size[1]/2+clearance)
                    target = dict(name='shape0',center_m=[front+size[0]/2+.023,cam['y']+side,z],size_m=list(size),
                        texture_seed=texture_seed,texture_grid=[5,8],tof_reflectance_proxy=rho,source_role=family)
                    frame = dict(id=f'{episode}_{step:02d}',episode=episode,family=family,category=family,
                        scene_group=group,pair_id=group,pair_member=member,pair_variant=member,split=split,
                        group='SHALLOW_BOUNDARY_STRESS' if family=='shallow_boundary_stress' else 'ORDINARY_MARGIN',
                        time_s=t,camera=cam,body_origin_m=[cam['x'],cam['y'],0.],objects=[target]+copy.deepcopy(context),
                        sensor_seed=noise_seed,tof_sensor_seed=noise_seed+1150003,
                        wearer_speed=math.hypot(motion['vx'],motion['amplitude']*motion['frequency']*math.cos(motion['frequency']*t+motion['phase'])),
                        radar_ghost=None)
                    truths.append(any(intersects(frame,obj) for obj in frame['objects']))
                    frames.append(frame)
                audits.append(dict(episode=episode,scene_group=group,split=split,pair_id=group,pair_member=member,
                    family=family,source_aabb_labels=truths))
            pairs.append(dict(pair_id=group,scene_group=group,split=split,category=family,episodes=episodes,
                intervention='TARGET_LATERAL_CENTERS_ONLY',shared_shape_texture_camera_sensor_seed=True))
    spec = dict(schema='mz136-scene-group-corridor-pairs-v1',seed=SEED,dt_s=DT,frames_per_episode=STEPS,
        authority='PROSPECTIVE_CONTROLLED_SIMULATION_DEVELOPMENT_NOT_PROTECTED_FINAL_OR_HARDWARE',
        rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,tof_rows=8,tof_columns=8,rgb_camera_count=1),
        background=dict(center_m=[14.3,0.,1.8],size_m=[.13,20.,9.],texture=False,tof_reflectance_proxy=.5),
        floor=dict(center_m=[4.2,0.,-.05],size_m=[26.,22.,.1],tof_reflectance_proxy=.3),
        corridor_m=dict(forward=[.2,3.6],lateral=[-.3,.3],height=[.4,2.05]),
        rng_authority='SAME_SEED_COMMON_RANDOM_START_NOT_IDENTICAL_DRAW_ALIGNMENT_CONDITIONAL_SENSOR_DRAWS',
        source_audit_authority='SOURCE_GEOMETRY_ONLY_NATIVE_BOUNDS_REQUIRED_FOR_TRAINING_AND_SCORING',
        scene_groups=groups,pairs=pairs,source_audit=audits,frames=frames,
        limitations=['Independent scene-group configurations share the renderer and sensor simulator.',
            'All real objects including context enter native truth; context is outside the query by geometry.',
            'Source pair IDs, split, family, latent geometry and seeds never enter predictor inputs.',
            'Pair sensor RNG starts match but geometry-dependent branches may consume different later draws.'])
    check_source(spec)
    return spec


def check_source(spec):
    frames=spec['frames']; groups=spec['scene_groups']; pairs=spec['pairs']
    assert len(frames)==288 and len(groups)==len(pairs)==24
    assert len({f['id'] for f in frames})==288
    assert Counter(f['split'] for f in frames)==dict(train=192,dev=48,test=48)
    assert len({g['background_geometry_signature'] for g in groups})==24
    assert len({g['background_signature'] for g in groups})==24
    by_episode={p:[] for pair in pairs for p in pair['episodes']}
    assert len(by_episode)==48
    memberships={}
    truth={}
    for f in frames:
        by_episode[f['episode']].append(f)
        memberships.setdefault(f['scene_group'],set()).add(f['split'])
        assert len(f['objects'])>=2 and f['objects'][0]['name']=='shape0'
        assert all(not intersects(f,o) for o in f['objects'][1:])
        assert not intersects(f,spec['background']) and not intersects(f,spec['floor'])
        target=f['objects'][0]; background=f['objects'][1]
        assert background['center_m'][0]-background['size_m'][0]/2 > target['center_m'][0]+target['size_m'][0]/2
        gt=any(intersects(f,o) for o in f['objects']);truth[f['id']]=gt
        if f['pair_member'] in ('in','out'):assert gt==(f['pair_member']=='in')
    assert all(len(splits)==1 for splits in memberships.values())
    changed=0; transitions=Counter()
    for pair in pairs:
        a,b=[by_episode[e] for e in pair['episodes']]
        assert len(a)==len(b)==STEPS
        for x,y in zip(a,b):
            assert truth[x['id']] != truth[y['id']]
            assert x['objects'][0]['center_m'][1] != y['objects'][0]['center_m'][1]
            xx,yy=copy.deepcopy(x),copy.deepcopy(y)
            for row in (xx,yy):
                for key in ('id','episode','pair_member','pair_variant'):row.pop(key)
                row['objects'][0]['center_m'][1]=0.
            assert xx==yy, 'Pair differs beyond target lateral center and identity annotations'
            changed+=1
        for episode in (a,b):
            labels=[truth[f['id']] for f in episode]
            transitions['enter']+=sum(not x and y for x,y in zip(labels,labels[1:]))
            transitions['exit']+=sum(x and not y for x,y in zip(labels,labels[1:]))
            if pair['category']=='shallow_boundary_stress':assert sum(x!=y for x,y in zip(labels,labels[1:]))==1
    for family in FAMILIES:
        assert Counter(g['split'] for g in groups if g['category']==family)==dict(train=4,dev=1,test=1)
    assert transitions==dict(enter=6,exit=6)
    return dict(status='PASS',frames=288,episodes=48,scene_groups=24,frame_pairs=changed,
        split_frames=dict(Counter(f['split'] for f in frames)),positive_frames=sum(truth.values()),
        negative_frames=len(truth)-sum(truth.values()),transitions=dict(transitions),
        group_overlap=0,paired_only_lateral_intervention=True,unique_background_geometries=24,
        all_context_in_native_object_list=True,source_truth_authority=spec['source_audit_authority'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()) or out.exists():raise ValueError('Fresh canonical artifact source directory required')
    spec=source();out.mkdir(parents=True)
    for name,obj in (('spec.json',spec),('source-audit.json',check_source(spec))):
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    freeze=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('spec.json','source-audit.json')},
        authority=spec['authority'],source_selection='FIXED_BEFORE_MODEL_FITS_AND_CAPTURE_OUTCOMES')
    (out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),**check_source(spec)),indent=2))


if __name__=='__main__':main()
