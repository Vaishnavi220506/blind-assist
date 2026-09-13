"""Fixed 20-episode finite-footprint ToF Development source, never predictions.

All 240 frames are retained. Explicit optical-reflectance proxies are latent
sensor-simulation parameters, independent of RGB albedo texture seeds. Static
far walls remain real target-list actors so native truth cannot silently omit a
wall that enters the corridor. Source AABB audits are not native measurements.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

SEED = 115013
TEXTURE_SEED = 115917
DT = .25
STEPS = 12
WEARER_SPEED_MPS = .025
FAMILY_EPISODES = dict(thin_rod_farwall=4, substantial_offroute=4, suspended_head=4,
    multitarget_competing_depth=4, clear_ghost=2, boundary_1cm_stress=2)


def object_definition(start, size, reflectance, velocity=(0.,0.,0.), role='obstacle'):
    return dict(start_m=list(start), size_m=list(size), velocity_mps=list(velocity),
        tof_reflectance_proxy=reflectance, source_role=role)


def intersects(frame, obj):
    lo=[c-s/2-b for c,s,b in zip(obj['center_m'],obj['size_m'],frame['body_origin_m'])]
    hi=[c+s/2-b for c,s,b in zip(obj['center_m'],obj['size_m'],frame['body_origin_m'])]
    return hi[0]>=.2 and lo[0]<=3.6 and hi[1]>=-.3 and lo[1]<=.3 and hi[2]>=.4 and lo[2]<=2.05


def source(texture_seed=TEXTURE_SEED):
    sensor_rng=random.Random(SEED);texture_rng=random.Random(texture_seed);scenes=[]

    def add(name,family,objects,yaw=0.,ghost=None):
        scenes.append(dict(episode='mz115_'+name,family=family,objects=objects,yaw_amplitude=yaw,ghost=ghost))

    # The wall front stays at 3.71125--3.78m relative to the wearer, outside
    # the 3.6m alert boundary. Central footprint rays can hit it below 4m;
    # more oblique rays may exceed the trace limit, which remains missing data.
    for k,(forward,lateral,width,rho,wall_rho) in enumerate(((2.47,.015,.026,.06,.78),
            (2.81,-.09,.042,.20,.55),(3.07,.13,.064,.72,.18),(2.63,-.18,.086,.42,.92))):
        add(f'rod_farwall_{k}','thin_rod_farwall',[
            object_definition((forward,lateral,1.12),(.058+.006*k,width,2.24),rho,role='thin_rod'),
            object_definition((3.82,0.,1.8),(.08,5.4,3.6),wall_rho,role='farwall_reference'),
        ],yaw=(0.,3.,-3.,0.)[k])

    # Substantial true-negative obstacles: inner lateral edges .45/.52/.47/.60m.
    for k,(forward,side,width,height,rho) in enumerate(((2.61,.93,.96,2.54,.18),
            (3.09,-1.12,1.20,2.70,.84),(2.83,.84,.74,1.36,.58),(3.21,-1.05,.90,1.58,.09))):
        add(f'offroute_{k}','substantial_offroute',[
            object_definition((forward,side,height/2),(.28+.02*k,width,height),rho,role='substantial_offroute'),
        ],yaw=(4.,-4.,0.,3.)[k],ghost=dict(x=.065,z=2.93) if k==3 else None)

    add('head_static_0','suspended_head',[
        object_definition((2.37,.04,1.84),(.17,.35,.19),.10,role='suspended_head'),
    ],yaw=3.)
    add('head_static_1','suspended_head',[
        object_definition((2.96,-.07,1.92),(.21,.50,.18),.76,role='suspended_head'),
    ],yaw=-3.)
    add('head_cross_0','suspended_head',[
        object_definition((2.71,.85,1.80),(.16,.40,.22),.32,(0.,-.64,0.),'suspended_head'),
    ],yaw=4.)
    add('head_cross_1','suspended_head',[
        object_definition((3.15,-.95,1.88),(.20,.56,.24),.88,(0.,.70,0.),'suspended_head'),
    ],yaw=-4.)

    # Near/far physical surfaces occupy overlapping horizontal view regions.
    # Reflectance deliberately changes ordering independently of visual texture.
    add('multi_weak_rod_bright_body','multitarget_competing_depth',[
        object_definition((2.41,-.02,1.12),(.06,.034,2.24),.07,role='near_weak_rod'),
        object_definition((3.17,.06,1.22),(.19,.42,2.44),.91,role='far_bright_body'),
    ])
    add('multi_bright_rod_weak_head','multitarget_competing_depth',[
        object_definition((2.24,.045,1.12),(.07,.052,2.24),.89,role='near_bright_rod'),
        object_definition((3.23,.12,1.86),(.20,.38,.24),.08,role='far_weak_head'),
    ],yaw=3.)
    add('multi_partial_occlusion','multitarget_competing_depth',[
        object_definition((2.07,-.15,1.0),(.24,.44,1.60),.82,(0.,.18,0.),'moving_near_barrier'),
        object_definition((3.11,.02,1.90),(.18,.45,.28),.13,role='partially_occluded_head'),
    ],yaw=-3.,ghost=dict(x=-.075,z=2.78))
    add('multi_weak_head_bright_wall','multitarget_competing_depth',[
        object_definition((2.49,-.05,1.90),(.14,.35,.12),.055,role='near_weak_head'),
        object_definition((3.31,.10,1.30),(.18,.50,2.60),.85,role='far_nearfield_wall'),
    ],yaw=2.)

    add('clear_reference','clear_ghost',[])
    add('clear_persistent_ghost','clear_ghost',[],yaw=3.,ghost=dict(x=.055,z=2.73))
    add('stress_enter_1cm','boundary_1cm_stress',[
        object_definition((2.91,.35,1.30),(.16,.08,1.44),.48,(0.,-.02/(11*DT),0.),'lateral_1cm_stress'),
    ])
    add('stress_exit_1cm','boundary_1cm_stress',[
        object_definition((3.07,-.33,1.32),(.18,.08,1.48),.48,(0.,-.02/(11*DT),0.),'lateral_1cm_stress'),
    ])

    frames=[];audits=[];counts=Counter();transitions=Counter();family_counts=Counter()
    for scene in scenes:
        sensor_seed=sensor_rng.randrange(2**30)
        definitions=[dict(obj,name=f'shape{k}',texture_seed=texture_rng.randrange(2**30)) for k,obj in enumerate(scene['objects'])]
        labels=[]
        for step in range(STEPS):
            t=step*DT;x=WEARER_SPEED_MPS*t
            objects=[dict(name=o['name'],center_m=[c+v*t for c,v in zip(o['start_m'],o['velocity_mps'])],
                size_m=list(o['size_m']),texture_seed=o['texture_seed'],texture_grid=[3,6],
                tof_reflectance_proxy=o['tof_reflectance_proxy'],source_role=o['source_role']) for o in definitions]
            frame=dict(id=f"{scene['episode']}_{step:02d}",episode=scene['episode'],family=scene['family'],time_s=t,
                camera=dict(x=x,y=0.,z=1.7,yaw=scene['yaw_amplitude']*math.sin(.83*t),pitch=-3.,roll=0.),
                body_origin_m=[x,0.,0.],objects=objects,sensor_seed=sensor_seed,
                wearer_speed=WEARER_SPEED_MPS,radar_ghost=scene['ghost'])
            positive=any(intersects(frame,obj) for obj in objects);labels.append(positive)
            counts['positive' if positive else 'negative']+=1;family_counts[(scene['family'],positive)]+=1;frames.append(frame)
        enters=sum(not a and b for a,b in zip(labels,labels[1:]));exits=sum(a and not b for a,b in zip(labels,labels[1:]))
        transitions.update(enter=enters,exit=exits)
        audits.append(dict(episode=scene['episode'],family=scene['family'],objects=definitions,
            camera_velocity_mps=[WEARER_SPEED_MPS,0.,0.],source_aabb_labels=labels,enters=enters,exits=exits))
    spec=dict(schema='mz115-finite-footprint-tof-single-rgb-source-v1',seed=SEED,texture_seed=texture_seed,
        authority='CONTROLLED_UE_RGB_NATIVE_COLLISION_FINITE_FOOTPRINT_TOF_PROXY_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',
        rig=dict(width=640,height=360,hfov_deg=70.,tof_hfov_deg=45.,tof_rows=8,tof_columns=8,rgb_camera_count=1),
        background=dict(center_m=[12.7,0.,1.75],size_m=[.14,18.6,8.5],texture=False,tof_reflectance_proxy=.50),
        floor=dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1],tof_reflectance_proxy=.30),
        fixed_before_capture=True,dt_s=DT,frames_per_episode=STEPS,family_episode_counts=FAMILY_EPISODES,
        source_design_aabb_frame_counts=dict(counts),source_design_transition_counts=dict(transitions),
        family_source_aabb_counts={family:dict(positive=family_counts[(family,True)],negative=family_counts[(family,False)]) for family in FAMILY_EPISODES},
        source_audit_authority='SOURCE_DESIGN_ONLY_NOT_NATIVE_ENGINE_MEASUREMENTS_OR_OBSERVABLE_LABELS',source_audit=audits,
        initial_body_origin_m=[0.,0.,0.],corridor_m=dict(forward=[.2,3.6],lateral=[-.3,.3],height=[.4,2.05]),
        reflectance_authority='EXPLICIT_LATENT_SENSOR_PROXY_INDEPENDENT_OF_RGB_ALBEDO_TEXTURE',
        limitations=[
            'All object-list walls, including nearfield walls and farwall references, participate in current-frame obstacle truth.',
            'Global background and floor are context omitted from target truth only because their AABBs provably miss the current corridor in every frame.',
            'Finite 3x3 subrays are sparse quadrature, not a continuous beam or physical photon/RF simulator. Thin weak rods may produce no positive signal.',
            'Near/far surfaces provide mixed-return opportunities, not guaranteed detections; oblique background-wall rays may exceed the 4m trace limit.',
            'Reflectance values are independent latent choices, not measured material properties or inferred RGB brightness.',
            'Camera advances only 6.875cm per episode; yaw is observation rotation, not future walking intention.',
            'Analytic source-AABB counts must be checked against actual engine bounds after capture; no labels may be silently selected or removed.',
            'All 20 episodes and 240 frames remain constructed Development; no natural-domain, hardware, safety or deployment claims.'],frames=frames)
    check_source(spec)
    return spec


def check_source(spec):
    frames=spec['frames'];assert len(frames)==240 and len({f['id'] for f in frames})==240
    episodes={f['episode'] for f in frames};assert len(episodes)==20
    assert Counter(f['episode'] for f in frames)=={e:12 for e in episodes}
    assert Counter(a['family'] for a in spec['source_audit'])==FAMILY_EPISODES
    assert spec['source_design_aabb_frame_counts']==dict(positive=144,negative=96)
    assert spec['source_design_transition_counts']==dict(enter=3,exit=3)
    assert len({f['sensor_seed'] for f in frames})==20
    for frame in frames:
        assert frame['camera']['pitch']==-3. and frame['camera']['roll']==0.
        assert 0<=frame['camera']['x']<=.1 and frame['body_origin_m']==[frame['camera']['x'],0.,0.]
        assert all(0<o['tof_reflectance_proxy']<=1 for o in frame['objects'])
        for context in (spec['background'],spec['floor']):assert not intersects(frame,context)
        for obj in frame['objects']:
            if obj['source_role']=='farwall_reference':
                relative_front=obj['center_m'][0]-obj['size_m'][0]/2-frame['body_origin_m'][0]
                assert 3.6<relative_front<4. and not intersects(frame,obj)
            if obj['source_role']=='substantial_offroute':
                assert abs(obj['center_m'][1])-obj['size_m'][1]/2>.4 and not intersects(frame,obj)
    for audit in spec['source_audit']:
        rows=[f for f in frames if f['episode']==audit['episode']]
        assert [r['time_s'] for r in rows]==[i*DT for i in range(STEPS)]
        assert all([o['name'] for o in r['objects']]==[o['name'] for o in audit['objects']] for r in rows)
        assert audit['source_aabb_labels']==[any(intersects(r,o) for o in r['objects']) for r in rows]
        if audit['family'] in ('substantial_offroute','clear_ghost'):assert not any(audit['source_aabb_labels'])
        if audit['family']=='boundary_1cm_stress':assert audit['enters']+audit['exits']==1
    return dict(status='PASS',frames=240,episodes=20,positive=144,negative=96,enters=3,exits=3)


def self_test():
    original=source();changed_texture=source(TEXTURE_SEED+1)
    assert any(a.get('texture_seed')!=b.get('texture_seed') for fa,fb in zip(original['frames'],changed_texture['frames']) for a,b in zip(fa['objects'],fb['objects']))
    for a,b in zip(original['frames'],changed_texture['frames']):
        assert a['sensor_seed']==b['sensor_seed'] and a['camera']==b['camera']
        assert [(o['center_m'],o['size_m'],o['tof_reflectance_proxy']) for o in a['objects']]==[(o['center_m'],o['size_m'],o['tof_reflectance_proxy']) for o in b['objects']]
    assert json.dumps(original,sort_keys=True)==json.dumps(source(),sort_keys=True)
    return dict(check_source(original),texture_rng_independence=True,deterministic_source=True,
        family_source_aabb_counts=original['family_source_aabb_counts'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--capture-source',type=Path,action='append',default=[],help='Explicit ready capture/sensor/helper files to hash into freeze receipt')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2));return
    if args.output is None or not args.capture_source:parser.error('--output and at least one --capture-source are required for a coupled source freeze')
    root=Path(__file__).resolve().parents[4];output=args.output.resolve();freeze=output.with_name('freeze.json')
    if not output.is_relative_to((root/'artifacts.local').resolve()) or output==freeze or output.exists() or freeze.exists():
        raise ValueError('Fresh canonical source/freeze files required')
    helpers=[p.resolve() for p in args.capture_source]
    if not all(p.is_file() for p in helpers):raise ValueError('Capture helpers must be ready before source freeze')
    spec=source();payload=(json.dumps(spec,indent=2,allow_nan=False)+'\n').encode('utf-8')
    receipt=dict(spec_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers},frames=240,episodes=20,
        source_design_aabb_frame_counts=spec['source_design_aabb_frame_counts'],source_design_transition_counts=spec['source_design_transition_counts'],
        selection='FIXED_SEED115013_ALL20_EPISODES_ALL240_FRAMES_NO_OUTCOME_SELECTION')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:handle.write(payload)
    with freeze.open('x',encoding='utf-8') as handle:handle.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
