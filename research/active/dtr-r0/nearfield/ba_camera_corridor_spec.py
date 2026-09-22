"""Frozen four-pair posed-trajectory source; geometry metadata is evaluator-only."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MAP_SHA = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'


def specification():
    bronze = '/Game/StreetLab/Materials/Bronze'
    def target(y, size, z=1.47, kind='cube'):
        return dict(name='target', kind=kind, center_m=[30., y, z], size_m=list(size), material=bronze)
    variants = [
        ('g1_center', 'g1_thin_position', 'inside', target(0., (.04,.04,1.), kind='cylinder')),
        ('g1_right', 'g1_thin_position', 'outside', target(.40, (.04,.04,1.), kind='cylinder')),
        ('g2_intrusion', 'g2_horizontal_boundary', 'inside', target(.20, (.08,.60,.08), z=1.72)),
        ('g2_grazing', 'g2_horizontal_boundary', 'exact_contact_boundary', target(.60, (.08,.60,.08), z=1.72)),
        ('g3_large', 'g3_width', 'inside', target(0., (.06,.40,1.))),
        ('g3_thin', 'g3_width', 'inside', target(0., (.06,.04,1.))),
        ('g4_inside', 'g4_same_zone', 'inside', target(.26, (.04,.04,.14), z=1.65)),
        ('g4_outside', 'g4_same_zone', 'outside', target(.40, (.04,.04,.14), z=1.65)),
    ]
    background = dict(name='background', kind='cube', center_m=[34.,0.,2.12], size_m=[.15,4.,4.],
                      material='/Game/StreetLab/Materials/Limestone')
    clips, cases = [], []
    for clip, pair, relation, obj in variants:
        clips.append(dict(clip_id=clip, pair_id=pair, target_name='target', frames=12,
                          expected_lateral_relation=relation,
                          expected_lateral_extent_m=[obj['center_m'][1]-obj['size_m'][1]/2,
                                                     obj['center_m'][1]+obj['size_m'][1]/2]))
        for i in range(12):
            camera = dict(x=26.4+.1*i, y=0., z=1.82, pitch=0., yaw=0., roll=0.)
            cases.append(dict(name=f'{clip}-{i:02d}', clip_id=clip, pair_id=pair,
                frame_in_clip=i, time_s=round(.2*i, 6), camera=camera,
                objects=[dict(obj), dict(background)], target_name='target',
                expected_lateral_relation=relation,
                expected_front_z_m=obj['center_m'][0]-obj['size_m'][0]/2-camera['x']))
    return dict(schema='ba-camera-corridor-source-v1', expected_map_sha256=MAP_SHA,
        map='/Game/StreetLab/WillowSampleV1', frames=96, clips=clips, cases=cases,
        profile=dict(width=640, height=360, hfov_deg=100., camera_height_above_floor_m=1.7,
            expected_road_world_z_m=.12, camera_coordinate_frame='X_RIGHT_Y_DOWN_Z_OPTICAL_FORWARD',
            corridor_m=dict(x=[-.3,.3], y=[-.2,.9], z=[.3,3.]),
            intended_height_above_floor_m=[.8,1.9], native_geometry_tolerance_m=.002),
        sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORIES_NOT_REAL_TIME_DYNAMIC_CAPTURE',
        timeline='Twelve nominal samples per clip,0.2s apart; camera advances0.1m/sample; actual render times separately retained',
        source_authority='Declared objects are evaluator-only; authenticate live render bounds, mesh/material paths and target-directed native traces',
        strict_contact='Closed corridor contact counts as strict intersection; g2_grazing is separately declared exact-contact boundary, never clear outside',
        same_zone='G4 target footprint is designed for0-based(row4,col4) throughout; verify actual projection, record actual proxy returns, never force equal values',
        observation_boundary='RGB/public calibration are observations; native depth/poses/actor identities/spec/traces are evaluator-only. A separate sealed sensor-construction step may derive the unchanged64-single-return proxy from native depth.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to((ROOT/'artifacts.local').resolve()) and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(specification(), indent=2, allow_nan=False), encoding='utf-8')
    print('SPEC_WRITTEN', output)
