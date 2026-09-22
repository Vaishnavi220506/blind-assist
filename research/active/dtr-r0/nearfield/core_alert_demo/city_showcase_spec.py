"""Prespecified native City Sample illustration views; no labels or score selection."""
import copy
import json
import math
from city_showcase_capture import OUT, ROOT, PROJECT, sha, write


def freeze():
    art = ROOT / 'artifacts.local'
    dense = art / 'work/city-crossregion-worker-20260908/dense-terminal-v3/dense-source-scouts-v3'
    scouting = art / 'nearfield/city-crossregion-v1-20260908/scouting'
    rows = [
        ('plaza', dense / 'dense_candidate_01', [(2, '广场钟柱与长椅', 'approach_retreat'), (3, '广场侧向经过', 'lateral')]),
        ('boulevard', dense / 'dense_candidate_02', [(1, '玻璃楼下的广告灯箱', 'approach_retreat'), (4, '林荫路侧移', 'lateral')]),
        ('residential', dense / 'dense_candidate_07', [(2, '住宅花池与楼梯', 'lateral')]),
        ('big01', scouting / 'big_candidate_01-v4', [(2, '城市街角接近与退开', 'approach_retreat')]),
        ('big04', scouting / 'big_candidate_04-v4', [(3, '另一街区横向经过', 'lateral')]),
        ('smallcity', art / 'nearfield/city-native-view-probe-20260908/capture-v1', [(4, '原生路桩接近与退开', 'approach_retreat')])]
    specs = {}; clips = []; provenance = []
    for region, source, views in rows:
        base = json.loads((source / 'source/spec.json').read_text(encoding='utf-8-sig'))
        receipt = json.loads((source / 'receipt.json').read_text(encoding='utf-8'))
        spec = {k: copy.deepcopy(base[k]) for k in ('map_asset', 'map_sha256', 'world_partition_region_m',
            'native_full_detail_only', 'background_hlod_min_distance_m')}
        spec.update(map_file=str(PROJECT.parent / ('Content/' + base['map_asset'].removeprefix('/Game/') + '.umap')),
            schema='city-showcase-illustration-v1', scope='ENGINEERING_ILLUSTRATION_NOT_RESEARCH_COHORT',
            settling_ticks=32, first_use_settling_ticks=64, settling_interval_s=0.,
            pair_export_mode='native_probe', export_appearance=True, export_controlled_targets=True,
            export_native_inventory=True, inventory_indices=[0], profile_frames=True, cases=[])
        assert sha(spec['map_file']) == spec['map_sha256']
        for view_index, title, motion in views:
            original = base['cases'][view_index]
            floor = next(r for r in receipt['native_floor_probes'] if r['case'] == original['name'])
            assert floor['hit']
            clip_id = f'{region}_{view_index:02d}'
            camera = copy.deepcopy(original['camera'])
            camera['z'] = floor['point_m'][2] + 1.7
            clip = dict(clip_id=clip_id, title=title, description='原生城市材质和场景，固定短路径；仅作机制演示。',
                environment=base['map_asset'] + ' / ' + region, motion=motion, frame_count=24,
                preview_index=12, source_spec=str(source / 'source/spec.json'), source_case_index=view_index,
                source_camera=original['camera'], nominal_reference_hit_z_m=floor['point_m'][2],
                height_authority='1.7 m above saved first vertical collision hit; this is not a walkable-floor guarantee')
            clips.append(clip)
            for i in range(24):
                p = copy.deepcopy(camera); t = i / 23; yaw = math.radians(p['yaw'])
                # Never move forward beyond the accepted scout viewpoint.
                if motion == 'approach_retreat':
                    travel = -.8 * abs(2 * t - 1)
                    p['x'] += math.cos(yaw) * travel; p['y'] += math.sin(yaw) * travel
                else:
                    travel = .6 * (t - .5)
                    p['x'] -= math.sin(yaw) * travel; p['y'] += math.cos(yaw) * travel
                spec['cases'].append(dict(name=f'{clip_id}_f{i:04d}', clip_id=clip_id, frame_in_clip=i,
                    time_s=round(i * .2, 10), camera=p, objects=[], probe_native_floor=True))
        specs[region] = spec
        provenance.append(dict(region=region, source=str(source), spec_sha256=sha(source/'source/spec.json'),
            receipt_sha256=sha(source/'receipt.json')))
    smoke = copy.deepcopy(specs['plaza'])
    base = json.loads((rows[0][1] / 'source/spec.json').read_text(encoding='utf-8-sig'))
    smoke['cases'] = [dict(name='hd_smoke_d1_case2', camera=base['cases'][2]['camera'], objects=[], probe_native_floor=True)]
    specs['smoke'] = smoke
    for name, spec in specs.items():
        write(OUT / 'specs' / (name + '.json'), spec)
    write(OUT / 'spec.json', dict(authority='ILLUSTRATIVE_ONLY_NO_GT_OR_PERFORMANCE_METRICS',
        clips=clips, frames=192, regions=6, rgb_size=[1920,1080], native_depth_size=[640,360],
        horizontal_fov_degrees=100., period_s=.2, timing='PRESCRIBED_SETTLED_POSES_NOT_LIVE_ENGINE_MOTION',
        all_planned_clips_retained=True, sources=provenance))
    write(OUT / 'protocol.json', dict(id='ba-city-showcase-20260920',
        maximum_frames=192, smoke_frames=1, scientific_claims=False,
        spec_hashes={n:sha(OUT/'specs'/(n+'.json')) for n in specs},
        master_spec_sha256=sha(OUT/'spec.json'), builder_sha256=sha(__file__),
        policy='Frozen existing ToF proxy and CoreAlertPolicy; no outcome-selected scenes, thresholds, or algorithms',
        lifecycle='One process per region, isolated cache port, persistent CitySample cache preserved',
        acceptance='HD dimensions and common pose/FOV, intact maps, native resource health, visual QA; no invented alert target'))
    region_ids = [row[0] for row in rows]
    write(OUT / 'plan.json', dict(
        clips=[dict(clip, region_id=next(region for region in region_ids
                                        if clip['clip_id'].startswith(region + '_')))
               for clip in clips],
        regions=[dict(id=region, spec='specs/' + region + '.json', capture=region)
                 for region in region_ids]))
    print(json.dumps(dict(status='FROZEN', frames=192, regions=6, clips=8)))


if __name__ == '__main__':
    freeze()
