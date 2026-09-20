"""Export native scene illustrations without adding evaluation labels or metrics.

Rendering data is new; geometry, sensor proxy and alert policy stay frozen.
All planned clips are retained irrespective of their alert response.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

from policy import CoreAlertPolicy, NEARFIELD
from ba_camera_corridor import sample_native
from tof_fov45_core import boxes45, simulate

ROOT = NEARFIELD.parents[3]
OUT = ROOT / 'artifacts.local/work/ba-city-showcase-20260920'
SITE = ROOT / 'artifacts.local/work/ba-core-alert-demo-20260920/site'
MOTIONS = dict(approach_retreat='接近 → 退开', lateral_pass='从左向右经过',
    reverse_view_approach_retreat='反向观察 · 接近 → 退开',
    approach_lateral_avoidance='接近 → 向侧方绕行',
    lateral_occlusion_reveal='横向移动 · 遮挡关系变化', lateral='侧向观察 · 区域支持变化')
ENVIRONMENTS = dict(plaza='City Sample · 石铺广场', boulevard='City Sample · 广告灯箱街角',
    residential='City Sample · 住宅花池与台阶', big01='City Sample · 玻璃楼步道',
    big04='City Sample · 临街广告亭', smallcity='City Sample · 路桩人行道')
DESCRIPTIONS = {
    'plaza_02': '沿金属长椅接近再退开，对照前景距离、区域支持和固定强证据线。',
    'plaza_03': '侧向经过钟柱与垃圾桶，观察相邻区域如何接收不同表面的返回。',
    'boulevard_01': '广告灯箱、树木与玻璃立面同框，逐帧查看近处返回与城市背景的区别。',
    'boulevard_04': '沿林荫街角横移，观察树干、路侧设施与名义 ToF 视场的相对位置。',
    'residential_02': '住宅花池、台阶与栏杆构成不同高度的表面，对照测距区间和侧视几何。',
    'big01_02': '在玻璃楼旁的步道接近再退开，观察画面变化与 64 区观测是否同步。',
    'big04_03': '横向经过临街广告亭与盆栽，查看最大分数区域怎样随视角变化。',
    'smallcity_04': '在小城市路桩人行道接近再退开，对照窄物体、路边绿植与通道支持。',
}
TITLES = {'big01_02': '玻璃楼步道接近与退开', 'big04_03': '滨水街角广告亭与盆栽'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = read(OUT / 'spec.json')
    protocol = read(OUT / 'protocol.json')
    assert sha(OUT/'spec.json') == protocol['master_spec_sha256']
    baseline = read(SITE.parent/'build-receipt.json')
    assert sha(SITE/'demo-data.js') == baseline['export_sha256']
    assert sha(Path(__file__).with_name('policy.py')) == baseline['code_sha256']['policy.py']
    for name in ('ba_camera_corridor.py', 'tof_fov45_core.py', 'tof_corridor_calibration.py'):
        relative = str((NEARFIELD/name).relative_to(ROOT))
        assert sha(NEARFIELD/name) == baseline['source_seals_and_payload_hashes'][relative]
    plan = read(OUT/'plan.json')
    metadata = {c['clip_id']: c for c in plan['clips']}
    rows, capture_receipts = [], {}
    for region in plan['regions']:
        cap = OUT/region['capture']
        hd = read(cap/'hd-source-receipt.json')
        assert hd['status']=='PASS' and hd['source_unchanged'] and hd['task_processes_released']
        for name, key in [('receipt.json','receipt_sha256'),('process-release.json','release_sha256'),
                          ('render-resource-health.json','health_sha256')]:
            assert sha(cap/name)==hd[key]
        receipt=read(cap/'receipt.json')
        region_spec=read(cap/'source/spec.json')
        assert receipt['spec_sha256']==sha(cap/'source/spec.json')==sha(OUT/region['spec'])
        assert sha(OUT/region['spec']) == protocol['spec_hashes'][region['id']]
        assert len(hd['frames'])==receipt['frame_count']==len(region_spec['cases'])
        for i,(row,case) in enumerate(zip(hd['frames'],region_spec['cases'])):
            pose=receipt['capture_poses'][i]
            assert pose['rgb']==pose['depth']==pose['beauty']
            assert pose['dimensions']==dict(rgb=[640,360],depth=[640,360],beauty=[1920,1080])
            assert all(v==100 for v in pose['hfov_degrees'].values())
            assert row['id']==case['name'] and row['clip_id']==case['clip_id']
            assert row['frame_in_clip']==case['frame_in_clip'] and row['time_s']==case['time_s']
        rows.extend(hd['frames'])
        capture_receipts[region['id']]=sha(cap/'hd-source-receipt.json')
    assert len(rows)==sum(c['frame_count'] for c in metadata.values())==192
    clips, hashes, source_hashes, observations = OrderedDict(), {}, {}, []
    assets = SITE / 'assets/showcase'
    assets.mkdir(parents=True, exist_ok=True)
    (OUT / 'observations').mkdir(exist_ok=True)
    policy = CoreAlertPolicy()
    for i, row in enumerate(rows):
        assert abs(row['time_s'] - row['frame_in_clip'] * .2) < 1e-6
        native_path = OUT / row['native_path']
        rgb_path = OUT / row['rgb_path']
        assert sha(native_path) == row['native_sha256']
        assert sha(rgb_path) == row['rgb_sha256']
        with Image.open(rgb_path) as rgb_image:
            assert rgb_image.size==(1920,1080)
        for p in (native_path, rgb_path):
            source_hashes[p.relative_to(OUT).as_posix()] = sha(p)
        native = np.load(native_path, allow_pickle=False)
        assert native.shape == (360, 640)
        frame_id = f's{i:04d}'
        identity = 'city-showcase-v1/' + row['id']
        values, _ = simulate(sample_native(native), identity, boxes45())
        output = policy.step(boxes=boxes45(), values=values, clip_id=row['clip_id'],
                             frame_id=frame_id, time_s=row['time_s'])
        observation_path = OUT / 'observations' / (frame_id + '.npz')
        np.savez_compressed(observation_path, boxes=boxes45(), values=values)
        observations.append(dict(id=frame_id, clip_id=row['clip_id'], time_s=row['time_s'],
                                 path=observation_path.relative_to(OUT).as_posix(),
                                 sha256=sha(observation_path), identity=identity))
        destination = assets / (frame_id + '.png')
        shutil.copyfile(rgb_path, destination)
        hashes[destination.relative_to(SITE).as_posix()] = sha(destination)
        assert sha(destination) == row['rgb_sha256']
        meta = metadata[row['clip_id']]
        if row['clip_id'] not in clips:
            clips[row['clip_id']] = dict(id=row['clip_id'], label=TITLES.get(row['clip_id'],meta['title']),
                type=meta['title'], layer='DEMO', layout='DEMO',
                background=ENVIRONMENTS[meta['region_id']],
                environment=ENVIRONMENTS[meta['region_id']],
                preview_index=meta.get('preview_index',12),
                description=DESCRIPTIONS[meta['clip_id']], motion=MOTIONS.get(meta.get('motion'), meta.get('motion', '')), frames=[])
        clips[row['clip_id']]['frames'].append(dict(id=frame_id, time_s=row['time_s'],
            rgb=destination.relative_to(SITE).as_posix(),
            rgb_size=[1920,1080],
            values=[float(v) if np.isfinite(v) else None for v in values], **output,
            evaluation=dict(truth=None, boundary=None, relation='UNLABELED')))
    assert set(clips) == set(metadata)
    for c in clips.values():
        assert len(c['frames']) == metadata[c['id']]['frame_count']
        assert all(abs(f['time_s'] - i * .2) < 1e-6 for i,f in enumerate(c['frames']))
        assert not c['frames'][0]['decision']['previous_strong']
        assert all(f['evaluation']['truth'] is None for f in c['frames'])
    payload = dict(id='showcase', title='City Sample · 高清城市巡游', illustrative=True,
        subtitle='六处原生城市位置、八段分步回放；1080p展示与固定64区读出，仅作演示，不并入验证成绩。',
        source=OUT.name, clips=list(clips.values()))
    path = SITE / 'extra-data.js'
    path.write_text('window.CORE_DEMO_EXTRA=' + json.dumps(payload, ensure_ascii=False,
                    allow_nan=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    (OUT / 'observations.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
    frozen_code = {p.name: sha(p) for p in [Path(__file__).with_name('policy.py'),
        NEARFIELD / 'ba_camera_corridor.py', NEARFIELD / 'tof_fov45_core.py',
        NEARFIELD / 'tof_corridor_calibration.py']}
    export = dict(status='PASS', scope='ILLUSTRATIVE_NATIVE_REPLAY_NOT_EVALUATION',
        built_at=datetime.now(timezone.utc).isoformat(), frames=len(rows), clips=len(clips),
        spec_sha256=sha(OUT/'spec.json'), plan_sha256=sha(OUT/'plan.json'), capture_receipts=capture_receipts,
        source_payload_hashes=source_hashes, asset_hashes=hashes, export_sha256=sha(path),
        frozen_code=frozen_code, no_truth=True, no_metrics=True, all_planned_clips_retained=True,
        motion='POSED_QUASI_STATIC_SAMPLED_TRAJECTORIES_NOT_REAL_TIME',
        limits=['Same simulated sensor proxy; not physical ToF.',
                'No evaluation truth or performance conclusions for these scene illustrations.',
                'Original 1296-frame frozen data and metrics are unchanged.'])
    (OUT/'export-receipt.json').write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:export[k] for k in ('status','scope','frames','clips')}))


if __name__ == '__main__':
    main()
