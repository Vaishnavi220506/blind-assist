"""Source materialization and observations-only frozen inference, separate stages."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

import ba_camera_corridor as c


def materialize():
    """Sensor-construction authority only; native truth never saved in observations."""
    from ba_nfo_data import sensor
    out = c.OUT; cap = out/'capture'; protocol = c.read(out/'protocol.json')
    assert not (out/'observation-seal.json').exists()
    receipt = c.read(cap/'receipt.json'); released = c.read(cap/'process-release.json')
    assert receipt['status'] == 'PASS' and receipt['frame_count'] == 96 and released['released']
    assert receipt['protocol_sha256'] == c.sha(out/'protocol.json') and receipt['source_unchanged']
    launch = c.read(cap/'launch-receipt.json')
    assert receipt['readiness_helper_sha256'] == launch['readiness_helper_sha256'] == c.sha(Path(__file__).with_name('ue_capture_readiness.py'))
    plugin = Path(launch['plugin_path'])
    assert c.sha(plugin) == launch['plugin_sha256']
    assert c.sha(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll') == launch['plugin_binary_sha256']
    spec = c.read(out/'spec.json'); assert c.sha(out/'spec.json') == protocol['spec_sha256']
    manifest = c.read(cap/'observations/manifest.json'); geometry = c.read(cap/'evaluator/geometry.json')
    assert len(manifest['frames']) == len(geometry) == len(spec['cases']) == 96
    folder = out/'observations'; folder.mkdir(exist_ok=True)
    observed, sources, checks = [], [], []
    clips = {row['clip_id']: i for i, row in enumerate(spec['clips'])}
    f = c.WIDTH/(2*np.tan(np.deg2rad(c.HFOV/2)))
    yy, xx = np.mgrid[:c.HEIGHT, :c.WIDTH]
    ra, rb = (xx+.5-c.WIDTH/2)/f, (yy+.5-c.HEIGHT/2)/f
    for i, (row, geo, case) in enumerate(zip(manifest['frames'], geometry, spec['cases'])):
        assert row['sample_index'] == geo['sample_index'] == i
        assert row['id'] == geo['id'] == case['name']
        ready = geo['readiness']; state = ready['observation']
        assert ready['status'] == 'READY' and ready == {k:v for k,v in receipt['view_readiness'][i].items() if k != 'sample_index'}
        assert all(state[k] == 0 for k in ('asset_compilation_remaining','shader_jobs_remaining','pending_render_assets'))
        assert not state['asset_registry_loading'] and state['streaming_update_completed']
        warm = 32 if case['frame_in_clip'] == 0 else 16
        assert geo['unchanged_warmup_ticks'] == warm and geo['post_ready_rgb_render_calls'] == warm+1
        rgbpath = cap/'observations'/row['rgb_path']; nativepath = cap/'evaluator'/geo['native_path']
        assert c.sha(rgbpath) == row['rgb_sha256'] == geo['rgb_sha256']
        assert c.sha(nativepath) == geo['native_sha256']
        native = np.load(nativepath, allow_pickle=False); assert native.shape == (c.HEIGHT, c.WIDTH)
        known = np.isfinite(native) & (native > 0)
        rgb = cv2.cvtColor(cv2.imread(str(rgbpath)), cv2.COLOR_BGR2RGB)
        assert rgb.shape == (c.HEIGHT, c.WIDTH, 3)
        low = cv2.resize(rgb, (c.LOW_W, c.LOW_H), interpolation=cv2.INTER_AREA)
        sample = c.sample_native(native)
        # Common random seed within each physical pair; no forced equal ranges.
        seed_identity = case['pair_id']+'/'+str(case['frame_in_clip'])
        _, _, boxes, values = sensor(sample, seed_identity)
        identity = f'f{i:04d}'
        path = folder/(identity+'.npz'); assert not path.exists()
        np.savez_compressed(path, rgb=low, boxes=boxes, values=values)
        observed.append(dict(id=identity, clip_id=f's{clips[row["clip_id"]]:02d}',
            frame_in_clip=row['frame_in_clip'], time_s=row['time_s'],
            prepared=path.relative_to(out).as_posix(), prepared_sha256=c.sha(path),
            rgb_path=rgbpath.relative_to(out).as_posix(), rgb_sha256=row['rgb_sha256'],
            calibration=manifest['calibration']))
        target = next(obj for obj in geo['objects'] if obj['name'] == case['target_name'])
        assert target['trace']['hit_expected_actor']
        center = np.array(target['render_bounds_center_m']); extent = np.array(target['render_bounds_extent_m'])
        camera = case['camera']
        world = np.stack([native+camera['x'], ra*native+camera['y'], camera['z']-rb*native], -1)
        on_target = known & np.all((world >= center-extent-.02) & (world <= center+extent+.02), axis=-1)
        inside = known & (native >= .3) & (native <= 3.) & (np.abs(ra*native) <= .3) & (rb*native >= -.2) & (rb*native <= .9)
        other = int((inside & ~on_target).sum())
        same_zone = None
        if case['pair_id'] == 'g4_same_zone':
            lo, hi = center-extent, center+extent
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            z = corners[:, 0]-camera['x']
            u = (corners[:, 1]-camera['y'])/z*f+c.WIDTH/2
            v = (camera['z']-corners[:, 2])/z*f+c.HEIGHT/2
            y0, x0, y1, x1 = boxes[4*8+4]
            same_zone = bool(u.min() >= x0*c.WIDTH/c.LOW_W and u.max() <= x1*c.WIDTH/c.LOW_W and
                             v.min() >= y0*c.HEIGHT/c.LOW_H and v.max() <= y1*c.HEIGHT/c.LOW_H)
            assert same_zone, 'Actual G4 geometry not inside the frozen shared zone'
        check = dict(id=identity, source_id=row['id'], native_valid_pixels=int(known.sum()),
                     visible_target_pixels=int(on_target.sum()), competing_corridor_pixels=other,
                     g4_same_zone=same_zone)
        checks.append(check)
        sources.append(dict(id=identity, original=row, geometry=geo, case=case, check=check,
                            sensor_seed_identity=seed_identity, sensor_native_sha256=geo['native_sha256']))
    # Source failure never becomes a method-negative or silent frame exclusion.
    admitted = all(r['visible_target_pixels'] > 0 and r['competing_corridor_pixels'] < 4 for r in checks)
    c.write(out/'source-admission.json', dict(status='PASS' if admitted else 'NOT_EVALUABLE',
        frames=96, checks=checks, source_unchanged=True, process_released=True,
        note='Native point-ray support within2cm authenticated target bounds; competing known native corridor support must be<4 pixels perframe. Exactcontact primitive label separate from sampled visible mask.'))
    c.write(out/'evaluator-source.json', sources)
    c.write(out/'observations.json', observed)
    assert admitted, 'Source admission failed; preserve all source evidence and stop before inference'
    c.write(out/'observation-seal.json', dict(status='COMPLETE', frames=96,
        protocol_sha256=c.sha(out/'protocol.json'), observations_sha256=c.sha(out/'observations.json'),
        evaluator_source_sha256=c.sha(out/'evaluator-source.json'),
        capture_geometry_sha256=c.sha(cap/'evaluator/geometry.json'),
        source_admission_sha256=c.sha(out/'source-admission.json'), materializer_sha256=c.sha(__file__),
        sensor_source_sha256=c.sha(Path(__file__).with_name('ba_nfo_data.py'))))
    print('CORRIDOR_OBSERVATIONS_SEALED', len(observed), flush=True)


@torch.inference_mode()
def infer():
    """This stage loads only observation files and frozen weights."""
    from ba_nfo_depthpro import model_and_transform, OUT as DP, WEIGHT_SHA, sha as bigsha
    from ba_nfo_zone_readout import load_base, BASE
    import ba_nfo_matched as m
    out = c.OUT; seal = c.read(out/'observation-seal.json'); protocol = c.read(out/'protocol.json')
    assert seal['status'] == 'COMPLETE' and seal['protocol_sha256'] == c.sha(out/'protocol.json')
    assert seal['observations_sha256'] == c.sha(out/'observations.json')
    assert not (out/'prediction-seal.json').exists()
    repairs = c.read(out/'mechanical-repairs.json')['repairs'] if (out/'mechanical-repairs.json').exists() else []
    for name, digest in protocol['hashes'].items():
        current = c.sha(Path(__file__).with_name(name))
        assert current == digest or any(name in ('ba_camera_corridor_capture.py','launch_ba_camera_corridor.py') and
            r['file'] == name and r['original_sha256'] == digest and r['repaired_sha256'] == current and
            r['model_calls_before_repair'] == 0 and r['unchanged_spec_sha256'] == protocol['spec_sha256'] and
            (r['frames_before_repair'] == 0 or r['kind'] == 'SOURCE_READINESS') for r in repairs), name
    assert bigsha(DP/'depth_pro.pt') == WEIGHT_SHA
    nfo_sha = '2e34ee78ea931ad8ff85d3b60ab7bccf020ce3e32120308dbdce0b19465449ae'
    assert bigsha(BASE/'trained-nfo.pt') == nfo_sha
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    c.write(out/'inference-launch.json', dict(protocol_sha256=c.sha(out/'protocol.json'),
        observation_seal_sha256=c.sha(out/'observation-seal.json'), runner_sha256=c.sha(__file__),
        nfo_weight_sha256=nfo_sha, depthpro_weight_sha256=WEIGHT_SHA,
        backend='CUDA fp32 NFO; CUDA fp16 official Depth Pro; CPU scalar/graph readout',
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        placement_evidence=str(DP/'backend.json'), placement='Reuse verified same retained model/device runtime; native640x360 was already exercised on G5'))
    rows = c.read(out/'observations.json'); dest = out/'predictions'; dest.mkdir(exist_ok=True)
    # Small frozen NFO stays loaded; inference call graph never accepts source geometry.
    nfo = load_base().cuda().eval().requires_grad_(False)
    depthpro, transform = model_and_transform('cuda')
    receipts = []; began = time.perf_counter()
    for i, row in enumerate(rows):
        path = out/row['prepared']; assert c.sha(path) == row['prepared_sha256']
        rgbpath = out/row['rgb_path']; assert c.sha(rgbpath) == row['rgb_sha256']
        target = dest/(row['id']+'.npz'); assert not target.exists()
        with np.load(path, allow_pickle=False) as data:
            rgb, boxes, values = data['rgb'].copy(), data['boxes'].copy(), data['values'].copy()
        native_rgb = cv2.cvtColor(cv2.imread(str(rgbpath)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2,0,1).copy()[None]).cuda()
        zones = torch.from_numpy(m.public_zones(values)[None]).cuda()
        torch.cuda.synchronize(); tick = time.perf_counter()
        probabilities = m.probabilities(nfo(x, zones), 'nfo')[0].float().cpu().numpy()
        torch.cuda.synchronize(); nfo_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        result = depthpro.infer(transform(native_rgb), f_px=torch.tensor(row['calibration']['fx'], device='cuda'))
        assert result['depth'].device.type == 'cuda'
        native_depth = result['depth'].float().cpu().numpy()
        torch.cuda.synchronize(); depthpro_seconds = time.perf_counter()-tick
        assert native_depth.shape == (c.HEIGHT,c.WIDTH) and np.isfinite(native_depth).all() and (native_depth > 0).all()
        low_depth = c.sample_native(native_depth)
        calibrated, calibration = c.global_scale(low_depth, boxes, values)
        raw, raw_support = c.tof_readout(boxes, values)
        nf, nf_possible, nf_definite = c.nfo_readout(probabilities)
        dp, dp_possible, dp_definite = c.depth_readout(calibrated)
        np.savez_compressed(target, nfo_probabilities=probabilities, native_depthpro=native_depth,
            depthpro_global=calibrated, nfo_possible=nf_possible,nfo_definite=nf_definite,
            depthpro_possible=dp_possible, depthpro_definite=dp_definite)
        item = dict(id=row['id'],clip_id=row['clip_id'],frame_in_clip=row['frame_in_clip'],time_s=row['time_s'],
            predictions=dict(raw_tof=raw,nfo=nf,depthpro_global=dp),raw_support=raw_support,
            global_calibration=calibration,nfo_seconds=nfo_seconds,depthpro_seconds=depthpro_seconds,
            sha256=c.sha(target),protocol_sha256=seal['protocol_sha256'],observation_sha256=row['prepared_sha256'])
        c.write(target.with_suffix('.json'),item); receipts.append(item)
        if (i+1)%12 == 0:
            print('CORRIDOR_INFERRED',i+1,'/96',round(time.perf_counter()-began,1),'seconds',flush=True)
    c.write(out/'prediction-seal.json',dict(status='COMPLETE',frames=96,outputs=receipts,
        protocol_sha256=c.sha(out/'protocol.json'),launch_sha256=c.sha(out/'inference-launch.json'),
        elapsed_seconds=time.perf_counter()-began,device=torch.cuda.get_device_name(),
        max_allocated_bytes=torch.cuda.max_memory_allocated(),training_updates=0,model_calls=192))
    print('CORRIDOR_PREDICTIONS_SEALED',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('materialize','infer'))
    args = parser.parse_args(); cv2.setNumThreads(1)
    {'materialize':materialize,'infer':infer}[args.command]()
