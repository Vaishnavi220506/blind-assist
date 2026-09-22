"""Public-only current-frame corridor union; no truth, model or memory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mz115_spatial_allocation import slant_envelope, zone_box, possible

ROOT = Path(__file__).resolve().parents[5]
ART = (ROOT/'artifacts.local').resolve()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def support_readout(row, yaw):
    if not row.get('imu_valid') or not finite(yaw):
        return dict(state='UNKNOWN_ORIENTATION', contributors=[])
    origin = row['camera_in_body_m']
    if len(origin) != 3 or not all(finite(v) for v in origin):
        raise ValueError('Invalid extrinsics')
    contributors = []
    if row['tof_packet_received']:
        for z in row['tof_zones']:
            for slot, t in enumerate(z['targets']):
                d, sigma, strength = (t.get(k) for k in ('distance_m', 'range_noise_sigma_m', 'signal_strength_proxy'))
                if (t.get('status') not in ('SIM_VALID', 'SIM_MERGED') or
                    not all(finite(v) for v in (d, sigma, strength)) or d <= 0 or min(sigma, strength) < 0):
                    continue
                ranges = (.02, 4.) if t['status'] == 'SIM_MERGED' else (max(.02, d-3*sigma), d+3*sigma)
                bounds = slant_envelope(zone_box(z, row['rgb_intrinsics']), ranges,
                    row['rgb_intrinsics'], (row['camera_pitch_deg'],)*2, (yaw,)*2, 0.)
                bounds = [[lo+o, hi+o] for (lo, hi), o in zip(bounds, origin)]
                if possible(bounds):
                    contributors.append(dict(sensor='tof', zone_id=z['zone_id'], target_index=slot,
                        bounds=bounds, status=t['status']))
    if row['radar_packet_received']:
        for slot, (r, a, valid) in enumerate(zip(row['radar_range_m'], row['radar_angle'], row['radar_valid'])):
            if valid and finite(r) and finite(a) and r > 0:
                x, y = r*math.cos(math.radians(a+yaw)), r*math.sin(math.radians(a+yaw))
                if .2 <= x <= 3.6 and abs(y) <= .3:
                    contributors.append(dict(sensor='radar', slot=slot, x=x, y=y,
                        height_authority='UNMEASURED_INHERITED_HORIZONTAL_BRANCH'))
    return dict(state='POSSIBLE_OCCUPANCY' if contributors else 'UNKNOWN_NO_INTERSECTING_SUPPORT', contributors=contributors)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();capture=a.capture.resolve();baseline=a.baseline.resolve()
    assert out.is_relative_to(ART) and out != ART and not out.exists()
    completed=json.loads((baseline/'completion.json').read_text())
    assert completed['status']=='PASS' and completed['evaluator_read'] is False
    for name,digest in completed['outputs'].items():
        assert sha(baseline/name)==digest
    receipt=json.loads((capture/'receipt.json').read_text())
    assert receipt['status']=='PASS' and sha(capture/'raw.jsonl')==receipt['hashes']['raw.jsonl']
    rows=[json.loads(l) for l in (capture/'raw.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (baseline/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==len(old)==receipt['frames']
    protocol=Path(__file__).with_name('MZ178_PROTOCOL_20260918.md')
    inputs={str(p):sha(p) for p in (capture/'raw.jsonl',baseline/'predictions.jsonl',protocol,Path(__file__),
        Path(__file__).resolve().parent.parent/'mz115_spatial_allocation.py')}
    result=[];ep=None;yaw=0.
    for row,b in zip(rows,old):
        assert row['id']==b['id'] and row['episode_id']==b['episode_id'] and row['time_s']==b['time_s']
        if row['episode_id']!=ep:
            ep=row['episode_id'];yaw=0.
        if row['imu_valid']:
            yaw+=float(row['delta_yaw'])
        assert abs(yaw-b['yaw'])<1e-10
        arms={name:support_readout(row, angle) for name,angle in [('head_control',0.),('body_current',yaw)]}
        result.append(dict(id=row['id'],episode_id=ep,time_s=row['time_s'],yaw=yaw,
            baseline_alert=b['astar_alert'],arms={name:dict(v,alert=b['astar_alert'] or bool(v['contributors'])) for name,v in arms.items()}))
    assert all(sha(p)==h for p,h in inputs.items())
    out.mkdir(parents=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result),encoding='utf-8')
    write(out/'input-seal.json',dict(inputs=inputs,evaluator_read=False,source_labels_used=False,
        backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE',memory=False,training_steps=0))
    write(out/'completion.json',dict(status='PASS',frames=len(rows),evaluator_read=False,
        outputs={n:sha(out/n) for n in ('predictions.jsonl','input-seal.json')}))
    print(json.dumps(dict(status='PASS',frames=len(rows))))


if __name__=='__main__':
    main()
