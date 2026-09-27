"""Per-frame inputs for a learned CNH readout (Development, fast lane).

For every frame: z1 = r/sqrt(v) (current frame) and z4 = the S2 four-frame transported
per-cell total over sqrt of its per-cell variance (noisy ego-motion, power 1, same
transport as cnh_track_a_gpu_readout.sequence_readouts), plus the six tau=0.75 query
supports. Labels/main/witness/strata come from the unit records; no oracle is read.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import cnh_track_a_gpu_readout as g
import cnh_track_a_scale_evaluate as se
from cnh_track_a_readout import noisy_poses

FAMILY = 'cnh-track-a-scale-v2-20260926'


def sequence_features(hist, ambient, bias, tq, noisy):
    r = g.T(hist)-g.T(bias)
    v = 16*g.T(ambient)[..., None]+g.T(bias).clamp_min(0)
    n = len(r)
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    total, var = rf.clone(), vf.clone()
    p = torch.as_tensor(np.asarray(noisy), dtype=g.D64, device=g.DEV)
    pairs = [(i, j) for i in range(1, n) for j in range(max(0, i-3), i)]
    if pairs:
        rel = torch.stack([torch.linalg.inv(p[i]) @ p[j] for i, j in pairs])
        A = g.transport(rel, 1).to(g.DT)
        for k, (i, j) in enumerate(pairs):
            total[i] += A[k] @ rf[j]
            var[i] += (A[k]*A[k]) @ vf[j]
    z4 = (total/var.clamp_min(1e-9).sqrt()).reshape(n, 8, 8, 16)
    z1 = (rf/vf.clamp_min(1e-9).sqrt()).reshape(n, 8, 8, 16)
    w = g.query_weights(torch.as_tensor(np.asarray(tq), dtype=g.D64, device=g.DEV))
    sup = (w >= .75).reshape(n, 6, 8, 8, 16)
    return z4.cpu().numpy(), z1.cpu().numpy(), sup.cpu().numpy()


def unit_features(geometry, sensor, unit, bias, family=FAMILY):
    se.sensor_module.FAMILY = family
    split, records, _ = se.unit_records(geometry, sensor, unit, -10, 1)
    rows = {}
    for rec in records:
        z4, z1, sup = sequence_features(rec['hist'], rec['ambient'], bias, rec['tq'],
                                        noisy_poses(rec['poses'], rec['ego_seed'], dt=.2))
        item = dict(z4=z4.astype(np.float16), z1=z1.astype(np.float16), sup=np.packbits(sup, axis=-1),
                    labels=rec['labels'], main=rec['main'], witness=rec['witness'],
                    config=np.full(len(z4), rec['config']), frame=np.arange(len(z4)))
        for k, x in item.items():
            rows.setdefault(k, []).append(x)
    out = {k: np.concatenate(x) for k, x in rows.items()}
    out['strata'] = se.strata_of(records)
    out['split'] = np.array(split)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, nargs='+', required=True, help='directories searched in order')
    p.add_argument('--bias', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--family', default=FAMILY)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bias = np.load(a.bias)
    for u in a.units:
        target = a.out/f'unit{u:03d}.npz'
        if target.exists():
            continue
        sensor = next((s for s in a.sensor if (s/f'unit{u:02d}-mount-10-observations.npz').exists()), None)
        if sensor is None:
            print(u, 'no sensor data, skipped', flush=True)
            continue
        np.savez_compressed(target, **unit_features(a.geometry, sensor, u, bias, a.family))
        print(u, flush=True)


if __name__ == '__main__':
    main()
