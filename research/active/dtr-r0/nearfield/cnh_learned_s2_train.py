"""S2 scores (noisy ego-motion, tau 0.75) for units without saved readouts, same GPU implementation.

Writes unitNN.npz with G0 (= S2/noisy|0.75) and labels so cnh_learned_readout.load can attach them.
"""
import argparse
from pathlib import Path
import numpy as np
import cnh_track_a_gpu_readout as g
import cnh_track_a_scale_evaluate as se
from cnh_track_a_readout import noisy_poses

p = argparse.ArgumentParser()
p.add_argument('--geometry', type=Path, required=True)
p.add_argument('--sensor', type=Path, required=True)
p.add_argument('--bias', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--family', required=True)
p.add_argument('--units', type=int, nargs='+', required=True)
a = p.parse_args()
se.sensor_module.FAMILY = a.family
bias = np.load(a.bias)
a.out.mkdir(parents=True, exist_ok=True)
for u in a.units:
    target = a.out/f'unit{u:02d}.npz'
    if target.exists() or not (a.sensor/f'unit{u:02d}-mount-10-observations.npz').exists():
        continue
    _, records, _ = se.unit_records(a.geometry, a.sensor, u, -10, 1)
    s2, labels = [], []
    for rec in records:
        out = g.sequence_readouts(rec['hist'], rec['ambient'], bias, rec['poses'], rec['tq'],
                                  noisy_poses(rec['poses'], rec['ego_seed'], dt=.2), with_r4=False)
        s2.append(out['S2/noisy|0.75'])
        labels.append(rec['labels'])
    np.savez_compressed(target, G0=np.concatenate(s2), labels=np.concatenate(labels))
    print(u, flush=True)
