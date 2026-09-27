"""Render a scale cohort with a perturbed sensor model (Development robustness checks).

Wraps cnh_track_a_v13_sensor.run and overrides SensorParameters fields for every SNR
tier, e.g. the real-anchored crosstalk (x13), range offset (histogram 7.5 cm farther
=> range_zero_m -0.075) and pulse tail. Geometry, seeds and labels are unchanged.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import cnh_track_a_v13_sensor as s

p = argparse.ArgumentParser()
p.add_argument('--geometry', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--unit', type=int, required=True)
p.add_argument('--family', required=True)
p.add_argument('--override', type=json.loads, required=True, help='JSON dict of SensorParameters fields')
a = p.parse_args()
base = s.reference_parameters


def patched():
    params, g3 = base()
    return {k: replace(v, **a.override) for k, v in params.items()}, dict(g3, override=a.override)


s.reference_parameters = patched
s.FAMILY, s.FORCE_RATE = a.family, 5
print(json.dumps(dict(unit=a.unit, override=a.override, rate_frames=s.run(a.geometry, a.output, a.unit, -10))))
