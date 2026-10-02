"""Optional fused engineering backend for the unchanged CVR quadrature recipe.

No training, scene labels or score access. One thread handles each quadrature
point, using original midpoint coordinates and angular/radial cell volumes.
Geometry/mass use float64; original Torch reductions sum the 27 points and
convert each frame to float32, followed by original ordered history accumulation.
This candidate is not enabled by this module: --check must verify and benchmark.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from cnh_cvr_projection import Projector, SHAPE, SUB, EDGE, WIDTH
from cnh_cvr_v2_materialize import BatchedProjector

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-sequence-extrinsic-yaw-20261002'
CHECK_NAME = 'fused_check_v3.json'

KERNEL = r'''
extern "C" __global__ void cvr_points(
    const double* points, const double* volumes, const double* z,
    const double* transforms, double* mass, bool* validity,
    const int npoints, const int nframes,
    const double edge, const double width, const double mass_numerator) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= npoints * nframes) return;
  const int frame = id / npoints;
  const int point = id - frame * npoints;
  const double* t = transforms + frame * 16;
  const double* values = z + frame * 1024;
    const double* p = points + point * 3;
    const double dx = p[0] - t[3];
    const double dy = p[1] - t[7];
    const double dz = p[2] - t[11];
    // Same row-vector inverse rigid transform: (point - translation) @ R.
    const double x = (dx*t[0] + dy*t[4]) + dz*t[8];
    const double y = (dx*t[1] + dy*t[5]) + dz*t[9];
    const double depth = (dx*t[2] + dy*t[6]) + dz*t[10];
    const double radius = sqrt((x*x + y*y) + depth*depth);
    const double divisor = fmax(depth, 1e-30);
    const long long ix = __double2ll_rz(floor(((x/divisor + edge) / (2.0*edge)) * 8.0));
    const long long iy = __double2ll_rz(floor(((y/divisor + edge) / (2.0*edge)) * 8.0));
    const long long ir = __double2ll_rz(floor(radius / width));
    const bool valid = depth > 0.0 && ix >= 0 && ix < 8 && iy >= 0 && iy < 8 && ir >= 0 && ir < 16;
    const int col = ix < 0 ? 0 : ix > 7 ? 7 : (int)ix;
    const int row = iy < 0 ? 0 : iy > 7 ? 7 : (int)iy;
    const int bin = ir < 0 ? 0 : ir > 15 ? 15 : (int)ir;
    const int cell = (row*8 + col)*16 + bin;
    const double weight = valid ? mass_numerator / volumes[cell] : 0.0;
    // Preserve signed invalid zero mass as well as the original reduction tree.
    mass[id] = values[cell] * weight;
    validity[id] = valid;
}
'''


class FusedProjector(Projector):
    def __init__(self, device='cuda'):
        # CuPy absence is an explicit unavailable backend, never an install.
        os.environ['CUPY_CACHE_DIR'] = str(OUT / 'cupy_kernel_cache')
        try:
            import cupy as cp
        except ImportError as error:
            raise RuntimeError('CuPy unavailable; retain original BatchedProjector') from error
        super().__init__(device=device)
        if SUB != 3 or SHAPE != (24, 17, 33):
            raise ValueError('Fused backend only supports unchanged 3x3x3 retained grid')
        # Original `valid * Python-float volume / 27` is computed in Torch's
        # default float dtype BEFORE dividing by float64 cell volumes. Preserve
        # that implicit rounding; using volume/27 directly in double differs.
        numerator = torch.ones((), dtype=torch.bool, device=device)*self.voxel_volume/(SUB**3)
        self.mass_numerator = float(numerator.double().item())
        self.numerator_dtype = str(numerator.dtype)
        self.cp = cp
        self.kernel = cp.RawKernel(KERNEL, 'cvr_points', options=(
            '--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'), backend='nvrtc')

    @torch.no_grad()
    def point_terms(self, z, transforms):
        n = len(z)
        if not 1 <= n <= 8 or len(transforms) != n:
            raise ValueError('Matching history length 1..8 required')
        values = torch.as_tensor(z, dtype=torch.float64, device=self.device).reshape(n, 1024).contiguous()
        t = torch.as_tensor(transforms, dtype=torch.float64, device=self.device).reshape(n, 4, 4).contiguous()
        nvoxels = int(np.prod(SHAPE))
        mass = torch.empty((n, nvoxels, SUB**3), device=self.device, dtype=torch.float64)
        valid = torch.empty((n, nvoxels, SUB**3), device=self.device, dtype=torch.bool)
        stream = torch.cuda.current_stream(device=self.device)
        # DLPack views alias Torch-owned tensors. All producer/consumer work is
        # submitted on this exact stream, including the subsequent Torch adds.
        # Public DLPack producers additionally synchronize stream handoff.
        with self.cp.cuda.Device(mass.device.index), self.cp.cuda.ExternalStream(stream.cuda_stream):
            arrays = [self.cp.from_dlpack(x) for x in (self.points, self.volumes, values, t, mass, valid)]
            npoints = nvoxels*SUB**3
            self.kernel(((n*npoints+255)//256,), (256,), tuple(arrays)+(
                np.int32(npoints), np.int32(n), np.float64(EDGE), np.float64(WIDTH), np.float64(self.mass_numerator)))
        return mass, valid

    @torch.no_grad()
    def sequence(self, z, transforms):
        n = len(z)
        mass, valid = self.point_terms(z, transforms)
        e = mass.sum(2).reshape(n, *SHAPE).float()
        c = valid.double().mean(2).reshape(n, *SHAPE).float()
        total, count = torch.zeros_like(e[0]), torch.zeros_like(c[0])
        for current, seen in zip(e, c):
            total += current
            count += seen
        return torch.stack([total, count, e[-1]]).reshape(3, *SHAPE)


def rotation(deg, axis):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def synthetic_cases():
    rng = np.random.default_rng(2026100216)
    cases = []
    for n in (1, 4, 8):
        z = rng.normal(0., 3., (n, 8, 8, 16)).astype(np.float16)
        for bias in (-1., 0., 1.):
            poses = np.repeat(np.eye(4)[None], n, axis=0)
            for i in range(n):
                poses[i, :3, :3] = rotation(15.+(i-n+1)*1.25, 'y') @ rotation(-10., 'x')
                poses[i, :3, 3] = [.012*(i-n+1), .007*(i-n+1), .157*(i-n+1)]
            b = np.eye(4); b[:3, :3] = rotation(bias, 'y')
            cases.append((f'n{n}_bias{bias:+.0f}', z, b[None] @ poses))
    cases.append(('identity', rng.normal(size=(1, 8, 8, 16)).astype(np.float16), np.eye(4)[None]))
    return cases


def retained_regression_case():
    from cnh_cvr_pilot import motion_metadata, relative_transforms
    path = ROOT / 'artifacts.local/work/cnh-near-range-20261001/features/evaluation/unit94002.npz'
    with np.load(path, allow_pickle=False) as data:
        z = data['z1'][data['scene'] == 0][:4].copy()
    sensor, travel, noisy = motion_metadata(94002, 0)
    return 'retained94002_c0_f3_bias0', z, relative_transforms(sensor, travel, noisy, 3)


def write_report(report):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / CHECK_NAME
    if path.exists():
        raise FileExistsError('Preserve existing fused check; inspect before another benchmark')
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def check():
    if (OUT / CHECK_NAME).exists():
        raise FileExistsError('Existing fused benchmark must not be silently repeated')
    started = time.monotonic()
    report = dict(status='NOT_RUN', enabled=False, gate='raw allclose(rtol=1e-6,atol=0) AND bit-exact float16 AND measured faster',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quadrature='original grid()/cell_volumes(), 3x3x3 midpoint; no changed sampling',
        kernel_options=['--fmad=false', '--prec-div=true', '--prec-sqrt=true'],
        data_scope='10 synthetic cases plus only retained failure unit94002/config0/frame3/bias0; no labels or model scores',
        original_frame_reduction='Torch float64 mass.sum(2), valid.double().mean(2); ordered float32 history additions',
        recipe_changed=False, cases=[])
    original, candidate = None, None
    try:
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        original, candidate = BatchedProjector(), FusedProjector()
        report['cupy_version'] = candidate.cp.__version__
        report['device'] = torch.cuda.get_device_name()
        report['original_weight_numerator'] = candidate.mass_numerator
        report['original_weight_numerator_dtype'] = candidate.numerator_dtype
        cases = synthetic_cases()+[retained_regression_case()]
        # Explicit synchronization only at verification/timing boundaries; the
        # production candidate shares the current stream without global stalls.
        for name, z, poses in cases:
            expected = original.sequence(z, poses)
            actual = candidate.sequence(z, poses)
            torch.cuda.current_stream().synchronize()
            delta = (actual-expected).abs()
            nonzero = expected != 0
            relative = delta[nonzero]/expected[nonzero].abs()
            raw_pass = bool(torch.allclose(actual, expected, rtol=1e-6, atol=0))
            half_diff = actual.half().view(torch.int16) != expected.half().view(torch.int16)
            half_pass = not bool(half_diff.any())
            report['cases'].append(dict(name=name, history=len(z), raw_pass=raw_pass, half_exact=half_pass,
                max_abs=float(delta.max()), max_relative_nonzero=float(relative.max()) if len(relative) else 0.,
                zero_reference_mismatches=int(torch.count_nonzero((expected == 0) & (actual != 0))),
                half_mismatch_elements=int(half_diff.sum())))
        if not all(x['raw_pass'] and x['half_exact'] for x in report['cases']):
            report.update(status='NOT_NUMERICALLY_EQUIVALENT', elapsed_s=time.monotonic()-started)
            write_report(report)
            print(json.dumps(report), flush=True)
            return
        timings = {}
        for n in (1, 4, 8):
            name, z, poses = next(c for c in cases if c[0] == f'n{n}_bias+1')
            timings[str(n)] = {}
            for backend, projector in (('original', original), ('fused', candidate)):
                for _ in range(2):
                    projector.sequence(z, poses)
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(7):
                    begin = time.perf_counter()
                    projector.sequence(z, poses)
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter()-begin)
                timings[str(n)][backend] = dict(median_s=float(np.median(elapsed)), samples_s=elapsed)
            timings[str(n)]['speedup'] = timings[str(n)]['original']['median_s']/timings[str(n)]['fused']['median_s']
        faster = all(value['speedup'] > 1 for value in timings.values())
        report.update(status='PASS_ENGINEERING_CANDIDATE' if faster else 'NOT_FASTER', enabled=False,
                      numerical_equivalence=True, faster_all_histories=faster, timings=timings,
                      elapsed_s=time.monotonic()-started,
                      limits=['Synthetic backend check only; caller must explicitly select candidate',
                              'No proof of real experiment cache equivalence until caller performs its retained canary',
                              'Original projector and experimental recipe remain unchanged'])
        write_report(report)
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report.update(status='UNAVAILABLE_OR_FAILED', error=repr(error), elapsed_s=time.monotonic()-started)
        if not (OUT / CHECK_NAME).exists():
            write_report(report)
        raise
    finally:
        original = candidate = None
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    check()
