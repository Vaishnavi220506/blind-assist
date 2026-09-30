"""Observation-only one-plane CNH decomposition, exploratory and uncalibrated.

Input is signed [8,8,16] CNH, ambient and nominal sensor parameters. No scene,
label, wall truth, target mask or pose is accepted. Keep the raw observation and
fitted plane alongside the signed residual: a fitted plane can itself be danger.
"""
from dataclasses import replace
import time

import numpy as np
from scipy.optimize import least_squares

from cnh_route_sensor import RAW_BIN_M, RAW_BINS, angular_rays, _pulse_matrix


class PlaneFitter:
    def __init__(self, params, samples=4):
        self.params = replace(params, noise_scale=0.)
        self.rays, weights = angular_rays(samples)
        self.weights = weights / weights.sum(-1, keepdims=True)
        self.pulse = _pulse_matrix(self.params).reshape(RAW_BINS, 16, 8).sum(-1)
        empty = np.zeros((8, 8, 16))
        k = int(np.floor((params.crosstalk_range_m-params.range_zero_m)/RAW_BIN_M))
        if 0 <= k < RAW_BINS:
            empty[:] = params.signal_counts*params.crosstalk_fraction*self.pulse[k]
        self.electronics = empty
        normals = []
        for pitch in np.deg2rad([-90, -60, -30, 0, 30, 60, 90]):
            for yaw in np.deg2rad(np.arange(-90, 91, 15)):
                n = self.normal(yaw, pitch)
                if not any(np.allclose(n, other[2]) for other in normals):
                    normals.append((yaw, pitch, n))
        self.grid = np.array([(y, p, d) for y, p, _ in normals for d in np.arange(.15, 4.51, .15)])
        self.templates = np.stack([self.plane(*x) for x in self.grid]).reshape(len(self.grid), -1)
        self.squared = self.templates**2
        self.backend = 'numpy/cpu'

    @staticmethod
    def normal(yaw, pitch):
        return np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])

    def plane(self, yaw, pitch, offset):
        """Unit-reflectance infinite plane; soft raw-bin assignment for fitting.

        Nominal forward uses hard bins. Interpolating between raw-bin centers
        makes geometry optimizable and is an explicit approximation, not truth.
        """
        cosine = self.rays @ self.normal(yaw, pitch)
        valid = cosine > 1e-8
        distance = offset / np.maximum(cosine, 1e-8)
        energy = self.params.signal_counts*self.weights*np.maximum(cosine, 0)/np.maximum(distance, .05)**2
        pos = (distance-self.params.range_zero_m)/RAW_BIN_M-.5
        index = np.floor(pos).astype(np.int64)
        frac = pos-index
        raw = np.zeros((64, RAW_BINS))
        rows = np.broadcast_to(np.arange(64).reshape(8, 8, 1), index.shape)
        for idx, weight in ((index, 1-frac), (index+1, frac)):
            ok = valid & (idx >= 0) & (idx < RAW_BINS)
            np.add.at(raw, (rows.ravel(), np.clip(idx, 0, RAW_BINS-1).ravel()), np.where(ok, energy*weight, 0).ravel())
        signal = (raw @ self.pulse).reshape(8, 8, 16)
        leak = self.params.neighbour_leak/4
        old = signal.copy()
        signal[1:] += leak*(old[:-1]-old[1:])
        signal[:-1] += leak*(old[1:]-old[:-1])
        signal[:, 1:] += leak*(old[:, :-1]-old[:, 1:])
        signal[:, :-1] += leak*(old[:, 1:]-old[:, :-1])
        return signal

    def grid_scores(self, y, inv_var):
        dot = self.templates @ (y*inv_var)
        norm = self.squared @ inv_var
        amplitude = np.clip(dot/np.maximum(norm, 1e-20), 0, 2)
        score = 2*amplitude*dot-amplitude**2*norm
        return score, amplitude

    def select_backend(self, histogram, ambient):
        """Benchmark the actual grid search including host transfers."""
        import torch
        from research_backend import BackendCandidate, DeviceObservation, Workload, benchmark
        y, inv_var = self.prepare(histogram, ambient)
        cpu = benchmark(BackendCandidate('numpy', 'cpu', lambda: self.grid_scores(y, inv_var),
            lambda _: DeviceObservation('cpu', 'CPU', 'numpy')), repeats=5)
        result = dict(cpu=cpu.__dict__, workload=Workload.BATCH_TENSOR.value)
        if not torch.cuda.is_available():
            result.update(selected='numpy/cpu', reason='ACCELERATOR_UNAVAILABLE')
            return result
        torch.backends.cuda.matmul.allow_tf32 = False
        matrix = torch.as_tensor(self.templates, dtype=torch.float64, device='cuda')
        square = matrix.square()

        def run():
            yy = torch.as_tensor(y*inv_var, dtype=torch.float64, device='cuda')
            ww = torch.as_tensor(inv_var, dtype=torch.float64, device='cuda')
            dot, norm = matrix @ yy, square @ ww
            amp = (dot/norm.clamp_min(1e-20)).clamp(0, 2)
            return (2*amp*dot-amp.square()*norm).cpu().numpy(), amp.cpu().numpy()

        gpu = benchmark(BackendCandidate('torch', 'cuda', run,
            lambda _: DeviceObservation('cuda', torch.cuda.get_device_name(), 'torch'), torch.cuda.synchronize), repeats=5)
        np.testing.assert_allclose(run()[0], self.grid_scores(y, inv_var)[0], rtol=1e-9, atol=1e-9)
        result['gpu'] = gpu.__dict__
        if gpu.median_seconds < cpu.median_seconds:
            self.backend = 'torch/cuda grid; scipy/cpu local refinement'
            self._cuda_matrix, self._cuda_squared = matrix, square
            result.update(selected=self.backend, reason='GPU_FASTER_MEASURED')
        else:
            result.update(selected=self.backend, reason='CPU_FASTER_MEASURED')
        return result

    def prepare(self, histogram, ambient):
        h, a = np.asarray(histogram, float), np.asarray(ambient, float)
        if h.shape != (8, 8, 16) or a.shape != (8, 8) or not np.isfinite(h).all() or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError('Need finite 8x8x16 signed CNH and nonnegative 8x8 ambient')
        # Observed noise proxy. Subtraction does not remove wall shot noise.
        var = np.maximum(h, 0)+16*a[..., None]+1
        return (h-self.electronics).ravel(), (1/var).ravel()

    def fit(self, histogram, ambient):
        started = time.perf_counter()
        y, inv_var = self.prepare(histogram, ambient)
        if hasattr(self, '_cuda_matrix'):
            import torch
            yy = torch.as_tensor(y*inv_var, dtype=torch.float64, device='cuda')
            ww = torch.as_tensor(inv_var, dtype=torch.float64, device='cuda')
            dot, norm = self._cuda_matrix @ yy, self._cuda_squared @ ww
            amp = (dot/norm.clamp_min(1e-20)).clamp(0, 2)
            score, amplitude = (2*amp*dot-amp.square()*norm).cpu().numpy(), amp.cpu().numpy()
        else:
            score, amplitude = self.grid_scores(y, inv_var)
        starts = np.argsort(score)[-3:][::-1]
        root_weight = np.sqrt(inv_var)
        best = None
        for i in starts:
            initial = np.r_[self.grid[i], amplitude[i]]
            solution = least_squares(lambda x: (x[3]*self.plane(*x[:3]).ravel()-y)*root_weight,
                initial, bounds=([-np.pi, -np.pi/2, .08, 0], [np.pi, np.pi/2, 5, 2]),
                loss='soft_l1', f_scale=2., diff_step=1e-3, max_nfev=65)
            if best is None or solution.cost < best.cost:
                best = solution
        fitted = best.x[3]*self.plane(*best.x[:3])
        residual = np.asarray(histogram)-self.electronics-fitted
        return dict(plane=fitted, residual=residual, electronics=self.electronics.copy(),
            normal=self.normal(*best.x[:2]), offset_m=float(best.x[2]), amplitude=float(best.x[3]),
            cost=float(best.cost), elapsed_s=time.perf_counter()-started,
            optimizer_success=bool(best.success), evaluations=int(best.nfev))
