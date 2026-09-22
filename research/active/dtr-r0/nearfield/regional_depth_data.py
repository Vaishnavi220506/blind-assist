"""Point-sampled RGB/public ToF inputs and a separate training/evaluator label API.

The source's distances are optical Z in metres, not radial distances. Eight
point rays per axis per zone are selected from the existing 192x256 lattice;
neither RGB nor depth is interpolated. RGB inputs are unnormalized [0, 1].
"""
import numpy as np


def sample_geometry(boxes):
    """Return native y, x and X/Z, Y/Z arrays, each [64,64], in zone order."""
    b = np.asarray(boxes)
    if b.shape != (64, 4) or b.dtype.kind not in 'iu':
        raise ValueError('Expected 64 integer y0,x0,y1,x1 lattice boxes')
    if not (np.all(b[:, :2] >= 0) and np.all(b[:, 2:] <= [192, 256])
            and np.all(b[:, 2:] > b[:, :2])):
        raise ValueError('Empty or out-of-lattice box')
    grid = b.reshape(8, 8, 4)
    if not (np.all(grid[:, :, 0] == grid[:, :1, 0])
            and np.all(grid[:, :, 2] == grid[:, :1, 2])
            and np.all(grid[:, :, 1] == grid[:1, :, 1])
            and np.all(grid[:, :, 3] == grid[:1, :, 3])
            and np.all(grid[:, :-1, 3] == grid[:, 1:, 1])
            and np.all(grid[:-1, :, 2] == grid[1:, :, 0])):
        raise ValueError('Expected a contiguous row-major rectangular 8x8 grid')
    frac = (np.arange(8) + .5) / 8
    ly = np.floor(grid[:, :1, 0] + (grid[:, :1, 2] - grid[:, :1, 0]) * frac).astype(np.int64).reshape(64)
    lx = np.floor(grid[:1, :, 1].T + (grid[:1, :, 3] - grid[:1, :, 1]).T * frac).astype(np.int64).reshape(64)
    # Exactly the existing source sample_native rule, then native pixel centers.
    y = np.floor((ly + .5) * 360 / 192).astype(np.int64)
    x = np.floor((lx + .5) * 640 / 256).astype(np.int64)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    focal = 640 / (2 * np.tan(np.deg2rad(50)))
    a = ((xx + .5 - 320) / focal).astype(np.float32)
    c = ((yy + .5 - 180) / focal).astype(np.float32)
    return yy, xx, a, c


def build_observation(rgb, ranges, boxes):
    """Observation-only [7,64,64]: RGB, range/8, validity, X/Z, Y/Z.

    No evaluator data or filesystem access is accepted. Invalid returns become
    zero with validity zero; they do not establish clear space.
    """
    rgb, ranges = np.asarray(rgb), np.asarray(ranges, dtype=np.float32)
    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected native uint8 RGB [360,640,3]')
    if ranges.shape != (64,):
        raise ValueError('Expected 64 public axial distances')
    yy, xx, a, b = sample_geometry(boxes)
    valid = np.isfinite(ranges) & (ranges >= .1) & (ranges < 8)
    observed = np.where(valid, ranges, 0).reshape(8, 8) / 8
    expand = lambda x: np.repeat(np.repeat(x, 8, axis=0), 8, axis=1)
    value = np.concatenate([
        rgb[yy, xx].transpose(2, 0, 1).astype(np.float32) / 255,
        np.stack([expand(observed), expand(valid.reshape(8, 8)), a, b]),
    ], axis=0).astype(np.float32)
    assert value.shape == (7, 64, 64) and np.isfinite(value).all()
    return value


def depth_labels(native, boxes):
    """TRAIN/EVALUATOR ONLY: visible-ray depth classes and corridor occupancy.

    Classes 0..79 cover [0,8) m in 0.1 m bins; class 80 is finite >=8 m.
    Nonpositive/nonfinite source depth is missing (-100), never a far label.
    Returned depth uses zero at missing rays, paired with an explicit valid mask.
    Occupancy describes the sampled visible surface, not whole-object extent.
    """
    native = np.asarray(native)
    if native.shape != (360, 640) or native.dtype.kind != 'f':
        raise ValueError('Expected native floating optical Z [360,640]')
    yy, xx, a, b = sample_geometry(boxes)
    raw = native[yy, xx]
    valid = np.isfinite(raw) & (raw > 0)
    near = valid & (raw < 8)
    classes = np.full((64, 64), -100, dtype=np.int64)
    classes[near] = np.minimum(np.floor(raw[near] / .1).astype(np.int64), 79)
    classes[valid & ~near] = 80
    depth = np.where(valid, raw, 0).astype(np.float32)
    occupancy = (valid & (depth >= .3) & (depth <= 3)
                 & (np.abs(a * depth) <= .3) & (b * depth >= -.2) & (b * depth <= .9))
    return dict(class_index=classes, depth=depth, occupancy=occupancy, valid=valid)
