"""RGB looming selectivity diagnostic (EXPLORE fast lane, Development).

Question: can class-agnostic, rotation-compensated image expansion (looming) separate
collision-course regions from visible benign structure? Frozen rules (written before any
score or truth statistic was computed): artifacts.local/work/cnh-rgb-looming-20260928/DECISIONS.md

Authority split: observe_* and looming_scores only see grey frames, intrinsics and a noisy
relative rotation. Depth, translation and future poses are read only by truth_cells, except
for the two arms explicitly marked privileged (exact rotation, true-heading FOE).
Data: local SANPO-Synthetic sessions (CC BY 4.0).
"""
import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import rankdata

DATA = Path(__file__).resolve().parents[4] / 'artifacts.local' / 'datasets' / 'sanpo-synthetic-ba-nfo'
S = 2                                   # half resolution
GRID = 8
HALF_FOV = np.radians(22.5)             # 45x45 deg central crop = ToF zone grid
A_GRID = (.25, .35, .5)                 # assumed step per frame gap (m)
Q_GRID = (.5, .9, .98)                  # within-cell pixel quantile
BANDS = ((.3, 1.), (1., 2.), (2., 3.), (3., 5.))
LATERAL = .6
# vertical corridor relative to the camera, y down (metres below camera positive)
VBAND = {'camera_head': (-.2, .9), 'camera_chest': (-.55, .55)}
PATH_MIN = 5.5
MIN_FRAC = .02
MIN_RELIABLE = 20
VARIANTS = ('M2', 'M2norot', 'M2exact', 'M2trueFOE')


# ---------------------------------------------------------------- data access
def sessions():
    rows, count = [], {}
    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        mount = next(c.name for c in d.iterdir() if c.name.startswith('camera_'))
        i = count.get(mount, 0)
        count[mount] = i + 1
        rows.append(dict(session=d.name, mount=mount, index=i, split='even' if i % 2 == 0 else 'odd'))
    return rows


def intrinsics(session):
    detail = json.loads((DATA / session / 'description.json').read_text())['session_camera_details'][0]
    c = detail['left_camera_params']
    return np.array([c['fx'], c['fy'], c['cx'], c['cy']]) / S, float(detail['fps'])


def observe_gray(session, mount, i):
    im = cv2.imread(str(DATA / session / mount / 'left' / 'video_frames' / f'{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (im.shape[1] // S, im.shape[0] // S), interpolation=cv2.INTER_AREA)


def _poses(session, mount):
    a = np.genfromtxt(DATA / session / mount / 'camera_poses.csv', delimiter=',', skip_header=1, usecols=range(1, 8))
    return Rot.from_quat(a[:, 3:]).as_matrix(), a[:, :3]


def _depth(session, mount, i):
    raw = np.frombuffer(gzip.open(DATA / session / mount / 'left' / 'depth_maps' / f'{i:06d}.float16.gz').read(), np.float16)
    h, w = int(raw[0]), int(raw[1])
    return raw[2:].astype(np.float32).reshape(h, w)[S // 2::S, S // 2::S]


def seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, ('cnh-rgb-looming-20260928',) + parts)).encode()).hexdigest()[:16], 16)


def observed_rotation(m_true, session, t, k, fps):
    """Gyro proxy: per-session per-axis bias U(-1,1) deg/s over the gap plus 0.2 deg white noise."""
    bias = np.random.default_rng(seed(session, 'bias')).uniform(-1, 1, 3) * k / fps
    white = np.random.default_rng(seed(session, t, 'white')).normal(0, .2, 3)
    return m_true @ Rot.from_rotvec(np.radians(bias + white)).as_matrix()


# ---------------------------------------------------------------- geometry helpers
def pixel_grid(shape):
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return u + .25, v + .25          # area-downsampled centres in K/2 coordinates


def cell_map(shape, K):
    fx, fy, cx, cy = K
    u, v = pixel_grid(shape)
    hx, hy = fx * np.tan(HALF_FOV), fy * np.tan(HALF_FOV)
    cu = np.floor((u - (cx - hx)) / (2 * hx) * GRID).astype(int)
    cv = np.floor((v - (cy - hy)) / (2 * hy) * GRID).astype(int)
    cells = cv * GRID + cu
    cells[(cu < 0) | (cu >= GRID) | (cv < 0) | (cv >= GRID)] = -1
    return cells


# ---------------------------------------------------------------- truth (evaluator only)
def truth_cells(session, mount, t, K, cells, R, p):
    """Label per cell: 1 positive, 0 benign negative, -1 excluded, -2 empty; None if path too short."""
    path = p[t:]
    seg = np.diff(path[:, :2], axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    total = float(seglen.sum())
    if total < PATH_MIN:
        return None
    keep = seglen > 1e-6
    A, seg, seglen = path[:-1][keep], seg[keep], seglen[keep]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))])[:-1][keep]
    zA, zB = path[:-1, 2][keep], path[1:, 2][keep]
    d = _depth(session, mount, t)
    fx, fy, cx, cy = K
    u, v = pixel_grid(d.shape)
    inside = cells >= 0
    valid = inside & np.isfinite(d) & (d > 0)
    uu, vv, dd = u[valid], v[valid], d[valid]
    X = np.stack([(uu - cx) / fx * dd, (vv - cy) / fy * dd, dd], 1) @ R[t].T + p[t]
    lat = np.full(len(X), np.inf)
    s = np.zeros(len(X))
    dz = np.zeros(len(X))
    for lo in range(0, len(X), 40000):
        x = X[lo:lo + 40000]
        rel = x[:, None, :2] - A[None, :, :2]
        tp = np.clip((rel * seg[None]).sum(-1) / seglen[None] ** 2, 0, 1)
        dist = np.linalg.norm(rel - tp[..., None] * seg[None], axis=-1)
        j = dist.argmin(1)
        r = np.arange(len(x))
        lat[lo:lo + 40000] = dist[r, j]
        s[lo:lo + 40000] = cum[j] + tp[r, j] * seglen[j]
        dz[lo:lo + 40000] = x[:, 2] - (zA[j] + tp[r, j] * (zB[j] - zA[j]))
    lo_v, hi_v = VBAND[mount]
    collide = (lat <= LATERAL) & (s >= .3) & (s <= total - .3) & (-dz >= lo_v) & (-dz <= hi_v)
    cid = cells[valid]
    ncell = np.bincount(cells[inside], minlength=GRID * GRID)
    col5 = np.bincount(cid[collide & (s <= 5)], minlength=GRID * GRID)
    colany = np.bincount(cid[collide], minlength=GRID * GRID)
    near = np.bincount(cid[dd <= 5], minlength=GRID * GRID)
    mins = np.full(GRID * GRID, np.inf)
    m5 = collide & (s <= 5)
    np.minimum.at(mins, cid[m5], s[m5])
    label = np.full(GRID * GRID, -1)
    label[(col5 >= MIN_FRAC * ncell) & (ncell > 0)] = 1
    label[(colany == 0) & (near >= MIN_FRAC * ncell) & (ncell > 0)] = 0
    label[(colany == 0) & (near < MIN_FRAC * ncell)] = -2
    band = np.full(GRID * GRID, -1)
    for b, (lo, hi) in enumerate(BANDS):
        band[(label == 1) & (mins >= lo) & (mins < hi)] = b
    band[(label == 1) & (mins >= BANDS[-1][1] - 1e-9) & (band < 0)] = len(BANDS) - 1
    return label, band, mins


# ---------------------------------------------------------------- method (observation only)
_DIS = None


def _flow(a, b):
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS.calc(a, b, None)


def observe_flow(g_prev, g_cur):
    """Backward flow on current pixels plus forward-backward reliability."""
    bw = _flow(g_cur, g_prev)
    fw = _flow(g_prev, g_cur)
    u, v = pixel_grid(g_cur.shape)
    xp, yp = u - .25 + bw[..., 0], v - .25 + bw[..., 1]
    back = cv2.remap(fw, xp.astype(np.float32), yp.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    err = np.linalg.norm(bw + back, axis=-1)
    reliable = np.isfinite(err) & (err <= 1 + .05 * np.linalg.norm(bw, axis=-1))
    return bw, reliable


def translational_flow(bw, K, M):
    """Forward translational flow at current pixels; M maps previous-camera rays into the current camera."""
    fx, fy, cx, cy = K
    u, v = pixel_grid(bw.shape[:2])
    xp, yp = u + bw[..., 0], v + bw[..., 1]
    ray = np.stack([(xp - cx) / fx, (yp - cy) / fy, np.ones_like(xp)], -1) @ M.T
    z = np.where(np.abs(ray[..., 2]) < 1e-6, 1e-6, ray[..., 2])
    xd, yd = fx * ray[..., 0] / z + cx, fy * ray[..., 1] / z + cy
    return np.stack([u - xd, v - yd], -1)


def estimate_foe(f, reliable, K):
    u, v = pixel_grid(f.shape[:2])
    mag = np.linalg.norm(f, axis=-1)
    sel = reliable & (mag > .3)
    sel &= (np.arange(f.shape[1])[None] % 4 == 0) & (np.arange(f.shape[0])[:, None] % 4 == 0)
    if sel.sum() < 50:
        return np.array([K[2], K[3]])
    x = np.stack([u[sel], v[sel]], 1)
    n = np.stack([-f[sel][:, 1], f[sel][:, 0]], 1) / mag[sel][:, None]
    w0 = mag[sel]
    F = np.array([K[2], K[3]])
    w = w0.copy()
    for _ in range(5):
        A = (n[:, :, None] * n[:, None, :] * w[:, None, None]).sum(0)
        b = (n * (n * x).sum(1, keepdims=True) * w[:, None]).sum(0)
        if abs(np.linalg.det(A)) < 1e-9:
            return np.array([K[2], K[3]])
        F = np.linalg.solve(A, b)
        r = np.abs(((F[None] - x) * n).sum(1))
        w = w0 * np.minimum(1, 10 / np.maximum(r, 1e-9))
    return F


def expansion(f, F):
    u, v = pixel_grid(f.shape[:2])
    rx, ry = u - F[0], v - F[1]
    r2 = rx ** 2 + ry ** 2
    e_rad = (f[..., 0] * rx + f[..., 1] * ry) / np.maximum(r2, 1e-6)
    sm = cv2.GaussianBlur(f, (0, 0), 3)
    div = np.gradient(sm[..., 0], axis=1) + np.gradient(sm[..., 1], axis=0)
    return np.where(r2 >= 64, e_rad, div / 2)


def membership(e, F, K, a, mount):
    fx, fy, cx, cy = K
    u, v = pixel_grid(e.shape)
    tx = np.arctan((u - cx) / fx) - np.arctan((F[0] - cx) / fx)
    ty = np.arctan((v - cy) / fy) - np.arctan((F[1] - cy) / fy)
    dist = a / np.maximum(e, 1e-6)
    mx, my = tx * dist, ty * dist
    lo, hi = VBAND[mount]
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return sig((LATERAL - np.abs(mx)) / .1) * sig((hi - my) / .1) * sig((my - lo) / .1)


def cell_quantiles(pix, reliable, order, starts, qs):
    """pix/reliable flattened in cell order; returns (len(qs), 64)."""
    out = np.zeros((len(qs), GRID * GRID))
    vals = np.where(reliable, pix, np.nan)[order]
    for c in range(GRID * GRID):
        seg = vals[starts[c]:starts[c + 1]]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= MIN_RELIABLE:
            out[:, c] = np.quantile(seg, qs)
    return out


def looming_scores(g_prev, g_cur, K, M_obs, mount, cells, order, starts, M_exact=None, heading=None):
    """All observation-side scores for one frame. M_exact/heading only feed privileged arms."""
    bw, rel = observe_flow(g_prev, g_cur)
    rel_flat = rel.ravel()
    names, rows = [], []
    configs = [('M2', M_obs, None), ('M2norot', np.eye(3), None)]
    if M_exact is not None:
        configs += [('M2exact', M_exact, None), ('M2trueFOE', M_exact, heading)]
    for name, M, Fh in configs:
        f = translational_flow(bw, K, M)
        F = estimate_foe(f, rel, K) if Fh is None else Fh
        e = expansion(f, F)
        epos = np.maximum(e, 0)
        if name == 'M2':
            qv = cell_quantiles(np.linalg.norm(f, axis=-1).ravel(), rel_flat, order, starts, Q_GRID)
            names += [f'M1|q{q}' for q in Q_GRID]; rows += list(qv)
            qv = cell_quantiles(epos.ravel(), rel_flat, order, starts, Q_GRID)
            names += [f'M1b|q{q}' for q in Q_GRID]; rows += list(qv)
        for a in A_GRID:
            qv = cell_quantiles((epos * membership(e, F, K, a, mount)).ravel(), rel_flat, order, starts, Q_GRID)
            names += [f'{name}|a{a}|q{q}' for q in Q_GRID]; rows += list(qv)
    return names, np.array(rows)


def m0_scores(K, cells):
    u, v = pixel_grid(cells.shape)
    out = np.zeros(GRID * GRID)
    for c in range(GRID * GRID):
        m = cells == c
        out[c] = -np.hypot(u[m].mean() - K[2], v[m].mean() - K[3])
    return out


# ---------------------------------------------------------------- per-session job
def session_job(args):
    info, outdir = args
    target = Path(outdir) / f"{info['session']}.npz"
    if target.exists():
        return info['session']
    cv2.setNumThreads(1)
    session, mount = info['session'], info['mount']
    K, fps = intrinsics(session)
    R, p = _poses(session, mount)          # truth + privileged arms only
    n = min(len(p), len(list((DATA / session / mount / 'left' / 'video_frames').glob('*.png'))))
    step = np.median(np.linalg.norm(np.diff(p, axis=0), axis=1))
    k = max(1, int(np.ceil(.25 / max(step, 1e-6) - 1e-9)))
    shape = observe_gray(session, mount, 0).shape
    cells = cell_map(shape, K)
    flat = cells.ravel()
    order = np.argsort(flat, kind='stable')
    starts = np.concatenate([[np.searchsorted(flat[order], 0)], [np.searchsorted(flat[order], c + 1) for c in range(GRID * GRID)]])
    m0 = m0_scores(K, cells)
    rows = dict(t=[], cell=[], label=[], band=[], mins=[])
    scores, names = [], None
    counts = dict(frames_total=0, frames_short_path=0, frames_eval=0, excluded=0, empty=0)
    grays = {}
    for t in range(k, n):
        counts['frames_total'] += 1
        tr = truth_cells(session, mount, t, K, cells, R, p)
        if tr is None:
            counts['frames_short_path'] += 1
            continue
        label, band, mins = tr
        counts['frames_eval'] += 1
        counts['excluded'] += int((label == -1).sum())
        counts['empty'] += int((label == -2).sum())
        for i in (t - k, t):
            if i not in grays:
                grays[i] = observe_gray(session, mount, i)
        M_true = R[t].T @ R[t - k]
        M_obs = observed_rotation(M_true, session, t, k, fps)
        d = R[t].T @ (p[t] - p[t - k])
        heading = np.array([K[0] * d[0] / d[2] + K[2], K[1] * d[1] / d[2] + K[3]]) if d[2] > 1e-3 else np.array(K[2:])
        names, sc = looming_scores(grays[t - k], grays[t], K, M_obs, mount, cells, order, starts, M_true, heading)
        keep = label >= 0
        for c in np.flatnonzero(keep):
            rows['t'].append(t); rows['cell'].append(c); rows['label'].append(label[c])
            rows['band'].append(band[c]); rows['mins'].append(mins[c])
        scores.append(np.vstack([sc, m0[None]])[:, keep].T)
        grays.pop(t - k - 1, None)
    names = (names or []) + ['M0']
    tmp = target.with_name(target.stem + '.partial.npz')
    np.savez_compressed(tmp, scores=np.concatenate(scores) if scores else np.zeros((0, len(names))),
                        names=np.array(names), **{key: np.array(val) for key, val in rows.items()},
                        meta=json.dumps(dict(info, k=k, step_median=float(step), fps_label=fps, **counts)))
    tmp.replace(target)
    return session


# ---------------------------------------------------------------- evaluation
def auc(y, s):
    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load(outdir, mount):
    data = []
    for f in sorted(Path(outdir).glob('*.npz')):
        z = np.load(f)
        meta = json.loads(str(z['meta']))
        if meta['mount'] != mount:
            continue
        data.append(dict(meta=meta, names=list(z['names']), scores=z['scores'], label=z['label'], band=z['band'], t=z['t']))
    return data


def pool(data, split):
    part = [d for d in data if d['meta']['split'] == split and len(d['label'])]
    sid = np.concatenate([np.full(len(d['label']), i) for i, d in enumerate(part)])
    return part, sid, np.concatenate([d['label'] for d in part]), np.concatenate([d['band'] for d in part]), \
        np.concatenate([d['scores'] for d in part])


def evaluate(outdir, mount, B=10000):
    data = load(outdir, mount)
    names = data[0]['names']
    col = {n: i for i, n in enumerate(names)}
    _, _, ye, be, se = pool(data, 'even')
    part, sid, yo, bo, so = pool(data, 'odd')
    chosen = {}
    for fam in ('M0', 'M1', 'M1b') + VARIANTS:
        cands = [n for n in names if n.split('|')[0] == fam]
        best = max(cands, key=lambda n: auc(ye, se[:, col[n]]))
        chosen[fam] = dict(config=best, even_auc=float(auc(ye, se[:, col[best]])))
    rng = np.random.default_rng(seed(mount, 'bootstrap'))
    ns = len(part)
    groups = [np.flatnonzero(sid == i) for i in range(ns)]
    masks = {'all': bo >= -1, '1-3m': (bo == 1) | (bo == 2)}
    for b, (lo, hi) in enumerate(BANDS):
        masks[f'{lo:g}-{hi:g}m'] = bo == b
    def sub(idx, key):
        m = masks[key][idx]
        keep = (yo[idx] == 0) | m
        return idx[keep]
    report = dict(mount=mount, sessions_even=len(data) - ns, sessions_odd=ns, chosen=chosen, bands={})
    for key in masks:
        idx = sub(np.arange(len(yo)), key)
        entry = dict(positives=int((yo[idx] == 1).sum()), negatives=int((yo[idx] == 0).sum()),
                     auc={fam: float(auc(yo[idx], so[idx, col[c['config']]])) for fam, c in chosen.items()})
        report['bands'][key] = entry
    comps = [('M2', 'M0'), ('M2', 'M1'), ('M2', 'M1b'), ('M2exact', 'M2'), ('M2trueFOE', 'M2exact'), ('M2', 'M2norot')]
    boot = {key: {'M2': []} | ({f'{a}-{b}': [] for a, b in comps} if key in ('all', '1-3m') else {}) for key in masks}
    fams = sorted({x for c in comps for x in c})
    for _ in range(B):
        pick = rng.integers(0, ns, ns)
        base = np.concatenate([groups[i] for i in pick])
        for key in masks:
            idx = sub(base, key)
            y = yo[idx]
            if key not in ('all', '1-3m'):
                boot[key]['M2'].append(auc(y, so[idx, col[chosen['M2']['config']]]))
                continue
            vals = {fam: auc(y, so[idx, col[chosen[fam]['config']]]) for fam in fams}
            boot[key]['M2'].append(vals['M2'])
            for a, b in comps:
                boot[key][f'{a}-{b}'].append(vals[a] - vals[b])
    for key in masks:
        report['bands'][key]['ci95'] = {n: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                                        for n, v in boot[key].items()}
    # false-candidate burden at even-calibrated recall points
    burden = {}
    frames_odd = sum(d['meta']['frames_eval'] for d in part)
    for fam in ('M2', 'M1', 'M1b', 'M0', 'M2trueFOE'):
        c = col[chosen[fam]['config']]
        entry = {}
        for rec in (.5, .8):
            thr = np.quantile(se[ye == 1, c], 1 - rec)
            fp = int(((yo == 0) & (so[:, c] >= thr)).sum())
            tp = float(((yo == 1) & (so[:, c] >= thr)).sum() / max(1, (yo == 1).sum()))
            entry[f'recall{int(rec * 100)}'] = dict(threshold=float(thr), odd_recall=tp, benign_cells=fp,
                                                     per_frame=fp / frames_odd, per_minute_at_5hz=fp / frames_odd * 300)
        burden[fam] = entry
    report['burden'] = burden
    report['frames_odd'] = frames_odd
    report['counts'] = {k: int(sum(d['meta'][k] for d in data)) for k in ('frames_total', 'frames_short_path', 'frames_eval', 'excluded', 'empty')}
    report['k'] = {d['meta']['session'][:8]: d['meta']['k'] for d in data}
    report['step_median'] = {d['meta']['session'][:8]: round(d['meta']['step_median'], 3) for d in data}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sessions', nargs='*')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=10000)
    a = ap.parse_args()
    raw = a.out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    if a.evaluate:
        rep = {m: evaluate(raw, m, a.bootstrap) for m in ('camera_head', 'camera_chest')}
        (a.out / 'report.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(json.dumps({m: dict(chosen=r['chosen'], bands={k: (v['auc'], v['positives'], v['negatives']) for k, v in r['bands'].items()}) for m, r in rep.items()}, indent=1))
        return
    todo = [s for s in sessions() if not a.sessions or s['session'][:8] in a.sessions]
    with ProcessPoolExecutor(a.workers) as ex:
        for name in ex.map(session_job, [(s, str(raw)) for s in todo]):
            print('done', name[:8], flush=True)


if __name__ == '__main__':
    main()
