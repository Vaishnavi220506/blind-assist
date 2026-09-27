"""Anchor simulator parameters on real H3-layout CNH recordings (fast lane, descriptive).

Input: simple-20260927T164234Z-cc4d0e (XIAO + VL53L8CH, 8x8 zones x 16 bins, start 0,
sub-sample 8, 5 Hz, 20 ms/zone, 1 MHz I2C), the same layout as the simulator's H3.
Static background segments (01, 04): cabinet surface, user-measured ~0.63-0.64 m
(origin unconfirmed, treated as 0.635 +/- 0.02 m); reflectance unknown (0.3-0.9 range).
Object segments (02, 03) are used only for status-code statistics.
Values = raw / 2^scaler (UM3183 5.7). Nothing here is a calibration or a truth label.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from cnh_route_sensor import SensorParameters, angular_rays, synthesize_response
from cnh_track_a_v13_sensor import reference_parameters

ROOT = Path('E:/linnan/linnan/artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e')
SEGMENTS = ('01-background-attempt1', '02-object-attempt2', '03-movement-attempt3', '04-restored-attempt4')
STATIC = ('01-background-attempt1', '04-restored-attempt4')
CENTRE = [27, 28, 35, 36]
DIST, DIST_ERR = .635, .02


def load(segment):
    frames, bad = [], 0
    for line in (ROOT/segment/'tof'/'raw.bin').read_text(encoding='utf-8', errors='replace').splitlines():
        if '"cnh_frame"' not in line:
            continue
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if len(f.get('hist_raw', [])) == 64 and all(len(h) == 16 for h in f['hist_raw']):
            frames.append(f)
        else:
            bad += 1
    frames.sort(key=lambda f: f['seq'])
    arr = lambda k: np.array([f[k] for f in frames], float)
    H = np.array([np.array(f['hist_raw'], float)/2.0**np.array(f['hist_scaler'], float) for f in frames])
    A = np.array([np.array(f['ambient_raw'], float)/2.0**np.array(f['ambient_scaler'], float) for f in frames])
    seq = arr('seq')
    return dict(H=H, A=A, D=arr('distance_mm'), S=arr('target_status').astype(int), seq=seq, ms=arr('ms'),
                bad_lines=bad, missing_seq=int((np.diff(seq)-1).clip(0).sum()))


def noise_regimes(d):
    """Background (far bins 8-13 vs ambient), target-bin superlinearity, bin-boundary jitter."""
    H, A = d['H'], d['A'].mean(0)
    m, v = H.mean(0), H.var(0, ddof=1)
    kb = np.polyfit(A, v[:, 8:14].mean(1), 1)
    sel = m[:, 2] > 50
    slope_bin2 = np.polyfit(np.log(m[sel, 2]), np.log(v[sel, 2]), 1)[0]
    S = H[:, :, 1:4].sum(2)
    slope_sum = np.polyfit(np.log(S.mean(0)[sel]), np.log(S.var(0, ddof=1)[sel]), 1)[0]
    c12 = np.array([np.corrcoef(H[:, z, 1], H[:, z, 2])[0, 1] for z in range(64)])
    return dict(background_var_vs_ambient=[float(x) for x in kb],
                corr_far_var_ambient=float(np.corrcoef(A, v[:, 8:14].mean(1))[0, 1]),
                target_bin2_loglog_slope=float(slope_bin2), bins1_3_summed_loglog_slope=float(slope_sum),
                corr_bin1_bin2_median=float(np.median(c12[sel])))


def temporal(d):
    H = d['H'][:, :, 8:]                                   # bins 8-15 (2.4-4.8 m): no cabinet return
    R = H-H.mean(0)
    s = R.std(0, ddof=1)
    ok = s > 0
    ac = [float(np.mean(((R[l:]*R[:-l]).mean(0)/np.maximum(s**2, 1e-12))[ok])) for l in (1, 2, 3, 4)]
    T = len(R)//4*4
    S4 = R[:T].reshape(T//4, 4, *R.shape[1:]).sum(1)
    return ac, float(np.median((S4.var(0, ddof=1)/np.maximum(4*s**2, 1e-12))[ok]))


def sim_zone_response(distance, rho):
    """Expected H3 counts [16] for a flat target filling one central zone, per unit signal_counts."""
    _, w = angular_rays(16)
    p = replace(SensorParameters(), signal_counts=1., noise_scale=0.)
    empty = synthesize_response(np.full((8, 8, 256), np.inf), rho, 1., w, params=p, seed=0)['histogram']
    full = synthesize_response(np.full((8, 8, 256), distance), rho, 1., w, params=p, seed=0)['histogram']
    return (full-empty)[3, 3].reshape(16, 8).sum(-1), empty[3, 3].reshape(16, 8).sum(-1)


def main():
    data = {s: load(s) for s in SEGMENTS}
    out = dict(source=str(ROOT), segments={})
    for s, d in data.items():
        ms = d['ms']
        out['segments'][s] = dict(frames=len(d['H']), bad_lines=d['bad_lines'], missing_seq=d['missing_seq'],
                                  rate_hz=float((len(ms)-1)/((ms[-1]-ms[0])/1000)))
    # 1. static check and noise structure
    regimes = {}
    for s in STATIC:
        d = data[s]
        ac, vr = temporal(d)
        regimes[s] = noise_regimes(d)
        out['segments'][s] |= dict(centre_firmware_distance_mm=float(np.median(d['D'][:, CENTRE])),
                                   noise=regimes[s], autocorr_lag1_4=ac, var_ratio_4frame=vr)
    cross = {}
    for fit_s, reg in regimes.items():
        for test_s in STATIC:
            d = data[test_s]
            far = d['H'][:, :, 8:14].var(0, ddof=1).mean(1)
            cross[f'{fit_s}->{test_s}'] = float(np.median(far/np.polyval(reg['background_var_vs_ambient'], d['A'].mean(0))))
    out['background_cross_prediction_obs_over_pred_median'] = cross
    # 2. signal anchor in the simulator's own SNR terms (reference: 8x8 zone, 2 m, rho 0.5).
    # SNR is scale-free in CNH units: s / sqrt(k*s + background), with k the low-signal
    # var/mean slope after subtracting the ambient-predicted background.
    bg = np.concatenate([data[s]['H'] for s in STATIC])
    amb = np.concatenate([data[s]['A'] for s in STATIC]).mean(0)
    m, v = bg.mean(0), bg.var(0, ddof=1)
    kb = np.mean([regimes[s]['background_var_vs_ambient'] for s in STATIC], 0)
    sel = (m > 3) & (m < 40)
    sel[:, :3] = False
    k = ((v-np.polyval(kb, amb)[:, None])[sel]/m[sel])
    centre = bg[:, CENTRE]
    win = slice(1, 5)
    s0 = float(centre[:, :, win].sum(2).mean())
    far_var = float(centre[:, :, 8:14].var(0, ddof=1).mean())
    rows = []
    for kq in (25, 50, 75):
        kk = float(np.percentile(k, kq))
        for rho in (.3, .5, .7, .9):
            for dist in (DIST-DIST_ERR, DIST, DIST+DIST_ERR):
                sig = s0*(dist/2.)**2*(.5/rho)
                rows.append(dict(k_percentile=kq, k=kk, rho=rho, distance_m=dist, reference_signal=float(sig),
                                 reference_snr=float(sig/np.sqrt(kk*sig+far_var*7/8))))
    snrs = [r['reference_snr'] for r in rows]
    out['anchor'] = dict(centre_window_mean=s0, centre_far_bin_var=far_var, low_signal_k_iqr=[float(np.percentile(k, q)) for q in (25, 50, 75)],
                         k_samples=int(sel.sum()), reference_snr_range=[float(min(snrs)), float(max(snrs))], rows=rows,
                         sim_tiers='SNR3/6/12 (reference_parameters)')
    # apparent range offset: simulator distance whose bin1/bin2 split matches the real one
    real_ratio = float(centre[:, :, 1].mean()/centre[:, :, 2].mean())
    grid = np.arange(.60, .80, .0025)
    ratios = [sim_zone_response(d, .5)[0] for d in grid]
    match = float(grid[int(np.argmin([abs(u[1]/u[2]-real_ratio) for u in ratios]))])
    out['range_offset'] = dict(real_bin1_over_bin2=real_ratio, sim_matching_distance_m=match,
                               apparent_offset_m=match-DIST, tape_uncertainty_m=DIST_ERR)
    real_shape = centre.mean((0, 1))
    real_win = s0
    # 3. crosstalk and peak shape
    unit5, empty_unit = sim_zone_response(DIST, .5)
    out['crosstalk'] = dict(real_bin0_over_window=float(centre[:, :, 0].mean()/real_win),
                            sim_bin0_over_window_rho05=float(empty_unit[0]/unit5[win].sum()),
                            real_shape_bins1_4_normalised=[float(x) for x in real_shape[1:5]/real_shape[1:5].max()],
                            sim_shape_bins1_4_normalised=[float(x) for x in unit5[1:5]/unit5[1:5].max()])
    # 4. status codes
    status = {}
    for s, d in data.items():
        codes, counts = np.unique(d['S'], return_counts=True)
        valid = d['D'] > 0
        strict = ((d['S'] == 5) & valid).mean()
        lenient = (np.isin(d['S'], (5, 6, 9)) & valid).mean()
        status[s] = dict(counts={int(k): int(v) for k, v in zip(codes, counts)},
                         strict_valid_fraction=float(strict), lenient_valid_fraction=float(lenient))
    out['status'] = status
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('h3_real_anchor.json')
    target.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
