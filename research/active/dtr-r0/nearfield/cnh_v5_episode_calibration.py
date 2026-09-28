"""Calibrate alarm thresholds by false episodes per simulated empty minute (EXPLORE fast lane).

For each arm/group, the threshold is chosen on calib units only as the lowest candidate whose
calib false-episode rate is <= the target; audit units then report tiny timely rate and the
realised false-episode rate. Uses saved frozen predictions (v5 and the robustness sweep); no
model rerun, training or audit threshold selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_v5_robustness as R

TARGETS = (1., 2., 5., 10.)


def episodes(p, alarms):
    group_empty = ~(p['y'] == 1).any(axis=(1, 2))
    ga = alarms[group_empty].any(axis=2)
    starts = ga & ~np.concatenate((np.zeros((len(ga), 1), bool), ga[:, :-1]), axis=1)
    return starts.sum()/(ga.size/5./60.), int(starts.sum()), ga.size/5./60.


def calibrate(cal, arm, target):
    empty = ~(cal['y'] == 1).any(axis=(1, 2))
    peaks = np.unique(cal[arm][empty].max(axis=2))
    best = float(np.nextafter(peaks[-1], np.inf))
    # Episode counts are not strictly monotone in threshold; lower the threshold from the top
    # and stop at the first candidate that exceeds the target.
    for thr in peaks[::-1]:
        rate, _, _ = episodes(cal, cal[arm] >= thr)
        if rate <= target:
            best = float(thr)
        else:
            break
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--robustness', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    R.setup(a.data)
    import cnh_v5_evaluate as E
    folders = {'snr6': a.data/'predictions'}
    folders.update({c: a.robustness/c for c in R.CONDITIONS if c != 'snr6' and (a.robustness/c).exists()})
    report = dict(scope='EXPLORE fast lane; thresholds from calib false episodes per simulated empty minute; '
                        'controlled generator only; minutes are 9-frame 5 Hz group-empty sequences, not real use',
                  targets=TARGETS, conditions={})
    for cond, folder in folders.items():
        units = R.load(folder)
        splits = {s: sorted(u for u, d in units.items() if str(d['split']) == s) for s in ('calib', 'audit')}
        res = {}
        for g, ix in E.GROUPS:
            cal, aud = (E.pack(units, splits[s], ix) for s in ('calib', 'audit'))
            rows = {}
            for arm in ('A0', 'A2'):
                rows[arm] = []
                for t in TARGETS:
                    thr = calibrate(cal, arm, t)
                    s = E.alert_stats(aud, aud[arm] >= thr)
                    cal_rate = episodes(cal, cal[arm] >= thr)[0]
                    rows[arm].append(dict(target=t, threshold=thr, calib_rate=cal_rate,
                                          audit_rate=s['false_episodes_per_simulated_empty_minute'],
                                          tiny_timely=s['tiny']['timely_count'], tiny_near=s['tiny']['near'],
                                          all_timely=s['all']['timely_count'], all_near=s['all']['near']))
            res[g] = rows
        report['conditions'][cond] = res
        for g in ('HEAD', 'BODY'):
            print(cond, g, ' | '.join(
                f"T{r0['target']:g}: S2 {r0['tiny_timely']}/{r0['tiny_near']}@{r0['audit_rate']:.2f} "
                f"A2 {r2['tiny_timely']}/{r2['tiny_near']}@{r2['audit_rate']:.2f}"
                for r0, r2 in zip(res[g]['A0'], res[g]['A2'])), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'episode_calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
