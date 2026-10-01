"""Consumed v4 Development: learned scores vs conditional signal reference.

No fitting or dataset regeneration. Joins saved evaluator-only conditional signal
references to saved predictions. The reference is not a physical detection bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import cnh_learned_readout as L


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('artifacts.local/work'))
    p.add_argument('--predictions', type=Path, default=Path('artifacts.local/work/cnh-learned-memory-20260928/predictions'))
    a = p.parse_args()
    out = a.root/'cnh-learned-gap-20260928'
    out.mkdir(parents=True, exist_ok=True)
    data = a.root.parent/'evidence/cnh-track-a-scale-v4-20260926-v1'
    learned = a.root/'cnh-learned-readout-20260928'
    units = {}
    for f in sorted((learned/'features-v4').glob('unit*.npz')):
        with np.load(f) as d:
            if str(d['split']) not in ('calib', 'audit'):
                continue
            u = int(f.stem[4:])
            units[u] = {k: d[v] for k, v in dict(y='labels', main='main', w='witness', strata='strata', config='config', frame='frame', split='split').items()}
        with np.load(a.predictions/f'unit{u:03d}.npz') as pred:
            units[u]['NN'] = pred['NN']
            for k in ('y','main','config','frame','strata'):
                assert np.array_equal(pred[k],units[u][k]), (u,k)
        with np.load(data/f'readouts-gpu/primary-mount-10-snr6/unit{u:02d}.npz') as scores:
            units[u]['S2'] = scores['S2__noisy@0.75']
            assert np.array_equal(scores['labels'],units[u]['y'])
            for k in ('config','frame'):
                assert np.array_equal(scores[k],units[u][k])
        d = units[u]
        assert d['S2'].shape == d['NN'].shape == d['y'].shape
    splits = {s: sorted(u for u,d in units.items() if str(d['split']) == s) for s in ('calib','audit')}
    seq = {s: L.sequences(units, keys, ('S2','NN')) for s,keys in splits.items()}
    result = dict(scope='consumed v4 Development; diagnostic only', units={s: len(v) for s,v in splits.items()}, alerts={}, slices={})
    thresholds = {}
    for g, boxes in L.GROUPS:
        for arm in ('S2', 'NN'):
            for b in (.05,.10,.20):
                thr = L.threshold_for_budget(seq['calib'], arm, boxes, b)
                thresholds[g,arm,b] = thr
                result['alerts'][f'{g}|{arm}|{b}'] = dict(threshold=thr, calib=L.summary(L.pairs(seq['calib'],arm,(1,1),boxes,thr)), audit=L.summary(L.pairs(seq['audit'],arm,(1,1),boxes,thr)))
    rows = []
    for u in splits['audit']:
        rr = json.loads((data/f'analysis/ceiling/unit{u}.json').read_text())
        d = units[u]
        ix = {(int(c),int(t)): i for i,(c,t) in enumerate(zip(d['config'], d['frame']))}
        for r in rr:
            n,q = ix[r['config'],r['frame']], r['box']
            assert d['y'][n,q] == 1 and d['main'][n]
            assert str(d['strata'][n,q]) == r['stratum']
            assert np.isclose(d['S2'][n,q],r['s2'],rtol=1e-6,atol=1e-6)
            r['nn'] = float(d['NN'][n,q])
        rows.extend(rr)
    result['oracle_units'] = len({r['unit'] for r in rows})
    for g,boxes in L.GROUPS:
        for st in ('all','tiny'):
            selected = [r for r in rows if r['box'] in boxes and (st=='all' or r['stratum']==st)]
            for field,edges in [('z_mf4',[0,2,5,10,float('inf')]),('range_m',[0,1,2,3,float('inf')])]:
                bins=[]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    rs=[r for r in selected if lo<=r[field]<hi]
                    row=dict(bin=f'[{lo},{hi})',n=len(rs))
                    if rs:
                        row['conditional_oracle_5pct_frame'] = float(np.mean(ndtr(np.array([r['z_mf4'] for r in rs])-1.6448536269514722)))
                        for arm,k in [('S2','s2'),('NN','nn')]:
                            for b in (.05,.10,.20):
                                row[f'{arm}_detected_budget_{b}']=sum(r[k]>=thresholds[g,arm,b] for r in rs)
                    bins.append(row)
                result['slices'][f'{g}|{st}|{field}']=bins
    result['limitations'] = ['Visible positive frames only, not event timeliness.', 'Oracle 5% per-template per-frame false positive probability is NOT comparable to calibrated sequence-query false-alert budgets.', 'Oracle uses known target template and true pose; learned input uses noisy transported 4-frame z plus current z and whole-scene context. Neither is a physical upper bound on the other.', 'Oracle ignores occluded-background changes; noise/signal model is uncalibrated.', 'Existing v3 rows not joined: uses v4 saved oracle and v4 frozen-model predictions only.']
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    conclusions = []
    for g,_ in L.GROUPS:
        bs=result['slices'][f'{g}|tiny|z_mf4']
        n=sum(r['n'] for r in bs)
        hit=sum(r.get('NN_detected_budget_0.1',0) for r in bs)
        weakmiss=bs[0]['n']-bs[0].get('NN_detected_budget_0.1',0)
        strong=sum(r['n'] for r in bs[2:])
        nnstrong=sum(r.get('NN_detected_budget_0.1',0) for r in bs[2:])
        s2strong=sum(r.get('S2_detected_budget_0.1',0) for r in bs[2:])
        conclusions.append(f'{g} 小目标检出 {hit}/{n}；漏检中 z4<2 为 {weakmiss}/{n-hit}；强信号 z4≥5 检出 NN {nnstrong}/{strong}、S2 {s2strong}/{strong}。')
    lines = ['# 学习读出差距诊断（v4，已消费 Development）', '',
             '结论：'+' '.join(conclusions)+' 这些分层定位当前读出的缺口，不证明物理上限。', '',
             '使用冻结三种子学习读出及现存 v4 信号参考。按当前帧可见正例分层；不训练、不选模型、不重新合成传感数据。', '',
             '|组别 / 小目标分层|可见正例数|S2 检出数 / 召回|学习读出检出数 / 召回|条件理想参考召回*|',
             '|---|---:|---:|---:|---:|']
    for g,_ in L.GROUPS:
        for field,label in [('z_mf4','z4'),('range_m','距离 m')]:
            for r in result['slices'][f'{g}|tiny|{field}']:
                n=r['n']
                if n:
                    s,nn=r['S2_detected_budget_0.1'],r['NN_detected_budget_0.1']
                    lines.append(f"|{g} {label} {r['bin']}|{n}|{s} / {s/n:.3f}|{nn} / {nn/n:.3f}|{r['conditional_oracle_5pct_frame']:.3f}|")
    lines += ['', 'S2 和学习读出使用相同 calib 序列×查询盒假警预算 10%，下列是假警实际计数（分母为空序列×查询盒）：']
    for g,_ in L.GROUPS:
        for arm in ('S2','NN'):
            x=result['alerts'][f'{g}|{arm}|0.1']
            c,d=x['calib'],x['audit']
            lines.append(f"- {g} {arm}：calib {round(c['false_alert_rate']*c['empty_pairs'])}/{c['empty_pairs']}；audit {round(d['false_alert_rate']*d['empty_pairs'])}/{d['empty_pairs']} = {d['false_alert_rate']:.3f}。")
    lines += ['', '*条件理想参考为 Φ(z_mf4−1.645)，其 5% 是已知模板逐帧名义假警率，与上表算法的 10% 序列假警预算口径不同，不能相减为可实现提升或判定接近物理上限。已知真位姿/目标模板的参考只计目标自身回波；网络使用带噪自运动四帧输运及当前帧、全场背景。忽略背景遮挡变化、模拟器未标定、仅可见正例、已消费数据共同限制结论。5%/20% 预算及全部尺寸切片见 result.json。', '', '下一步：据弱信号/强信号切片定位现有模型漏检；另用记忆实验检查离开视场后的证据，不将当前诊断升级为正式结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result['slices'],indent=1))


if __name__ == '__main__':
    main()
