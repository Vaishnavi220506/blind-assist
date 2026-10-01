"""Render existing Development evidence; no model training or evaluation tuning.

Run with the project plotting Python. Optional --trial-results takes the generated
seed_metrics.template.json format and summarizes matched baseline/variant seeds.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-evidence-figures-20261001'
COLORS = ['#3974a8', '#d58737', '#8d65ad', '#499883', '#263d55']
ARMS = ('UB0', 'UB1', 'UB2', 'UB3', 'UB4')
LABELS = ('覆盖至 B0', '覆盖至 B1', '覆盖至 B2', '覆盖至 B3', 'FULL 全覆盖')
COND = ('no_panel', 'B4')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def style():
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
        'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9ba8b4', 'axes.labelcolor': '#34465a', 'text.color': '#23384d',
        'xtick.color': '#34465a', 'ytick.color': '#34465a', 'svg.fonttype': 'none'})


def savefig(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    plt.close(fig)


def verify_and_reference(sweep):
    """Recompute all 100 arm/group/cell metrics from persisted query ledger."""
    rows = [json.loads(s) for s in (WORK/'cnh-coverage-sweep-20260930/sample_ledger.jsonl').read_text().splitlines()]
    assert len(rows) == 11520 and len({r['unit'] for r in rows}) == 144
    references = []
    for kind in ('random', 'corner'):
        for entry in sweep[kind]['table']:
            a, b = entry['arm'], entry['bin']
            aucs, bers, fn, fp = [], [], 0, 0
            for group in ('HEAD', 'BODY'):
                rr = [r for r in rows if (r['kind'], r['bin'], r['group']) == (kind, b, group)]
                y = np.array([r['label'] for r in rr], bool)
                score = np.array([r['scores'][a] for r in rr])
                pred = np.array([r['pred'][a] for r in rr], bool)
                p, n = int(y.sum()), int((~y).sum())
                assert (p, n) == (144, 432)
                fni, fpi = int((y & ~pred).sum()), int((~y & pred).sum())
                aucs.append(float((rankdata(score)[y].sum()-p*(p+1)/2)/(p*n)))
                bers.append(.5*(fni/p+fpi/n))
                fn += fni; fp += fpi
            assert abs(np.mean(aucs)-entry['auc']) < 1e-12
            assert abs(np.mean(bers)-entry['ber']) < 1e-12
            references.append(dict(dataset='coverage_sweep_83000_83143', ensemble_seeds=3,
                kind=kind, bin=b, arm=a, n_units=144, P=288, N=864,
                auc_macro_head_body=entry['auc'], BER=entry['ber'], FN=fn, FP=fp,
                FNR=fn/288, FPR=fp/864, role='historical_reference_only'))
    write(OUT/'historical_reference.json', references)
    return references


def coverage_figure(sweep):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.81, wspace=.28, hspace=.40)
    fig.suptitle('训练覆盖扩大后，远外推退化收回；距离本身仍不能解释全部差异', fontsize=18, x=.065, ha='left', y=.975)
    fig.text(.065, .925, '面积轴扫描 · 五组 CVR 读出 · 每格 144 个评估单位 · HEAD / BODY 指标取平均', fontsize=11)
    for row, kind in enumerate(('random', 'corner')):
        for col, (metric, title) in enumerate((('auc', 'AUC ↑'), ('ber', '冻结工作点 BER ↓'))):
            ax = axes[row, col]
            for i, a in enumerate(ARMS):
                cells = [t for t in sweep[kind]['table'] if t['arm'] == a]
                x = np.array([t['bin'] for t in cells])
                y = np.array([t[metric] for t in cells])*(100 if metric == 'ber' else 1)
                # Dashed segments end outside this arm's training support.
                ax.plot(x, y, '--', color=COLORS[i], alpha=.85, lw=1.5)
                ax.plot(x[:i+1], y[:i+1], '-', color=COLORS[i], lw=2)
                ax.scatter(x, y, s=24, color=COLORS[i], zorder=3)
            ax.set_title(('随机面板' if row == 0 else '同侧近距面板')+' | '+title, loc='left', fontsize=12)
            ax.set_xticks(range(5), ['B0\n.03–.09', 'B1\n.09–.27', 'B2\n.27–.8', 'B3\n.8–2.4', 'B4\n2.4–7'])
            ax.set_xlabel('评估面积区间 / m²')
            ax.set_ylabel('AUC' if metric == 'auc' else 'BER / %')
            ax.set_ylim((.50, 1) if metric == 'auc' else (8, 48))
            ax.grid(axis='y', color='#e3e8ed', lw=.8)
        ax = axes[row, 2]
        for i, a in enumerate(ARMS[:-1]):
            cells = [t for t in sweep[kind]['table'] if t['arm'] == a and t['distance'] > 0]
            x = np.array([t['distance'] for t in cells])
            y = np.array([t['relative_loss'] for t in cells])
            ci = np.array([t['relative_loss_ci'] for t in cells])
            ax.plot(x, y, '-o', color=COLORS[i], ms=4, lw=1.7)
            ax.errorbar(x, y, yerr=[y-ci[:, 0], ci[:, 1]-y], fmt='none', color=COLORS[i], alpha=.50, capsize=3)
            for xx, yy, t in zip(x, y, cells):
                offset = ((6, -14), (6, 23), (6, 11), (25, -1))[i] if xx == 1 else (4, (5, 13, -12, -16)[i])
                ax.annotate('B'+str(t['bin']), (xx, yy), xytext=offset, textcoords='offset points', fontsize=8, color=COLORS[i])
        ax.axhline(0, color='#9ba8b4', lw=1)
        ax.set_title('相对 FULL 的 AUC 损失 ↓', loc='left', fontsize=12)
        ax.set_xticks(range(1, 5)); ax.set_xlim(.8, 4.4); ax.set_ylim(-.04, .39)
        ax.set_xlabel('超出覆盖上界的区间数 d'); ax.set_ylabel('AUC(FULL) − AUC(本臂)')
        ax.grid(axis='y', color='#e3e8ed', lw=.8)
    handles = [plt.Line2D([], [], color=c, marker='o', lw=2, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .895), ncol=5, frameon=False)
    fig.text(.065, .09, '实线：覆盖内；虚线：外推。右栏误差线为原扫描的单位重采样 95% 区间，仅含评估抽样，不含训练波动。', fontsize=10)
    fig.text(.065, .047, '每格每组 P=144、N=432；三种子集成。BER = ½(FNR + FPR)，须与漏报、误报分开解读。仅为当前生成器的 Development。', fontsize=10)
    savefig(fig, 'coverage_performance')


def separability_figure(old, shallow):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(left=.065, right=.98, bottom=.20, top=.80, wspace=.27, hspace=.37)
    fig.suptitle('表示探针与模型输出：浅边界更难，大墙下读出信号变弱', fontsize=19, x=.065, ha='left', y=.975)
    fig.text(.065, .924, '特权可分性诊断 · 每条件 24 个单位、每场景 12 次噪声采样 · 柱高为中位数，误差线为单位分布 Q25–Q75', fontsize=11)
    chart_data = []
    for col, margin in enumerate((.015, .045, .12)):
        for row in range(2):
            ax = axes[row, col]
            names = ('raw8', 'z1_8', 'voxel15') if row == 0 else (('logit15', 'logit5') if col == 0 else ('logit',))
            labels = ('原始 8 帧', 'z1 8 帧', '体素末帧') if row == 0 else (('末帧', '实际五帧加权') if col == 0 else ('末帧（逐对翻转方向）',))
            for i, cond in enumerate(COND):
                stats = []
                for name in names:
                    old_name = dict(raw8='raw_hist', z1_8='z1', voxel15='voxel').get(name, name)
                    st = shallow['summary'][cond]['probes'][name] if col == 0 else old['summary'][f'{cond}|{margin}'][old_name]
                    stats.append(st)
                    chart_data.append(dict(margin=margin, condition=cond, representation=name, **st,
                        direction='fixed_high_is_inside' if col == 0 and row == 1 else 'clean_reference_per_pair',
                        error_bars='unit_Q25_Q75_not_confidence_interval'))
                y = np.array([st['median'] for st in stats]); lo = np.array([st['q25'] for st in stats]); hi = np.array([st['q75'] for st in stats])
                x = np.arange(len(names))+(i-.5)*.32
                ax.bar(x, y, width=.30, color=('#67869f', '#c87943')[i], alpha=.90,
                       hatch='///' if row == 1 and col > 0 else None)
                ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt='none', color='#40556a', capsize=4, lw=1)
                for xx, yy in zip(x, y):
                    ax.annotate(f'{yy:.2f}', (xx, yy), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
            ax.set_xticks(range(len(names)), labels)
            ax.set_ylabel('匹配方向 d′' if row == 0 else ('固定方向 d′' if col == 0 else '参考方向 d′'))
            ax.set_title(f'{margin*100:g} cm | '+('表示探针' if row == 0 else '模型输出'), loc='left', fontsize=12)
            ax.axhline(0, color='#9ba8b4', lw=1); ax.grid(axis='y', color='#e3e8ed', lw=.8); ax.set_axisbelow(True)
            ax.margins(y=.20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in zip(('#67869f', '#c87943'), ('无墙', 'B4 同侧近距大墙'))]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .888), ncol=2, frameon=False)
    fig.text(.065, .14, '1.5 cm：严格跨墙配对、logit 方向固定。4.5 / 12 cm：跨墙条件目标重新抽样，logit 按每对无噪声参考差翻转（斜线柱）。', fontsize=10)
    fig.text(.065, .096, '各面板纵轴尺度不同；不同表示、边距与实验不连成“信息损失链”。Q25–Q75 是场景分布，不是均值或中位数的置信区间。', fontsize=10)
    fig.text(.065, .052, '特权方向知道两个候选位置；探针不是可达上限。1.5 cm 大墙外侧目标与墙重叠 11/24；转弯历史标签会变。仅为仿真 Development。', fontsize=10)
    write(OUT/'separability_chart_data.json', chart_data)
    savefig(fig, 'separability_panels')


def historical_table(references):
    lines = ['# 新方法收益与代价比较', '',
        '以下原模型是历史参照，不能代替新试点同轮、同数据、同配方重训的基线。所有数值来自持久化结果及逐查询记录。', '',
        '## 历史扫描参照（FULL 三种子集成）', '',
        '每格 144 个单位，HEAD/BODY 各 P=144、N=432。AUC 为两组平均；计数合并两组，FNR/FPR 分母分别为 288/864。', '',
        '|场景|面积|AUC|BER|漏报 / 288|误报 / 864|FNR|FPR|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in references:
        if r['arm'] == 'UB4':
            lines.append(f"|{'随机' if r['kind']=='random' else '同侧近距'}|B{r['bin']}|{r['auc_macro_head_body']:.3f}|{100*r['BER']:.2f}%|{r['FN']}|{r['FP']}|{100*r['FNR']:.2f}%|{100*r['FPR']:.2f}%|")
    lines += ['', '## 新试点待填（同轮基线与方法逐种子）', '',
        '|模型|种子|区域 / 高度查询|独立单位|P / N|AUC|BER|FN / FP|FNR / FPR|训练秒 / 参数数 / 推理成本|',
        '|---|---|---|---:|---|---:|---:|---|---|---|',
        '|同轮重训基线|待运行|B3、B4、其他区域分别填；HEAD/BODY 分开|待填|待填|—|—|—|—|待测|',
        '|分辨率臂|待运行|与基线同一评估集合|待填|待填|—|—|—|—|待测|',
        '|数据配比臂|待运行|与基线同一评估集合|待填|待填|—|—|—|—|待测|', '',
        '生成的 seed_metrics.template.json 提供最小接入格式。区域定义、样本角色、阈值规则、种子配对在试点自身方案中确定；本表不增设通过门槛。', '',
        '- 逐种子保存 HEAD/BODY AUC 与 FN/FP/P/N，不把不同区域 AUC 混成总体 AUC。跨区域总体 AUC 需由原始分数另算。',
        '- 新臂与同轮基线用相同评估单位、相同种子编号及校准规则；方法阈值在自己的校准数据上按共同规则确定，不按评估结果选择。',
        '- 汇总同时给均值、最小—最大值和逐种子差。这里只显示种子离散程度，不构造覆盖训练不确定性的置信区间。',
        '- 误报率是当前评估构成下的查询误报比例，不能换算真实提醒次数。参数量、训练时间与推理成本据实填；未测留空。',
        '- 旧墙描述重训含新增结构，约 0.03 的外推差异不能直接当作纯训练波动的标准差。两个具体改法失败也不能推出必须使用多帧；当前输入已经融合多帧。', '']
    (OUT/'method_comparison.md').write_text('\n'.join(lines), encoding='utf8')
    write(OUT/'seed_metrics.template.json', dict(dataset_id=None, evaluation_unit_ids=None,
        region_definitions=None, calibration_rule=None, notes='用真实记录替换空字段；records 逐模型/种子/区域/HEAD或BODY填写。每条记录含实际单位列表。',
        records=[dict(model='baseline', seed=None, region=None, group=None, n_units=None, unit_ids=None,
            P=None, N=None, FN=None, FP=None, AUC=None, threshold_source=None,
            train_s=None, parameter_count=None, inference_ms_per_query=None,
            inference_benchmark=None)]))


def trial_summary(path):
    """No pooling AUC, threshold selection or seed-level inferential claims."""
    data = read(path); records = data['records']
    assert data['dataset_id'] and data['calibration_rule'] and data['evaluation_unit_ids'] and data['region_definitions']
    eval_ids = set(data['evaluation_unit_ids'])
    assert len(eval_ids) == len(data['evaluation_unit_ids'])
    index = {}
    for r in records:
        key = (r['model'], r['seed'], r['region'], r['group'])
        assert key not in index and r['group'] in ('HEAD', 'BODY')
        assert isinstance(r['seed'], int) and 0 <= r['AUC'] <= 1
        assert isinstance(r['n_units'], int) and r['threshold_source'] and r['region'] in data['region_definitions']
        assert len(set(r['unit_ids'])) == len(r['unit_ids']) == r['n_units'] > 0
        assert set(r['unit_ids']) <= eval_ids
        for k in ('P', 'N', 'FN', 'FP'):
            assert isinstance(r[k], int) and r[k] >= 0
        assert r['P'] > 0 and r['N'] > 0 and r['FN'] <= r['P'] and r['FP'] <= r['N']
        r = dict(r, FNR=r['FN']/r['P'], FPR=r['FP']/r['N'])
        r['BER'] = .5*(r['FNR']+r['FPR']); index[key] = r
    output = []
    groups = sorted({(r['model'], r['region'], r['group']) for r in records})
    for model, region, group in groups:
        selected = sorted((r for k, r in index.items() if (k[0], k[2], k[3]) == (model, region, group)), key=lambda r: r['seed'])
        entry = dict(model=model, region=region, group=group, seeds=[r['seed'] for r in selected], per_seed=selected)
        entry['descriptive'] = {k: dict(mean=float(np.mean([r[k] for r in selected])),
            min=min(r[k] for r in selected), max=max(r[k] for r in selected)) for k in ('AUC', 'BER', 'FNR', 'FPR')}
        if model != 'baseline':
            baseline_keys = {k[1] for k in index if (k[0], k[2], k[3]) == ('baseline', region, group)}
            assert baseline_keys == set(entry['seeds']), 'baseline/variant seed sets differ'
            delta = []
            for r in selected:
                base = index['baseline', r['seed'], region, group]
                assert (r['P'], r['N'], r['n_units']) == (base['P'], base['N'], base['n_units'])
                assert set(r['unit_ids']) == set(base['unit_ids']), 'baseline/variant unit sets differ'
                delta.append(dict(seed=r['seed'], **{k: r[k]-base[k] for k in ('AUC', 'BER', 'FN', 'FP', 'FNR', 'FPR')}))
            entry['paired_delta_vs_baseline'] = delta
        output.append(entry)
    write(OUT/'trial_summary.json', dict(dataset_id=data['dataset_id'], calibration_rule=data['calibration_rule'],
        scope='descriptive paired seeds; no automatic pass/fail or training confidence interval', results=output))
    lines = ['# 新试点逐种子比较', '', f"数据集：{data['dataset_id']}。以下是描述性种子离散程度，不是训练不确定性的置信区间。", '',
        '|模型|区域|查询|种子|AUC|BER|漏报 / P|误报 / N|ΔAUC|ΔBER / pp|ΔFN / ΔFP|',
        '|---|---|---|---:|---:|---:|---|---|---:|---:|---|']
    for entry in output:
        diffs = {r['seed']: r for r in entry.get('paired_delta_vs_baseline', [])}
        for r in entry['per_seed']:
            delta = diffs.get(r['seed'])
            tail = f"{delta['AUC']:+.4f}|{100*delta['BER']:+.2f}|{delta['FN']:+d} / {delta['FP']:+d}" if delta else '—|—|—'
            lines.append(f"|{entry['model']}|{entry['region']}|{entry['group']}|{r['seed']}|{r['AUC']:.4f}|{100*r['BER']:.2f}%|{r['FN']} / {r['P']}|{r['FP']} / {r['N']}|{tail}|")
        desc = entry['descriptive']
        lines.append(f"|{entry['model']}|{entry['region']}|{entry['group']}|均值（范围）|{desc['AUC']['mean']:.4f} ({desc['AUC']['min']:.4f}–{desc['AUC']['max']:.4f})|{100*desc['BER']['mean']:.2f}% ({100*desc['BER']['min']:.2f}–{100*desc['BER']['max']:.2f})|—|—|—|—|—|")
    lines += ['', '|模型|种子|训练秒|参数数|推理 ms / query|测量条件|', '|---|---:|---:|---:|---:|---|']
    costs = set()
    for r in index.values():
        cost = tuple(r.get(k) for k in ('model', 'seed', 'train_s', 'parameter_count', 'inference_ms_per_query', 'inference_benchmark'))
        if cost not in costs:
            costs.add(cost)
            lines.append('|'+ '|'.join('未测' if v is None else str(v) for v in cost)+'|')
    (OUT/'trial_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


def verify_probe_summaries(old, shallow):
    shallow_rows = [json.loads(s) for s in (WORK/'cnh-shallow-boundary-20261001/pairs.jsonl').read_text().splitlines()]
    checked = 0
    for key, summary in old['summary'].items():
        cond, margin = key.split('|')
        for rep, expected in summary.items():
            vals = [r[rep] for r in old['rows'] if r['condition'] == cond and r['margin'] == float(margin)]
            assert len(vals) == 24
            for k, actual in zip(('q25', 'median', 'q75'), np.percentile(vals, [25, 50, 75])):
                assert abs(actual-expected[k]) < 1e-12
            checked += 1
    for cond, summary in shallow['summary'].items():
        for rep, expected in summary['probes'].items():
            vals = [r['probes'][rep]['signed_dprime'] for r in shallow_rows if r['condition'] == cond]
            assert len(vals) == 24
            for k, actual in zip(('q25', 'median', 'q75'), np.percentile(vals, [25, 50, 75])):
                assert abs(actual-expected[k]) < 1e-12
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--trial-results', type=Path)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); style()
    sweep = read(WORK/'cnh-coverage-sweep-20260930/results.json')
    old = read(WORK/'cnh-representation-separability-20260930/results.json')
    shallow = read(WORK/'cnh-shallow-boundary-20261001/results.json')
    refs = verify_and_reference(sweep)
    probe_stats = verify_probe_summaries(old, shallow)
    coverage_figure(sweep); separability_figure(old, shallow); historical_table(refs)
    if args.trial_results:
        trial_summary(args.trial_results)
    write(OUT/'checks.json', dict(status='PASS', query_records=11520, independent_units=144,
        recomputed_arm_group_cells=100, historical_reference_cells=len(refs), recomputed_probe_summaries=probe_stats,
        training=False, inference=False, new_bootstrap=False,
        source_folders=['cnh-coverage-sweep-20260930', 'cnh-representation-separability-20260930', 'cnh-shallow-boundary-20261001']))
    print('PASS: 100 arm/group/cell metrics reproduced; figures and comparison files:', OUT)


if __name__ == '__main__':
    main()
