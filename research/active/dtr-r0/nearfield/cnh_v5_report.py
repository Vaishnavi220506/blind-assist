"""Render frozen v5 outputs without changing scores, thresholds or decisions."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    a = p.parse_args()
    r = json.loads((a.data/'analysis/v5_results.json').read_text())
    audit_n = len(r['split_units']['audit'])
    checks = []
    for g in ('HEAD', 'BODY'):
        assert r['AP'][g]['units'] == audit_n, 'AP silently excluded audit units'
        checks.append(f'{g}: all {audit_n} audit units included in AP')
        assert r['guard_only'][g]['calib']['false_pair_rate'] <= .01
        for budget in (.01,.05,.10,.20):
            for arm in ('A0','A1','A2'):
                row = r['working_points'][f'{g}|{arm}|{budget:.2f}']
                assert row['calib']['false_pair_rate'] <= budget
            a2 = r['working_points'][f'{g}|A2|{budget:.2f}']['audit']
            a3 = r['working_points'][f'{g}|A3|{budget:.2f}']['audit']
            for subset in ('all','tiny'):
                assert a3[subset]['near'] == a2[subset]['near']
                assert a3[subset]['never_count'] <= a2[subset]['never_count']
    for strata in r['strong_BODY'].values():
        for row in strata['working_points'].values():
            assert row['detected']['A3'] - row['detected']['A2'] == row['rescued_A2_miss']
    assert r['primary_pass'] == all(r['comparisons'][g]['A2-A0']['ci95'][0] > 0 for g in ('HEAD','BODY'))
    checks += ['All calibration working-point budgets and fixed guards satisfied',
               'A3 near-event detections are a superset; no assumption that first-hit timeliness is monotone',
               'Strong-frame rescue counts reconcile with A3-A2 detections', 'Primary boolean matches frozen CI rule']
    (a.data/'analysis/acceptance_checks.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2))
    lines = ['# CNH v5 冻结复现结果', '',
             f"**主判据：{'通过' if r['primary_pass'] else '未通过'}。** 新种子族 `{r['family']}`；calib {len(r['split_units']['calib'])}单位，audit {audit_n}单位。A2−A0 的 HEAD/BODY 配对95%区间按生成前规则同时判断；未重训、未调参。", '',
             '| 方法 | HEAD 宏 AP | BODY 宏 AP | HEAD / BODY 胜过 A0 的单位数 |',
             '|---|---:|---:|---|']
    for arm in ('A0','A1','A2','A3'):
        wins = []
        for g in ('HEAD','BODY'):
            rows = list(r['AP'][g]['per_unit'].values())
            wins.append(f"{sum(x[arm]>x['A0'] for x in rows)}/{len(rows)}")
        lines.append(f"| {arm} | {r['AP']['HEAD']['macro'][arm]:.4f} | {r['AP']['BODY']['macro'][arm]:.4f} | {' / '.join(wins) if arm!='A0' else '基线'} |")
    lines += ['', 'A0=S2；A1=冻结三种子NN；A2=A1后固定五分数因果平滑；A3=A2 OR固定1%校准预算S2支路。A3 AP使用冻结的guard置顶排序编码，只作描述。']
    for g in ('HEAD','BODY'):
        d = r['comparisons'][g]['A2-A0']
        lines.append(f"- {g} A2−A0：{d['delta']:+.4f}，95%CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]；胜/平/负 {d['wins']}/{d['ties']}/{d['losses']}。")
    lines += ['', '![提醒曲线](timely_vs_false_episodes.png)', '',
              '横轴是每模拟空场景分钟的连续误报段数：5Hz、每config评估9帧，组内3查询盒合并；每个短片段首帧报警另计一段。只取该组全部查询始终无正例的config。此横轴不是实测每分钟提醒次数；短片段重置与连续时间有区别。点由calib阈值映射到audit，不在audit选择阈值；误报段可能随阈值降低合并，曲线不强制单调。', '',
              '## 预先固定的四工作点', '',
              '单元格：**小目标及时率 / 每模拟空分钟误报段数 / audit空查询对误报率**。A3沿用A2阈值后直接OR，不重新凑相同预算。']
    for g in ('HEAD','BODY'):
        den = r['working_points'][f'{g}|A0|0.10']['audit']
        lines += ['', f"### {g}", '',
            f"小目标近事件 {den['tiny']['near']}；空查询对 {den['empty_pairs']}；全空config {den['empty_group_sequences']}，{den['empty_group_frames']}帧 / 5Hz = {den['empty_group_minutes']:.3f}模拟空分钟。", '',
            '| calib预算 | A0 | A1 | A2 | A3 |', '|---|---|---|---|---|']
        for budget in (.01,.05,.10,.20):
            cells = []
            for arm in ('A0','A1','A2','A3'):
                x = r['working_points'][f'{g}|{arm}|{budget:.2f}']['audit']
                cells.append(f"{x['tiny']['timely']:.1%} / {x['false_episodes_per_simulated_empty_minute']:.2f} / {x['false_pair_rate']:.1%}")
            lines.append(f"| {budget:.0%} | " + ' | '.join(cells) + ' |')
    lines += ['', '## BODY 强信号补报（z_mf4≥5、当前可见正帧）', '',
        '| 子集 | 正帧分母 | A0 / A1 / A2 / A3 检出（10%工作点） | A3补回A2 |', '|---|---:|---|---:|']
    for name in ('all','tiny'):
        x = r['strong_BODY'][f'{name}|0.0-inf']; w = x['working_points']['0.10']
        lines.append(f"| {name} | {x['n']} | " + ' / '.join(str(w['detected'][arm]) for arm in ('A0','A1','A2','A3')) + f" | {w['rescued_A2_miss']} |")
    sel = r['demo_guard_selection']
    lines += ['', f"A3满足冻结的仿真演示选择规则：**{sel['eligible_in_simulation']}**（需补回至少1个BODY强信号A2漏检，且两组guard本身audit空对误报率均≤1%）。这不是主判据或安全性检验。"]
    for g in ('HEAD','BODY'):
        x = r['guard_only'][g]['audit']
        lines.append(f"- {g} guard自身：{x['false_pairs']}/{x['empty_pairs']} = {x['false_pair_rate']:.2%}。")
    lines += ['', '## 范围与执行记录', '',
        '冻结提交2d64dc20，先冻结后生成。启动器预建目录冲突发生在任何新样本生成前；695b5fa8只移除预建目录，原失败日志/源码和同种子恢复记录保留，未改配置、判据或预算。生成器/既有读出来自9b4e68ae隔离快照；源码、模型身份、数据门槛和CPU/GPU对账见任务根收据。',
        '这是用户指定EXPLORE通道中的新种子冻结复现，只支持同一未充分标定生成器的分布内结论。v4保持已消费，不把v5推广为真实准确率、物理上限或安全能力；强信号正帧不等于大箱子类别。本轮没有把旧S3列为比较臂，不扩写成v5已证实超越S3。因设备暂不方便，演示交付历史真实输入回放和程序；实时采集、物理配准、曝光到提醒延迟未验证，真实A3亦未校准。', '',
        '回放证据更正：02段整段有物体、无负帧，不能判定误报或选择性；迁移未成立的主证据改为04恢复背景中HEAD左/中均128/128触发，03运动段作对照，逐盒计数见H3回放报告。“空”指无新增物体，不等于几何空查询盒，不能当真实应用假警率。候选原因尚未分离：真实回放减静止场景均值并用经验方差，训练输入减串扰项并用含环境光的噪声底；64cm桌面柜面仍在名义查询范围内。未事后调阈值，回放不能证明避障效果。', '',
        '完整计数、阈值、逐单位AP、描述性比较及距离分层见 [v5_results.json](v5_results.json)；[接受前检查](acceptance_checks.json)；[仿真演示阈值](demo_thresholds.json)。']
    (a.data/'analysis/REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(a.data/'analysis/REPORT.md')


if __name__ == '__main__':
    main()
