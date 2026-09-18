"""Render all saved sequence timelines without rerunning predictions or metrics."""
import argparse
import html
import json
from pathlib import Path


METHODS = ('astar', 'raw_hgb', 'astar_hysteresis')


def number(value):
    return 'UNKNOWN' if value is None else f'{value:.2f}'


def render(rows):
    sections = []
    for row in rows:
        frames = row['frames']
        tracks = [('风险走廊或接触', [f['risk_truth'] for f in frames]),
                  ('目标有效 ToF 返回', [f['target_returned'] for f in frames]),
                  ('身体包络接触', [f['contact_truth'] for f in frames])]
        tracks += [(method, [f['predictions'][method]['alert'] for f in frames]) for method in METHODS]
        stripes = []
        for label, values in tracks:
            cells = ''.join('<span class="cell '+('unknown' if v is None else 'yes' if v else 'no')+
                '" title="'+html.escape(f"{frames[i]['id']} | {frames[i]['time_s']:.1f}s | {v}")+'"></span>'
                for i, v in enumerate(values))
            stripes.append(f'<div class="track"><label>{label}</label><div class="cells">{cells}</div></div>')
        table = []
        for method in METHODS:
            m = row['methods'][method]
            table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [method,
                number(m['first_alert_s']), number(m['first_alert_distance_m']),
                number(m['first_correct_alert_s']), number(m['first_correct_alert_lead_before_contact_s']),
                m['missed_complete_event'], m['side_pass_false_alert_segments'],
                number(m['side_pass_false_alert_duration_s']), number(m['release_delay_s']), m['release_status']])+'</tr>')
        sections.append(f'<section><h2>{html.escape(row["sequence_id"])}</h2><p>{row["family"]} / {row["process"]} / layout {row["layout"]}</p>'+
            '<p>投影可见 '+number(row['target_first_visible_s'])+' s ('+row['target_first_visible_status']+') → 目标有效返回 '+
            number(row['first_valid_tof_s'])+' s ('+row['first_valid_tof_status']+') → 接触 '+number(row['contact_first_s'])+
            ' s / 最近通过 '+number(row['nearest_pass_s'])+' s</p>'+''.join(stripes)+
            '<p class="axis">时间从左到右：0.0–7.9 s；每格 0.1 s。悬停查看帧 ID。</p><div class="scroll"><table><tr>'+''.join('<th>'+v+'</th>' for v in
            ['方法','首报 s','首报前向净距 m','首次风险内提醒 s','接触前提前 s','完整事件漏检','旁路误报段','旁路误报 s','解除延迟 s','解除状态'])+'</tr>'+''.join(table)+'</table></div></section>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MZ177 连续过程时间轴</title>
<style>body{font:15px system-ui;background:#f3f6fa;color:#182538;margin:24px;max-width:1600px}section{background:white;padding:20px;margin:22px 0;border:1px solid #d7dfeb;border-radius:8px}h2{font-size:18px}.track{display:flex;align-items:center;margin:6px 0}.track label{width:180px;flex:none}.cells{display:flex;flex:1;gap:1px}.cell{height:18px;flex:1;min-width:2px}.yes{background:#1769aa}.no{background:#e2e8f0}.unknown{background:#d89027}.axis{color:#526078}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #d9e1eb;padding:8px;text-align:left}th{background:#eef3f8}</style>
<h1>24 条连续序列：固定基线时间轴</h1><p>受控 UE Development；名义 10 Hz，非实测传输延迟。蓝色=成立，浅灰=不成立，橙色=UNKNOWN。</p>
<p>可见性是原生包围盒投影代理，不是 RGB 像素可见测量；起始即出现的目标为左删失。有效 ToF 必须是收到的有效 public 返回并由离线 lineage 归属目标。头动仅为静态占用诊断。停步后退的风险解除观测窗口很短，不代表稳定解除。</p>
<p>首报净距是前向几何净距，不是传感器量程。提前量仅对风险内正确提醒计算；UNKNOWN 或漏检不能当作零延迟。所有帧相关，不是独立样本。</p>'''+''.join(sections)+'</html>'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.evaluation/'timelines.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 24 and len({r['sequence_id'] for r in rows}) == 24
    target = a.evaluation/'timelines.html'
    with target.open('x', encoding='utf-8') as f:
        f.write(render(rows))
    print(target.resolve())


if __name__ == '__main__':
    main()
