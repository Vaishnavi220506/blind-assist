"""Render saved frozen v5 audit arrays. No inference, training or threshold search."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG, PANEL, TEXT, MUTED = '#08111f', '#111e30', '#ecf4fc', '#91a7be'
CYAN, AMBER, RED = '#42ded0', '#ffc26b', '#ff7286'
GROUPS = [('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5))]
QN = ['头部·左', '躯干·左', '头部·中', '躯干·中', '头部·右', '躯干·右']
TITLES = ['小目标：头部及时提醒', '小目标：躯干及时提醒', 'BODY 组空：观察误报代价', '反例：平滑仍会报晚或漏报']
IDS = ['head_tiny_gain', 'body_tiny_gain', 'empty_comparison', 'a2_counterexample']


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def load_unit(data, u, causal):
    with np.load(data / 'predictions' / f'unit{u:03d}.npz') as f:
        d = {k: f[k] for k in f.files}
    d['A2'] = causal(d['NN'], d['config'], d['frame'], alpha=.5, window=5)
    return d


def select(data, out, causal, thresholds):
    assert (out / 'selection_rules.json').exists(), 'Rules must precede inspection'
    candidates = {k: [] for k in IDS}
    bases = dict.fromkeys(IDS, 0)
    unknown = 0
    for u in range(32, 96):
        d = load_unit(data, u, causal)
        for c in sorted(np.unique(d['config'])):
            idx = np.flatnonzero(d['config'] == c)
            idx = idx[np.argsort(d['frame'][idx])][3:]
            y, w = d['y'][idx], d['w'][idx]
            unknown += int(((y != 0) & (y != 1)).sum())
            for g, qs in GROUPS:
                a0 = d['S2'][idx] >= thresholds[g]['A0']
                a2 = d['A2'][idx] >= thresholds[g]['A2']
                if not (y[:, qs] == 1).any():
                    bases['empty_comparison'] += 1
                    if a0[:, qs].any() or a2[:, qs].any():
                        candidates['empty_comparison'].append(dict(unit=u, config=int(c), query=qs[0], group=g,
                            S2_false_frames=int(a0[:, qs].any(axis=1).sum()), A2_false_frames=int(a2[:, qs].any(axis=1).sum())))
                for q in qs:
                    pos = y[:, q] == 1
                    if not pos.any() or np.min(w[pos, q]) > 1:
                        continue
                    bases['a2_counterexample'] += 1
                    hits0, hits2 = pos & a0[:, q], pos & a2[:, q]
                    def status(hits):
                        if not hits.any():
                            return 'miss'
                        return 'timely' if w[np.flatnonzero(hits)[0], q] >= 1 else 'late'
                    s0, s2 = status(hits0), status(hits2)
                    tiny = Counter(d['strata'][idx, q][pos].tolist()).most_common(1)[0][0] == 'tiny'
                    rec = dict(unit=u, config=int(c), query=q, group=g, tiny=tiny, S2_status=s0, A2_status=s2)
                    if s2 != 'timely':
                        candidates['a2_counterexample'].append(rec)
                    cat = 'head_tiny_gain' if g == 'HEAD' else 'body_tiny_gain'
                    if tiny:
                        bases[cat] += 1
                        if s2 == 'timely' and s0 != 'timely':
                            candidates[cat].append(rec)
    report = dict(selection_rules_sha256=digest(out / 'selection_rules.json'), evaluated_units=64,
                  configs=2048, evaluated_frames_per_config=9, unknown_labels=unknown, categories={})
    for cat, rows in candidates.items():
        rows.sort(key=lambda r: (r['unit'], r['config'], r['query']))
        report['categories'][cat] = dict(base_population=bases[cat], eligible=len(rows),
            eligible_unique_configs=len({(r['unit'], r['config']) for r in rows}),
            selected=rows[0] if rows else None,
            reason='Lexicographically first eligible (unit,config,query); not largest effect',
            eligible_ids=[[r['unit'], r['config'], r['query']] for r in rows])
    save(out / 'selected_segments.json', report)
    return report


FONTS = {}
def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc', size)
    return FONTS[key]


def txt(draw, xy, s, size=24, color=TEXT, bold=False):
    draw.text(xy, str(s), fill=color, font=font(size, bold))


def render(segment, di, frame, thresholds, boxes, frozen=False):
    im = Image.new('RGB', (W, H), BG)
    dr = ImageDraw.Draw(im)
    rec, c, p = segment['rec'], segment['geometry'], segment['pred']
    q = rec['query']; group = rec['group']
    txt(dr, (42, 25), '仿真（Track A 程序生成）', 34, TEXT, True)
    dr.rounded_rectangle((1280, 28, 1878, 82), 18, fill='#263349')
    txt(dr, (1305, 38), 'v5 audit   ·   非真实传感器', 27, AMBER, True)
    txt(dr, (44, 90), f'{di+1:02d} / 04   {TITLES[di]}', 30, CYAN, True)
    txt(dr, (44, 132), f'单位 {rec["unit"]:02d}  /  配置 {rec["config"]:02d}  /  关注 {QN[q]}   ·   按预先规则取首个合格样本', 23, MUTED)
    for rect in ((40, 190, 754, 988), (774, 190, 1187, 988), (1207, 190, 1880, 988)):
        dr.rounded_rectangle(rect, 20, fill=PANEL)
    txt(dr, (60, 210), '场景真值 · 局部等距示意', 26, TEXT, True)
    txt(dr, (60, 250), '青框：关注查询盒  橙色：目标  灰线：背景', 20, MUTED)
    # Fixed camera for this segment; real mesh/world transforms drive all positions.
    base = np.linalg.inv(np.asarray(c['world_from_Q'][0]))
    def local(points):
        x = np.asarray(points).reshape(-1, 3)
        return (np.c_[x, np.ones(len(x))] @ base.T)[:, :3]
    poses = np.asarray(c['world_from_Q'])
    centers = local(poses[:, :3, 3])
    obj = [(o, local(np.asarray(o['triangles_world'])).reshape(-1, 3, 3)) for o in c['objects']]
    corners = np.asarray(list(itertools.product((0, 1), repeat=3)))
    qb = []
    for box in boxes:
        verts = np.where(corners, box[1], box[0])
        world = (np.c_[verts, np.ones(8)] @ poses[frame].T)[:, :3]
        qb.append(local(world))
    pts = np.concatenate([x.reshape(-1, 3) for _, x in obj] + [centers] + qb)
    # Fixed bounds from all path positions and all obstacle meshes (no animated zoom).
    static = np.concatenate([x.reshape(-1, 3) for o, x in obj if o['category'] != 'BACKGROUND'] + [centers, np.array([[-1, -.3, 0], [1, 1.8, 5.]])])
    px = static[:, 0] + .3 * static[:, 2]
    pz = static[:, 1] - .32 * static[:, 2]
    sx = min(140., 620 / max(np.ptp(px), 3.5), 560 / max(np.ptp(pz), 3.5))
    ox = 400 - sx * (px.min() + px.max()) / 2
    oy = 595 - sx * (pz.min() + pz.max()) / 2
    def project(x):
        x = np.asarray(x)
        return (float(ox + sx*(x[0]+.3*x[2])), float(oy + sx*(x[1]-.32*x[2])))
    scene = Image.new('RGBA', (W,H), (0,0,0,0))
    dr = ImageDraw.Draw(scene)
    ground = float(c['height'])
    for z in np.arange(0, 6.01, .5):
        dr.line([project([-1.5, ground, z]), project([1.5, ground, z])], fill='#263a50', width=1)
    for x in np.arange(-1.5, 1.51, .5):
        dr.line([project([x, ground, 0]), project([x, ground, 6])], fill='#263a50', width=1)
    tris = [(tr, o) for o, mesh in obj for tr in mesh]
    tris.sort(key=lambda z: -float(z[0][:, 2].mean()))
    for tr, o in tris:
        if o['category'] == 'BACKGROUND':
            dr.polygon([project(v) for v in tr], outline='#314359')
    for tr, o in tris:
        if o['category'] != 'BACKGROUND':
            dr.polygon([project(v) for v in tr], fill='#ba793b', outline=AMBER)
    for o, mesh in obj:
        if o['category'] == 'BACKGROUND': continue
        pos = project(mesh.reshape(-1, 3).mean(axis=0))
        txt(dr, (pos[0]+5, pos[1]-24), f'#{o["id"]} {o["category"]}', 18, AMBER)
    for k in range(6):
        color = CYAN if k == q else '#395468'
        for a, b in itertools.combinations(range(8), 2):
            if np.sum(corners[a] != corners[b]) == 1:
                dr.line([project(qb[k][a]), project(qb[k][b])], fill=color, width=3 if k == q else 1)
    cp = centers[frame]
    head = project(cp)
    foot = project([cp[0], ground, cp[2]])
    dr.line((head, foot), fill='#d3e4f4', width=7)
    dr.ellipse((head[0]-10, head[1]-10, head[0]+10, head[1]+10), fill='#d3e4f4')
    txt(dr, (foot[0]-34, foot[1]+10), '使用者', 20, MUTED)
    crop=scene.crop((60,290,735,865));im.paste(crop,(60,290),crop)
    dr=ImageDraw.Draw(im)
    txt(dr, (62, 883), '几何仅供解释；不作为网络输入', 22, MUTED)
    txt(dr, (62, 922), '固定局部视窗；背景裁剪，查询盒可重叠', 20, MUTED)
    txt(dr, (794, 210), '保存的 z4 特征', 26, TEXT, True)
    txt(dr, (794, 250), '8 × 8 分区 · 取 16 bin 最大值', 20, MUTED)
    heat = segment['z4'][frame].max(axis=-1)
    cmap = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 8) / 8 * 255), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    for y in range(8):
        for x in range(8):
            xx, yy = 802+x*45, 305+y*45
            dr.rounded_rectangle((xx, yy, xx+42, yy+42), 5, fill=tuple(cmap[y, x]))
    for i in range(352):
        cl = cv2.applyColorMap(np.array([[round(i/351*255)]], np.uint8), cv2.COLORMAP_INFERNO)[0, 0, ::-1]
        dr.line((804+i, 692, 804+i, 710), fill=tuple(cl))
    txt(dr, (800, 720), '0', 20, MUTED); txt(dr, (1137, 720), '8', 20, MUTED)
    txt(dr, (794, 767), '显示裁剪：0–8', 22, TEXT)
    txt(dr, (794, 805), f'当前最大值：{heat.max():.2f}', 23, CYAN)
    txt(dr, (794, 862), '来自已保存特征文件', 21, MUTED)
    txt(dr, (794, 900), '不重算模型、不插值造帧', 20, MUTED)
    txt(dr, (1227, 210), '六查询盒 · 分值 / 阈值 / 时间线', 26, TEXT, True)
    txt(dr, (1227, 252), 'S2 经典读出        A2 学习读出＋平滑', 22, MUTED)
    for k in range(6):
        yy = 298 + k*100
        g = 'HEAD' if k % 2 == 0 else 'BODY'
        selected = k == q
        dr.rounded_rectangle((1225, yy, 1861, yy+92), 9, fill='#193345' if selected else '#162437')
        truth = p['y'][frame, k] == 1
        txt(dr, (1238, yy+7), QN[k], 21, CYAN if selected else TEXT, selected)
        txt(dr, (1238, yy+35), '真值：障碍' if truth else '真值：空', 17, AMBER if truth else MUTED)
        for arm, xpos, color in [('S2', 1410, AMBER), ('A2', 1640, CYAN)]:
            thr = thresholds[g]['A0' if arm == 'S2' else 'A2']; val=float(p[arm][frame, k])
            alarm = val >= thr and frame >= 3
            txt(dr, (xpos, yy+6), f'{val:+.2f} / {thr:+.2f}', 20, TEXT)
            if alarm: dr.ellipse((xpos-15,yy+12,xpos-5,yy+22),fill=color)
        txt(dr,(1238,yy+62),'预热·不计入' if frame<3 else '白框：当前帧',16,MUTED)
        for j,(label,vals,color) in enumerate([
            ('真值',p['y'][:,k]==1,'#c096ff'),
            ('S2',p['S2'][:,k]>=thresholds[g]['A0'],AMBER),
            ('A2',p['A2'][:,k]>=thresholds[g]['A2'],CYAN)]):
            ly=yy+36+j*17;txt(dr,(1370,ly-3),label,13,MUTED)
            for t in range(12):
                xx=1413+t*36
                fill='#334055' if t<3 else (color if vals[t] else '#253549')
                dr.rounded_rectangle((xx,ly,xx+29,ly+11),2,fill=fill,outline=TEXT if t==frame else None,width=1)
    txt(dr,(1227,914),'紫：真值障碍  橙：S2报警  青：A2报警',20,MUTED)
    txt(dr, (1227, 946), '每行12帧；0–2预热，不计指标；3–11为评估帧', 19, MUTED)
    if frozen:
        state = '结尾定格 / 解释停留（无新增样本）'
    else:
        state = f'4× 慢放 · 原始 5 Hz · 第 {frame+1:02d}/12 帧 · 样本时刻 {frame/5:.1f}s'
    txt(dr, (42, 1010), state, 24, TEXT)
    txt(dr, (1130, 1012), '阈值来自 calib 10% 空对预算；非实机阈值', 23, AMBER)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ffmpeg', type=Path); ap.add_argument('--select-only', action='store_true'); a=ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.data/'source'))
    from cnh_learned_memory_fusion import causal_ewma
    from cnh_track_a_readout import BOXES
    results=json.loads((a.data/'analysis/v5_results.json').read_text())
    thresholds={g:{arm:results['working_points'][f'{g}|{arm}|0.10']['threshold'] for arm in ('A0','A2')} for g,_ in GROUPS}
    selection=select(a.data,a.out,causal_ewma,thresholds)
    if a.select_only: return
    intermediate=a.out/'simulation_v5_mp4v.mp4'
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
    assert writer.isOpened()
    hashes={str(p):digest(p) for p in (Path(__file__),a.out/'selection_rules.json',a.data/'analysis/v5_results.json',a.data/'source/cnh_learned_memory_fusion.py')}
    try:
        for di,cat in enumerate(IDS):
            rec=selection['categories'][cat]['selected']
            if rec is None: raise RuntimeError('No eligible example; build explicit fallback card')
            u,c=rec['unit'],rec['config']; d=load_unit(a.data,u,causal_ewma)
            idx=np.flatnonzero(d['config']==c);idx=idx[np.argsort(d['frame'][idx])]
            assert np.array_equal(d['frame'][idx],np.arange(12))
            gp=a.data/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json';fp=a.data/'features'/f'unit{u:03d}.npz';pp=a.data/'predictions'/f'unit{u:03d}.npz'
            geo=next(x for x in json.loads(gp.read_text())['configs'] if x['config']==c)
            with np.load(fp) as f:
                fi=np.flatnonzero(f['config']==c);fi=fi[np.argsort(f['frame'][fi])];z4=f['z4'][fi].astype(np.float32)
            for p in (gp,fp,pp): hashes[str(p)]=digest(p)
            segment=dict(rec=rec,geometry=geo,pred={k:d[k][idx] for k in ('S2','A2','y')},z4=z4)
            for frame in range(12):
                im=render(segment,di,frame,thresholds,BOXES)
                if frame in (0,5,8,11):im.save(a.out/f'segment{di+1}_frame{frame:02d}.png')
                bgr=np.asarray(im)[:,:,::-1]
                for _ in range(24):writer.write(bgr)
            im=render(segment,di,11,thresholds,BOXES,True)
            for _ in range(72):writer.write(np.asarray(im)[:,:,::-1])
    finally:writer.release()
    video=intermediate
    if a.ffmpeg:
        video=a.out/'simulation_v5.mp4'
        subprocess.run([str(a.ffmpeg),'-y','-i',str(intermediate),'-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    cap=cv2.VideoCapture(str(video));count=0; dims=set()
    while True:
        ok,frame=cap.read()
        if not ok:break
        count+=1;dims.add((frame.shape[1],frame.shape[0]))
    cap.release();assert count==1440 and dims=={(W,H)},(count,dims)
    receipt=dict(status='COMPLETE',video=str(video.resolve()),fps=FPS,frames=count,duration_s=count/FPS,resolution=[W,H],
        data_frames_per_segment=12,original_sample_rate_hz=5,slowmotion_factor=4,end_hold_seconds=2.4,full_decode_pass=True,thresholds=thresholds,source_sha256=hashes,video_sha256=digest(video))
    save(a.out/'video_receipt.json',receipt)
    lines=['# v5 仿真解释视频','',f'已生成 1920×1080、30 fps、48 s 视频。四段各12 s，实际各含12个5 Hz样本；4倍慢放9.6 s，结尾定格2.4 s。没有插值、模型重跑、重训或阈值修改。','', '|片段|基础分母|合格候选|唯一配置|选取 unit/config/query|','|---|---:|---:|---:|---|']
    for i,cat in enumerate(IDS):
        r=selection['categories'][cat];s=r['selected'];lines.append(f'|{TITLES[i]}|{r["base_population"]}|{r["eligible"]}|{r["eligible_unique_configs"]}|{s["unit"]}/{s["config"]}/{s["query"]}|')
    lines+=['','每类均取按 unit/config/query 排序的首个合格样本。前两类是筛选出的改善案例，空场景类条件为“至少一种方法发生误报”，第四类展示A2报晚或漏报；不能从这四例推断总体比例。分母分别是HEAD/BODY小目标近事件、全组空配置×组、全部近事件。','',f'数据范围64个audit单位、2048个配置，每配置评估9帧；UNKNOWN标签数={selection["unknown_labels"]}。原始帧0–2为预热，不计提醒指标。阈值逐组读取 v5_results 的10% calib工作点；不同方法分数不在同一标度。','', '场景几何、查询盒和真值仅为解释叠加，不输入网络；热图为保存z4在16个bin上的最大值，显示截断0–8。图为几何等距示意，不是相机画面或真实传感器。','', '完整视频1440帧解码通过；源码/输入/视频哈希见 video_receipt.json。代表帧PNG用于人工视觉检查。']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
