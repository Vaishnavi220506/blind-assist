"""Publication figures from three saved JSON reports; no inference or recalibration."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
import numpy as np

COLORS={'A0':'#D8872D','A1':'#8A9098','A2':'#168F8A'}
NAMES={'A0':'S2 经典读出','A1':'NN 学习读出','A2':'A2 学习＋平滑'}
GROUPS=['HEAD','BODY']
GROUP_NAMES={'HEAD':'HEAD（头部）','BODY':'BODY（躯干）'}
CONDITIONS=['v5-stored','snr3','snr12','pitch-5','pitch+5']
CONDITION_NAMES=['标称 SNR6','SNR3','SNR12','以为 −15°','以为 −5°']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':11,'axes.labelsize':11,'axes.titlesize':13,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#84909A',
        'axes.labelcolor':'#243444','text.color':'#243444','xtick.color':'#526272','ytick.color':'#526272',
        'grid.color':'#DFE5E9','grid.linewidth':.7,'savefig.facecolor':'white',
        'svg.fonttype':'path','axes.unicode_minus':False,'figure.dpi':120})


def export(fig,out,stem):
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight',pad_inches=.15)
    fig.savefig(out/f'{stem}.svg',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def figure1(v,out):
    fig,ax=plt.subplots(1,2,figsize=(12.5,5.2),gridspec_kw={'width_ratios':[1.1,1]})
    xx=np.arange(2);w=.23
    for i,arm in enumerate(('A0','A1','A2')):
        vals=[v['AP'][g]['macro'][arm] for g in GROUPS]
        bars=ax[0].bar(xx+(i-1)*w,vals,w,color=COLORS[arm],label=NAMES[arm],zorder=3)
        ax[0].bar_label(bars,labels=[f'{x:.3f}' for x in vals],padding=4,fontsize=10)
    ax[0].set(xticks=xx,xticklabels=[GROUP_NAMES[g] for g in GROUPS],ylim=(0,1),ylabel='按单位宏平均 AP')
    ax[0].set_title('(a) 三种冻结读出',loc='left');ax[0].grid(axis='y',zorder=0);ax[0].legend(loc='upper center',bbox_to_anchor=(.5,1),frameon=False,fontsize=9,ncol=3)
    for i,g in enumerate(GROUPS):
        r=v['comparisons'][g]['A2-A0'];lo,hi=r['ci95'];d=r['delta']
        ax[1].errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',color=COLORS['A2'],capsize=6,ms=8,lw=2)
        ax[1].annotate(f'Δ={d:+.3f}   95% CI [{lo:.3f}, {hi:.3f}]\n胜出 {r["wins"]}/{r["units"]} 个单位',(d,i),xytext=(0,24),textcoords='offset points',ha='center',fontsize=10)
    ax[1].axvline(0,color='#72808C',lw=1,ls='--');ax[1].grid(axis='x')
    ax[1].set(yticks=[0,1],yticklabels=[GROUP_NAMES[g] for g in GROUPS],ylim=(1.6,-.65),xlim=(-.005,.16),xlabel='配对 AP 差：A2 − S2')
    ax[1].set_title('(b) 单位配对差与 95% CI',loc='left')
    fig.suptitle('图1｜v5 独立种子复现：AP 提升与配对证据',fontsize=16,y=1.02)
    fig.text(.5,-.015,'64 个 audit 单位；误差条属于配对差，不是单方法 AP 的置信区间。',ha='center',fontsize=10,color='#627181')
    fig.tight_layout();export(fig,out,'fig1_v5_ap')


def figure2(v,e,out):
    fig,axes=plt.subplots(1,2,figsize=(14,6.7))
    marks={1:'o',2:'^',5:'s',10:'D'}
    for ax,g in zip(axes,GROUPS):
        xs=[]
        for arm in ('A0','A2'):
            rows=v['curves'][f'{g}|{arm}'];x=[r['false_episodes_per_simulated_empty_minute'] for r in rows];y=[r['tiny']['timely']*100 for r in rows]
            xs+=x;ax.plot(x,y,color=COLORS[arm],lw=2,label=NAMES[arm],zorder=2)
        ax.set(xlabel='audit 实际误报段数 / 模拟空场景分钟\n（模拟分钟，非真实使用）',ylabel='小目标近事件及时率（%）',xlim=(0,max(xs)*1.025),ylim=(0,80 if g=='HEAD' else 65))
        n=v['curves'][f'{g}|A0'][0]['tiny']['near'];ax.set_title(f'{GROUP_NAMES[g]} · n={n} 个近事件',loc='left');ax.grid(alpha=.8);ax.legend(loc='upper left',frameon=False)
        for arm in ('A0','A2'):
            for j,r in enumerate(e['conditions']['snr6'][g][arm]):
                target=int(r['target']);x=r['audit_rate'];y=100*r['tiny_timely']/r['tiny_near']
                ax.scatter(x,y,s=54,marker=marks[target],color=COLORS[arm],edgecolor='white',lw=.8,zorder=5)
                if target in (5,10):
                    ax.annotate(f'目标{target}',(x,y),xytext=(7,12 if arm=='A2' else -20),textcoords='offset points',color=COLORS[arm],fontsize=9)
        # Low-range inset retains actual audit coordinates, not nominal targets.
        ins=ax.inset_axes([.49,.10,.47,.32])
        for arm in ('A0','A2'):
            rows=v['curves'][f'{g}|{arm}'];ins.plot([r['false_episodes_per_simulated_empty_minute'] for r in rows],[100*r['tiny']['timely'] for r in rows],color=COLORS[arm],lw=1.3)
            for r in e['conditions']['snr6'][g][arm][:2]:
                t=int(r['target']);x=r['audit_rate'];y=100*r['tiny_timely']/r['tiny_near']
                ins.scatter(x,y,s=33,marker=marks[t],color=COLORS[arm],edgecolor='white',lw=.5,zorder=5)
                ins.annotate(f'目标{t}',(x,y),xytext=(3,7 if arm=='A2' else -12),textcoords='offset points',fontsize=7,color=COLORS[arm])
        ins.set(xlim=(0,3),ylim=((43,60) if g=='HEAD' else (5,24)))
        ins.set_title('0–3 段/分钟局部',fontsize=9);ins.tick_params(labelsize=8);ins.grid(alpha=.6)
        ax.indicate_inset_zoom(ins,edgecolor='#86929A',alpha=.3)
    fig.suptitle('图2｜v5 小目标提醒权衡与事后 episode 校准工作点',fontsize=16,y=1.01)
    legend=[Line2D([0],[0],marker=m,color='none',markerfacecolor='#637380',markeredgecolor='#637380',label=f'calib 目标 {t} 段/分钟') for t,m in marks.items()]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,-.015),ncol=4,frameon=False,fontsize=10)
    fig.text(.5,-.058,'曲线来自冻结空对预算网格；标记为事后 episode 校准，坐标取 audit 实际值。1–2 段目标未经真实使用验证。',ha='center',fontsize=10,color='#627181')
    fig.tight_layout(rect=(0,.07,1,.96));export(fig,out,'fig2_timely_false_alarm')


def figure3(r,out):
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.2),gridspec_kw={'width_ratios':[1.05,1]})
    ys=np.arange(5)
    for row,g in enumerate(GROUPS):
        a,b=axes[row]
        for arm,offset in [('A0',-.11),('A2',.11)]:
            vals=[r['conditions'][c]['AP'][g][arm] for c in CONDITIONS]
            a.scatter(vals,ys+offset,color=COLORS[arm],s=56,label=NAMES[arm],zorder=4)
            for x,y in zip(vals,ys+offset):a.annotate(f'{x:.3f}',(x,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=9,color=COLORS[arm])
        a.set(yticks=ys,yticklabels=CONDITION_NAMES,xlim=(.4,1),xlabel='宏 AP（点估计；不画 CI）');a.invert_yaxis();a.grid(axis='x');a.set_title(f'{GROUP_NAMES[g]}：五条件 AP',loc='left');a.legend(loc='lower left',frameon=False,fontsize=9)
        for i,c in enumerate(CONDITIONS):
            p=r['conditions'][c]['paired'][g];d=p['delta'];lo,hi=p['ci95']
            b.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color=COLORS['A2'],ms=6)
            b.annotate(f'{d:+.3f} [{lo:.3f}, {hi:.3f}]',(hi,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=9)
        b.axvline(0,color='#758490',ls='--',lw=1);b.grid(axis='x');b.set(yticks=ys,yticklabels=CONDITION_NAMES,xlabel='配对 ΔAP = A2 − S2（95% CI）',xlim=(-.005,.21));b.invert_yaxis();b.set_title(f'{GROUP_NAMES[g]}：配对差的区间',loc='left')
    fig.suptitle('图3｜同布局敏感性检查：信号强度与安装角误认',fontsize=16,y=1.01)
    fig.text(.5,-.015,'每条件 64 个 audit 单位；真实安装角 −10°，误认条件为以为 −15° / −5°。右侧区间仅属于配对差。',ha='center',fontsize=10,color='#627181')
    fig.tight_layout();export(fig,out,'fig3_robustness')


def table1(e,out):
    rows=[]
    for g in GROUPS:
        for target in (1,2,5,10):
            for arm in ('A0','A2'):
                r=next(r for r in e['conditions']['snr6'][g][arm] if r['target']==target)
                rows.append([g,'S2' if arm=='A0' else 'A2',f'{target:g}',f'{r["calib_rate"]:.3f}',f'{r["audit_rate"]:.3f}',f'{r["tiny_timely"]}/{r["tiny_near"]}',f'{100*r["tiny_timely"]/r["tiny_near"]:.1f}%',f'{r["all_timely"]}/{r["all_near"]}'])
    fig,ax=plt.subplots(figsize=(14,8.7));ax.axis('off')
    labels=['组别','方法','calib 目标\n段/分钟','calib 实际\n段/分钟','audit 实际\n段/分钟','小目标及时\nn/N','小目标\n及时率','全部近事件及时\nn/N']
    table=ax.table(cellText=rows,colLabels=labels,loc='center',cellLoc='center',colWidths=[.085,.065,.115,.13,.13,.125,.105,.175],bbox=[0,.10,1,.85])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#D5DDE2');cell.set_linewidth(.6)
        if i==0:cell.set_facecolor('#EAF0F4');cell.get_text().set_weight('bold');cell.set_height(.052)
        else:
            cell.set_facecolor('#F0F8F7' if rows[i-1][1]=='A2' else 'white')
            if j==1:cell.get_text().set_color(COLORS['A2' if rows[i-1][1]=='A2' else 'A0']);cell.get_text().set_weight('bold')
    fig.suptitle('表1｜标称条件的 episode 校准工作点（事后描述性分析）',fontsize=16,y=.98)
    fig.text(.5,.07,'误报段分母：calib 9.6、audit 19.2 模拟空场景分钟；目标是校准约束，audit 实际负担可偏离。',ha='center',fontsize=11)
    fig.text(.5,.03,'复用已查看的 v5；1、2 段/分钟只是名义研究目标，尚未验证真实使用中的可接受范围。',ha='center',fontsize=10,color='#627181')
    export(fig,out,'table1_episode_working_points');return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    paths={'v5':a.work/'cnh-track-a-v5-20260928/data/analysis/v5_results.json','robustness':a.work/'cnh-v5-robustness-20260928/robustness_results.json','episode':a.work/'cnh-v5-robustness-20260928/episode-calibration/episode_calibration.json'}
    initial={k:sha(p) for k,p in paths.items()};d={k:json.loads(p.read_text()) for k,p in paths.items()}
    a.out.mkdir(parents=True,exist_ok=True);style();figure1(d['v5'],a.out);figure2(d['v5'],d['episode'],a.out);figure3(d['robustness'],a.out);rows=table1(d['episode'],a.out)
    extracted=dict(AP={g:d['v5']['AP'][g]['macro'] for g in GROUPS},paired={g:d['v5']['comparisons'][g]['A2-A0'] for g in GROUPS},
        curves={f'{g}|{arm}':d['v5']['curves'][f'{g}|{arm}'] for g in GROUPS for arm in ('A0','A2')},episode_nominal=d['episode']['conditions']['snr6'],robustness=d['robustness']['conditions'],table1_rows=rows)
    write(a.out/'extracted_plot_data.json',extracted)
    assert {k:sha(p) for k,p in paths.items()}==initial,'Source input changed during rendering'
    manifest=dict(inputs={k:dict(path=str(p.absolute()),sha256=initial[k]) for k,p in paths.items()},inputs_unchanged=True,builder_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in a.out.iterdir() if p.suffix in ('.png','.svg')},png_dpi=300,svg_font='glyph paths for portability',scope='Rendering/extraction only; no model, threshold or metric recomputation')
    write(a.out/'hash_manifest.json',manifest)
    text='''# CNH 论文图表

本目录只从三份既有 JSON 提取数值并绘图，没有重跑模型、重新选阈值或产生新实验。

**图1（fig1_v5_ap）**：v5 三种冻结读出在 HEAD/BODY 的按单位宏平均 AP。右侧单独给出 A2−S2 配对差及既有95% CI，二组均为64/64单位胜出；区间不属于单个方法AP。64个audit单位属于同一程序生成器的新种子复现。来源：v5_results.json。同一程序生成器内，不代表真实传感器或安全能力。

**图2（fig2_timely_false_alarm）**：S2/A2 的 v5 小目标近事件及时率—实际误报段曲线，HEAD分母424、BODY436。线来自冻结的空对预算网格；圆/三角/方/菱形分别标记事后episode校准目标1/2/5/10段/分钟，实际横纵坐标取audit结果，不把目标当横坐标。局部窗显示0–3段区间。episode校准复用已查看的v5，属于事后描述性分析；1–2段目标未经真实用户验证。不宣称A2在全部范围误报更低。来源：v5_results.json、episode_calibration.json。同一程序生成器内，不代表真实传感器或安全能力。

**图3（fig3_robustness）**：标称、SNR3、SNR12与安装角误认条件的AP点估计，以及单独面板的A2−S2配对差95% CI。真实安装角始终−10°；pitch-5 / pitch+5分别表示以为−15° / −5°。误差条仅用于配对差。每条件64个audit单位，复用v5布局；模型冻结，SNR档位按条件校准bias及阈值。绝对AP随条件变化，图不意味着排名或分数不变，不能写成外部分布或真实硬件鲁棒性。来源：robustness_results.json。同一程序生成器内，不代表真实传感器或安全能力。

**表1（table1_episode_working_points）**：标称SNR6条件，分别列HEAD/BODY、S2/A2在episode目标1/2/5/10的校准实际负担、audit实际负担、小目标及时n/N及全部近事件及时n/N。权威来源为episode_calibration.json；不能由robustness报告中的10%空对工作点替代。每组calib/audit空场景时长：320/640个全组空配置×9帧÷5Hz÷60=9.6/19.2分钟。短片段时间归一化负担不是连续真实使用误报率。同一程序生成器内，不代表真实传感器或安全能力。

所有图表提供300dpi PNG和SVG（文字转字形路径，避免跨机字体缺失）。机器可读数值为extracted_plot_data.json，输入/输出哈希为hash_manifest.json。

## 输入来源
'''
    text+='\n'.join(f'- {k}: `{p.absolute()}`' for k,p in paths.items())+'\n'
    (a.out/'FIGURES.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
