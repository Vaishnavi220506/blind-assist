"""Post-run paired descriptive analysis; never changes predictions or arms."""
import numpy as np
from cnh_rgb_distribution_alignment import OUT, ARMS, read, save


def main():
    ledger=read(OUT/'case-ledger.json')
    scenes=sorted({r['scene'] for r in ledger})
    rng=np.random.default_rng(2026100217)
    indices={s:[i for i,r in enumerate(ledger) if r['scene']==s] for s in scenes}
    samples=[np.concatenate([indices[s] for s in rng.choice(scenes,len(scenes),replace=True)]) for _ in range(2000)]
    pairs={}
    for arm in ARMS[3:]:
        pairs[arm]={}
        for baseline in ('depthpro','zone_q10'):
            ea=np.array([abs(r['errors_m'][arm]) for r in ledger])
            eb=np.array([abs(r['errors_m'][baseline]) for r in ledger])
            difference=(ea<=.02).astype(float)-(eb<=.02).astype(float)
            abs_difference=(ea-eb)*100
            pairs[arm][baseline]=dict(
                within2_delta_pp=float(100*difference.mean()),
                within2_delta_pp_scene_bootstrap_ci95=np.quantile([100*difference[ix].mean() for ix in samples],[.025,.975]).tolist(),
                mean_abs_error_delta_cm=float(abs_difference.mean()),
                mean_abs_error_delta_cm_scene_bootstrap_ci95=np.quantile([abs_difference[ix].mean() for ix in samples],[.025,.975]).tolist(),
                fixed2cm_rescues=int(((ea<=.02)&(eb>.02)).sum()),
                fixed2cm_breaks=int(((ea>.02)&(eb<=.02)).sum()),
                by_scene={s:dict(within2_delta_count=int(difference[ix].sum()),mean_abs_error_delta_cm=float(abs_difference[ix].mean())) for s,ix in indices.items()})
    save(OUT/'paired-analysis.json',dict(role='post-run descriptive fixed-arm paired comparison; no model changes',
        n=15,scene_clusters=8,resamples=2000,seed=2026100217,pairs=pairs))
    report=(OUT/'REPORT.md').read_text(encoding='utf8')
    for r in ledger:
        report=report.replace(r['id'],r['id'].replace('|','\\|'))
    lead=('结论：预先指定的跨格仿射主方法没有建立可靠增益。≤2cm从DepthPro的4/15升到7/15，但仍低于原q10的8/15，P95从10.70cm恶化到20.28cm。'
          '逆距离消融为9/15、P95=12.82cm，是可进一步固定复核的线索，不能结果后改称主方法成功。\n\n')
    report=report.replace('# 跨粗格距离分布校准RGB深度小试\n\n','# 跨粗格距离分布校准RGB深度小试\n\n'+lead)
    lines=['','## 固定臂的配对差（事后描述，不改变方法）','',
           '区间采用2000次8场景簇重采样，seed=2026100217；不是独立验证。百分点差为正表示≤2cm比例增加，MAE差为负表示误差减少。','',
           '|方法 vs 对照|≤2cm差 pp [95%场景区间]|MAE差 cm [95%场景区间]|修复/破坏例数|','|---|---|---|---|']
    for arm,comparisons in pairs.items():
        for baseline,p in comparisons.items():
            ci=p['within2_delta_pp_scene_bootstrap_ci95'];mi=p['mean_abs_error_delta_cm_scene_bootstrap_ci95']
            lines.append(f'|{arm} vs {baseline}|{p["within2_delta_pp"]:+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]|{p["mean_abs_error_delta_cm"]:+.2f} [{mi[0]:+.2f},{mi[1]:+.2f}]|{p["fixed2cm_rescues"]}/{p["fixed2cm_breaks"]}|')
    lines+=['','失败解释限定为当前证据：ai_051_005-cam_01-0083的全图仿射偏置+0.731m，把原本已偏大的局部距离进一步抬高，净距误差8.64→27.45cm；ai_047_009-cam_00-0061需要尺度2.811、偏置−1.094m且全格分位残差中位0.577m，显示该帧单个全图仿射关系并不紧。去掉目标格甚至邻域后，主法尾部仍差，不能只归因于拟合时使用了目标格。这里没有读取目标逐像素GT来拟合或选择系数。',
           '', '下一步建议：以固定逆距离臂和简单q10为对照，扩大同域可用原候选并检查场景集中性；只有稳定跨场景收益才值得继续做更接近硬件的传感器代理。所有消融、失败例和旧结果继续保留。']
    (OUT/'REPORT.md').write_text(report+'\n'.join(lines)+'\n',encoding='utf8')
    (OUT/'source'/'cnh_rgb_distribution_alignment_analysis.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())


if __name__=='__main__':main()
