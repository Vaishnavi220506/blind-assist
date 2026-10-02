"""Decision-changing controls and angular quadrature audit for dither pilot.

Post-result diagnostic only: preserves initial and rank-repaired outputs.
No new target selection, tuning, or confirmatory claim.
"""
from pathlib import Path
import argparse
import shutil
import numpy as np

import cnh_tof_dither_pilot as p


def paired_error_audit(out):
    ledger=p.read(out/'case-ledger.json')
    extra=p.read(out/'audit/case-ledger.json')
    scenes=sorted({r['scene'] for r in ledger})
    rng=np.random.default_rng(2026100203)
    samples=rng.integers(0,len(scenes),size=(2000,len(scenes)))
    comparisons=[('static_inverse_square_peak',False),('inverse_square_shuffled',False),
                 ('static_maxmask_inverse_square_peak',True),('phase_mean_inverse_square',True)]
    result=dict(role='post-result fixed-policy paired absolute-error diagnostic; lower difference is better',
                n=15,scene_clusters=8,comparisons={},sign_errors={})
    signs={r['id']:r['gt_clearance_m']<=0 for r in ledger}
    for arm,other in [('inverse_square',False)]+comparisons:
        records=extra if other else ledger
        errs=[r['errors_m'][arm] for r in records]
        inside=[i for i,r in enumerate(ledger) if signs[r['id']]]
        outside=[i for i,r in enumerate(ledger) if not signs[r['id']]]
        result['sign_errors'][arm]=dict(inside_n=len(inside),outside_n=len(outside),
            inside_called_outside=sum(errs[i] is not None and ledger[i]['gt_clearance_m']+errs[i]>0 for i in inside),
            outside_called_inside=sum(errs[i] is not None and ledger[i]['gt_clearance_m']+errs[i]<=0 for i in outside),
            missing=sum(e is None for e in errs))
        if arm=='inverse_square':
            continue
        diff=np.array([(abs(a['errors_m']['inverse_square'])-abs(b['errors_m'][arm]))*100
                      for a,b in zip(ledger,records)])
        scene_values=[diff[[i for i,r in enumerate(ledger) if r['scene']==scene]] for scene in scenes]
        boot=[np.concatenate([scene_values[k] for k in selected]) for selected in samples]
        result['comparisons']['inverse_square minus '+arm]=dict(
            mean_paired_absolute_error_delta_cm=float(diff.mean()),
            median_paired_absolute_error_delta_cm=float(np.median(diff)),
            mean_scene_bootstrap95_cm=np.percentile([x.mean() for x in boot],[2.5,97.5]).tolist(),
            median_scene_bootstrap95_cm=np.percentile([np.median(x) for x in boot],[2.5,97.5]).tolist(),
            per_case_delta_cm=diff.tolist(),
            errors_improved=int((diff < -1e-10).sum()),errors_worsened=int((diff > 1e-10).sum()),
            errors_tied=int((abs(diff)<=1e-10).sum()),
            leave_one_scene_out_mean_cm={scene:float(np.concatenate([x for j,x in enumerate(scene_values)
                if scenes[j]!=scene]).mean()) for scene in scenes})
    result['source_sha256']=p.sha(__file__)
    p.save(out/'audit/paired-error-result.json',result)
    shutil.copyfile(__file__,out/'audit/paired-error-source.py')
    main=p.read(out/'result.json')
    supplements=p.read(out/'audit/result.json')
    table=[('整格静态inverse²最强峰',main['arms']['static_inverse_square_peak']),
           ('静态同邻区+mask最大覆盖格最强峰',supplements['arms']['static_maxmask_inverse_square_peak']),
           ('五相位inverse²解混',main['arms']['inverse_square']),
           ('同观测打乱相位',main['arms']['inverse_square_shuffled']),
           ('同观测抹去相位差异后解混',supplements['arms']['phase_mean_inverse_square'])]
    lines=['# 小角度旋转与粗格距离归属：条件诊断','',
        '**保留长尾改善信号，但尚未建立旋转本身的稳定收益。** 同15例/8场景中，五相位inverse²解混的净距误差≤2cm为10/15，原格静态峰为9/15；加入相同邻区与预测mask的简单静态对照则为11/15。解混P95绝对误差4.93cm优于这两个静态对照的12.83/6.94cm，但相对后者的配对平均绝对误差改善仅0.39cm，场景区间跨0。','',
        '|条件|≤2cm / 15|绝对误差P50 / P95(cm)|','|---|---:|---:|']
    for title,metric in table:
        lines.append(f'|{title}|{metric["within2cm"]}|{metric["median_abs_cm"]:.2f} / {metric["p95_abs_cm"]:.2f}|')
    lines+=['','主实验采用固定5个yaw（−2.8125/−1.40625/0/+1.40625/+2.8125°），精确同光心旋转；32×32角采样/格、5cm几何距离箱。预测mask由已有Depth Pro边缘内侧seed和±25%相对深度连通域产生；对象真值只用于模拟及最后评价。每个完整粗格都产生观测，mask不筛除光子。','',
        '条件解码为 H(t,z,b)=w(t,z)F(b)+(1−w(t,z))B(z,b)，非负拟合共同前景谱F与各格独立、跨相位稳定的背景谱。该模型假设不能覆盖任意倾斜表面或变化背景。逆平方臂保留预测几何占比w，未用真值距离校正混合权重。','',
        '|五相位inverse²相对对照|平均配对绝对误差变化(cm，负数更好)|场景bootstrap95%|改善/退步/不变|',
        '|---|---:|---:|---:|']
    for name,metric in result['comparisons'].items():
        ci=metric['mean_scene_bootstrap95_cm']
        lines.append(f'|{name}|{metric["mean_paired_absolute_error_delta_cm"]:.2f}|[{ci[0]:.2f}, {ci[1]:.2f}]|{metric["errors_improved"]}/{metric["errors_worsened"]}/{metric["errors_tied"]}|')
    lines+=['','以上配对误差变化的中位数均为0；大多数例读数不变，长尾收益集中在少数例。相位均值对照与动态臂14/15例读数一致；动态只在一个墙面案例进一步降低误差7.82cm，去掉其所属场景后这一差异为0。均匀几何臂与相位打乱均为10/15；不能把成功数差归因于运动。','',
        '符号诊断：7个走廊内边缘中，原格静态峰误判为外3/7；动态、同邻区mask静态、打乱相位与相位均值均0/7。8个走廊外边缘五者均误判为内1/8。因只有2/15个GT边缘距身体线≤2cm，这不是浅擦碰召回或误停率。','',
        '20个固定Poisson抽样的inverse²敏感性为201/300 case-draw满足2cm，仍只有15个独立案例，未做配对静态噪声比较。光子尺度4000、每bin环境均值0.1均为假设；无脉冲卷积/串扰/反射率变化，不能当硬件性能。','',
        '数值与实现验证：纯前景格的零背景列曾触发错误拒答，首轮目录完整保留；移除无关零列后同配置重跑，两例恢复可输出，最终15/15可比。纯前景/混合恢复、静态可辨识性、相位打乱与投影夹具通过。32→64采样时inverse²全部15读数不变；uniform一例变化，成功数10→9。留相位预测误差已存账本，说明低维模型存在明显失配。','',
        '下一决定：不把本轮称为突破或进入报警否决。保留“正确相位可能压低少数长尾”这一可证伪方向；先改进局部表面模型/观测匹配，避免将mask和邻区信息的静态收益错记为运动增益。','',
        '边界统一：已消费且预选干净边缘的Development；已知粗格/地面先验；只有纯旋转同光心，不含真实头动平移、遮挡变化、运动模糊、顺序扫描、姿态误差和实测NIR响应；5cm几何箱不是L8CH/CNH硬件分辨率。','',
        '证据：[主结果](result.json)、[逐例账本](case-ledger.json)、[补充控制与数值积分](audit/result.json)、[配对误差与符号](audit/paired-error-result.json)、[检查](verification.json)。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(result)


def run(out):
    audit=out/'audit'
    if audit.exists():
        raise RuntimeError('Preserve audit outputs')
    audit.mkdir()
    shutil.copyfile(__file__,audit/Path(__file__).name)
    ledger=p.read(out/'case-ledger.json')
    old={r['id']:r for r in p.read(p.PRIOR/'case-ledger.json')}
    manifest={r['id']:r for r in p.read(p.CACHE/'manifest.json')}
    cameras={r['id']:r['camera_matrix'] for r in p.read(p.CACHE/'observations.json')}
    p.save(audit/'PLAN.json',dict(role='post-result diagnostic, no threshold tuning',
        questions=['Is apparent dither gain explained by RGB mask and extra neighboring cells?',
            'Do32vs64midpoint samples per zone axis change selected ranges?'],
        control='static same-zone-set max predicted-mask-overlap zone, then peak; zoneID tie break',
        flatten='phase-mean each zone histogram repeated5times, retain actual weights; removes phase association',
        quadrature='same5yaw/every rule, only32to64numerical integration convergence',
        source_sha256=p.sha(__file__),pilot_source_sha256=p.sha(p.__file__)))
    records=[]
    for i,row in enumerate(ledger):
        with np.load(out/'observations'/f'case-{i:02d}.npz') as blob:
            observations={key:blob[key] for key in blob.files}
        zones=row['retained_zones']
        ranges={}
        chosen=min(zones,key=lambda z:(-observations['overlap'][2,z],z)) if zones else None
        for kind in ('uniform','inverse_square'):
            ranges['static_maxmask_'+kind+'_peak']=p.static_readouts(
                observations['static_'+kind],chosen)['peak'] if chosen is not None else None
            flat=np.repeat(observations[kind].mean(axis=0,keepdims=True),5,axis=0)
            ranges['phase_mean_'+kind]=p.decode(flat,observations['overlap'],zones)['radial_m']
        item=old[row['id']]
        source=manifest[row['frame_id']]
        camera=cameras[row['frame_id']]
        with np.load(p.CACHE/'predictions/native'/f'{row["frame_id"]}.npz') as blob:
            mask,_=p.prediction_mask(blob['native_depth'],item['prediction'])
        ref=p.reference_frame(p.ROOT,source,camera)
        high=p.simulate(ref['radial'],mask,camera,n=64)
        for kind in ('uniform','inverse_square'):
            ranges['quadrature64_'+kind]=p.decode(high[kind],high['overlap'],zones)['radial_m']
        errors={arm:row['radial_edge_factor']*r-.30-row['gt_clearance_m']
                if r is not None else None for arm,r in ranges.items()}
        records.append(dict(id=row['id'],scene=row['scene'],chosen_static_zone=chosen,
            radial_ranges_m=ranges,errors_m=errors,reference_errors_m=row['errors_m']))
        print('audited',i+1,'/15',flush=True)
    arms=list(records[0]['errors_m'])
    result=dict(n=15,scenes=8,arms={arm:p.summarize(records,arm) for arm in arms})
    result['quadrature_selected_range_changes']={kind:sum(
        r['radial_ranges_m']['quadrature64_'+kind]!=base['radial_ranges_m'][kind]
        for r,base in zip(records,ledger)) for kind in ('uniform','inverse_square')}
    result['paired_success_differences']={}
    scenes=sorted({r['scene'] for r in records})
    rng=np.random.default_rng(2026100202)
    samples=rng.integers(0,len(scenes),size=(2000,len(scenes)))
    comparisons=[('inverse_square','static_inverse_square_peak',False),
        ('inverse_square','inverse_square_shuffled',False),
        ('uniform','uniform_shuffled',False),
        ('inverse_square','static_maxmask_inverse_square_peak',True),
        ('inverse_square','phase_mean_inverse_square',True)]
    for candidate,baseline,supplement in comparisons:
        differences=[]
        for base,extra in zip(ledger,records):
            a=base['errors_m'][candidate]
            b=(extra if supplement else base)['errors_m'][baseline]
            differences.append(int(a is not None and abs(a)<=.02)-int(b is not None and abs(b)<=.02))
        scene_values=[np.array([d for r,d in zip(records,differences) if r['scene']==scene]) for scene in scenes]
        draws=[np.concatenate([scene_values[k] for k in selected]).mean()*100 for selected in samples]
        result['paired_success_differences'][candidate+' minus '+baseline]=dict(
            percentage_points=float(np.mean(differences)*100),scene_bootstrap95=np.percentile(draws,[2.5,97.5]).tolist(),
            improves=sum(d==1 for d in differences),worsens=sum(d==-1 for d in differences))
    p.save(audit/'case-ledger.json',records)
    p.save(audit/'result.json',result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=p.ROOT/'artifacts.local/work/cnh-tof-dither-20261002-rankfix')
    parser.add_argument('--paired-errors',action='store_true')
    args=parser.parse_args()
    if args.paired_errors:
        paired_error_audit(args.out)
    else:
        run(args.out)
