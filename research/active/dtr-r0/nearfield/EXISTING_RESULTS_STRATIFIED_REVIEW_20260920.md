# 已有结果分层审查：主体价值、挑战边界与证据强度

2026-09-20。用户授权的已有证据审查，依据[当前任务合同](CAMERA_FORWARD_CONTRACT_20260919.md#useful-core-capability-and-challenge-coverage)。
**结论：旧 A* 的主体价值应得到更明确的认可；简单表示和 public-positive 的局部收益值得保留；当前 Calibration 的主要问题仍是误报。新目标没有使全部负对照翻正。**

本次读取既存报告和保存的决策，重算分层账目；没有模型推理、训练、采集、阈值选择、重标注或新硬件。
它是事后、已消费数据的描述性审查，不是新确认实验。历史数值、冻结实验结论和结构化继承角色均不修改。
范围为与当前障碍目标直接相关的三条证据线：旧四传感器 corridor-fusion、当前 camera-forward、NFO 定位组件；不宣称审完所有历史导航/识别路线。

## 1. 分层规则与可比性

| 证据线 | 输入与原任务 | 本次分层 | 证据能支持什么 |
| --- | --- | --- | --- |
| 旧 singleconfirm288 | RGB/ToF/Radar/IMU；旧走廊与 5cm clear 口径 | BODY+HEAD clear 144；Rod clear 72；Boundary 72，完整覆盖原288帧 | 同一组保存决策下的主体/挑战取舍，不能直接替代两传感器验证 |
| 旧 public-positive v2 | 同四传感器链；old/new/MZ146/MZ158 各288帧 | 各 cohort 单独按上述三层分解 | 已消费 Development 的分组拟合/校准收益；不合并为一个泛化总分 |
| 当前 Core432 | RGB+8×8单回波代理；相机前向体积、2cm几何边界带 | 完整 INSIDE+OUTSIDE layouts 288；BOUNDARY layout 144 | 同模拟器新配置下的主体检测和负例负担；本次子集结论为事后解释 |
| 原 Thin96 | 原wide代理或后续nominal45代理，须分别标识 | 原96整体保留为挑战历史，内含大板、横杆、4cm杆/薄板、same-zone小前景等 | 不是96个细杆帧，更不是日常障碍的抽样分布 |
| NFO 500-image test | RGB+8×8模拟ToF，像素近场定位，66场景/6家族 | zone内near-area与near/far/missing return | 定位组件证据；没有可直接映射的物理BODY/HEAD/Core事件标签 |

子集来自既有 family、stratum、layout 元数据，不按模型成败挑帧。原全体、负例、UNKNOWN及挑战层保留。
旧 BODY/HEAD144 为72正/72负，来自12个配置、24段；Rod72和Boundary72各36正/36负。
当前 Core布局288 为72正/216负、24 clips；保留 INSIDE clips 的入界前负期，仍含4个原2cm带内的近深度负帧。
这是“完整主体布局”报告口径，不将它冒充严格排除全部几何boundary的分母。
原逐帧几何层另为 INSIDE72全正、BOUNDARY80含72正/8负、OUTSIDE280全负；两种分层不可混用。
仅看 INSIDE72 的“0 FP”没有低误报意义。重复帧不是独立样本，两个任务之间不比较排名或直接相减。

## 2. 旧四传感器：A* 的主体表现已很强，复杂度未显示额外主体收益

以下 TP/FP/FN 与时序从已有逐帧决策重算；主体分母固定144帧、72正/72负。

| 方法 | 主体 TP/FP/FN | Precision / Recall / FPR | 误报段 / 采样时长 | 主体事件 / 最大首次观察延迟 |
| --- | --- | --- | --- | --- |
| 旧 A | 71/14/1 | 83.53% / 98.61% / 19.44% | 8 / 3.50s | 12/12 / 0s |
| A + single public head | 71/14/1 | 83.53% / 98.61% / 19.44% | 8 / 3.50s | 12/12 / 0s |
| A* / Multi | 70/1/2 | 98.59% / 97.22% / 1.39% | 1 / 0.25s | 12/12 / 0s |
| Raw 表示 / Single 表示，各自 | 70/1/2 | 98.59% / 97.22% / 1.39% | 1 / 0.25s | 12/12 / 0s |
| A* + BCE residual | 69/1/3 | 98.57% / 95.83% / 1.39% | 1 / 0.25s | 12/12 / 0s |
| A* + ranking residual | 69/4/3 | 94.52% / 95.83% / 5.56% | 3 / 1.00s | 12/12 / 0s |

**主体全部12事件在片段首帧已经存在，均left-censored；0s不能证明提前预警。** 旧报告“Core18/18”包括6个杆类事件，本次没有把它写成BODY/HEAD18/18。
主体144中15帧没有ToF返回，仍计入；不报警不等于已确认安全/空闲。
A*相对A减少13个主体FP、丢1个HEAD后续TP，事件及首帧提醒相同；不是无损帧级支配。
Raw/Single与A*在主体上不只是总数相同：144帧报警决策逐项相同，因此这里没有复杂多关联带来的额外主体效果。
这也不证明更便宜或可删Radar/IMU；Raw仍有2217个传感器/几何特征及7个非关联RGB汇总。

挑战层继续保留，不能因移出主体而消失：

| 方法 | Rod72 TP/FP/FN；事件 | Boundary72 TP/FP/FN；事件 |
| --- | --- | --- |
| A | 26/11/10；5/6 | 14/12/22；8/12 |
| A* | 26/2/10；6/6 | 11/3/25；6/12 |
| Raw | 29/4/7；6/6 | 12/4/24；7/12 |
| Single | 29/4/7；6/6 | 11/4/25；8/12 |
| BCE residual | 32/9/4；6/6 | 19/7/17；9/12 |
| Ranking residual | 32/15/4；6/6 | 17/6/19；8/12 |

A*/Raw/Single有一个杆类事件延迟0.5s；两个residual把它恢复为0s，却损失同一个主体BODY帧。
因此BCE/ranking的召回收益主要服务挑战层，不是当前主体升级的依据；ranking还增加主体误报。
[原确认](corridor_fusion_v1/PUBLIC_SINGLE_RESULTS_20260917.md)、[表示消融](corridor_fusion_v1/REPRESENTATION_RESULTS_20260918.md)、[CCRL](corridor_fusion_v1/CCRL_RESULTS_20260918.md)的原结论保留。

## 3. 其他旧组件：有可继承收益，也有仍然明确的无增益

public-positive v2普通BCE的主体144分层如下，每行正72/负72，独立保留各cohort：

| cohort | A → calibrated OR 的 TP/FP/FN | 主体最大首次观察延迟 | 误报段 / 时长，两臂相同 |
| --- | --- | --- | --- |
| old | 71/11/1 → 71/11/1 | 0s → 0s | 6 / 2.75s |
| changed（保存键new） | 67/14/5 → 72/14/0 | 1.25s → 0s | 8 / 3.50s |
| MZ146 | 71/9/1 → 71/9/1 | 0.25s → 0.25s | 4 / 2.25s |
| MZ158 | 71/13/1 → 71/13/1 | 0s → 0s | 7 / 3.25s |

各臂主体事件均12/12，均left-censored。changed补回的5帧全部来自同一个HEAD配置，不能说成5个新检出事件。
**这是值得保留的主体响应改善证据，且未增加主体FP。** 原8帧总收益另外3帧在杆类。
但v2使用分组模型/工作点，不能等同后来单一模型；[single版本新场景](corridor_fusion_v1/PUBLIC_SINGLE_RESULTS_20260917.md)主体完全等于A，缺少其预期的returned-support补漏机会，属于迁移收益未被检验，不是已证伪或已确认。
来源：[v2](corridor_fusion_v1/PUBLIC_POSITIVE_V2_RESULTS_20260917.md)。

| 组件 | 重审发现 | 保留解释 |
| --- | --- | --- |
| [Pixel/query](corridor_fusion_v1/PIXEL_QUERY_RESULTS_20260918.md) | 固定工作点禁用，融合主体仍70/1/2；84.33% query排序属于表示诊断 | 保留表示能力，告警增益仍为零；新目标不自动挑新阈值 |
| [Tail rescue](corridor_fusion_v1/TAIL_RESCUE_RESULTS_20260918.md) | 保存summary主体70/3/2，较A*多2FP且未补主体TP；杆层也无补回 | 负结果不是仅由极限杆类拖累 |
| [S1 conditional C](corridor_fusion_v1/SPECIALIST_RESULTS_20260917.md) | old四个FP改善全在杆类；[完成的迁移](corridor_fusion_v1/CONFIRMATION_FINAL_20260917.md)主体67/14/5完全不变 | 保留旧局部正结果；具体配方迁移无增益仍成立，不能引用首次采集超时作最终结论 |
| [Surface oracle](corridor_fusion_v1/ORACLE_SURFACE_CEILING_RESULTS_20260918.md) | 旧两个BODY/HEAD漏帧分数仍约0.23/0.24；没有主体补回 | 特定模型/特征替换无收益，不扩成全部几何方法无效 |

Pixel/tail表采用保存summary核对与分层加总，未独立读取其NPZ预测重算；其证据强度与上节逐帧复核作区分。

## 4. 当前两传感器：检测能力成立于受控Core，但误报不能被分层掩盖

下表按完整INSIDE+OUTSIDE布局288帧重算，72正/216负、12事件；四臂prediction-UNKNOWN均247/288。
UNKNOWN可与保守报警同时出现，不能当成互斥第三类直接相减；原TN=0，本审查只称未报警负帧，不称确认空闲。

| 方法 | TP/FP/FN | Precision / Recall / FPR | 误报段 / 采样时长 | 事件 |
| --- | --- | --- | --- | --- |
| Raw nominal45 | 72/148/0 | 32.73% / 100% / 68.52% | 24 / 29.6s | 12/12 |
| Calibration | 72/126/0 | 36.36% / 100% / 58.33% | 24 / 25.2s | 12/12 |
| Calibration + noRGB | 72/120/0 | 37.50% / 100% / 55.56% | 24 / 24.0s | 12/12 |
| Calibration + RGB | 71/118/1 | 37.57% / 98.61% / 54.63% | 25 / 23.6s | 12/12 |

Calibration的受控主体召回值得保留，但相对于Raw只缩短误报时长，没有减少此子集的误报段数。
RGB比Calibration少8FP、多1FN、多1误报段；比noRGB仅少2FP，也多1FN和1段。
noRGB在这个主体子集没有TP/事件损失，比“全面失效”更准确；但不能据此抹掉它的原Boundary/native-retention失败或自动启用。
所有主体事件in-event首报均在entry1.2s，然而Raw12/12、其余10/12事件在入界时已持续报警；这不等于精准起报。
采样时长为报警帧数×0.2s，不是自然使用中的打扰时长，也不是设备延迟。

| 完整布局 | Calibration TP/FP/FN；段数 | RGB TP/FP/FN；段数 |
| --- | --- | --- |
| INSIDE144，72正/72负 | 72/39/0；11 | 71/39/1；11 |
| OUTSIDE144，全负 | 0/87/0；13 | 0/79/0；14 |
| BOUNDARY144，72正/72负 | 72/20/0；16 | 63/17/9；14 |
| 原完整432，144正/288负 | 144/146/0；40 | 134/135/10；39 |

Boundary事件Calibration12/12，RGB/noRGB均11/12，但漏的事件不同；RGB另有0.2s/0.4s两个接触事件延迟。
RGB/noRGB分别抑制30/35个含native corridor贡献的zone样本。原“无损迁移失败”依然正确；新主体目标只改变它的解释范围。
若另按逐帧几何严格排除boundary，352帧（72正/280负）的Calibration为72/140/0，FPR50%；同样不支持低误报。
来源：[协议](CORE_TRANSFER_PROTOCOL_20260920.md)、[结果](CORE_TRANSFER_RESULTS_20260920.md)、[基线](CALIBRATION_BASELINE_20260920.md)。

## 5. Thin96 与信息上限：细杆并非在所有观测配置下都失败

原96包含36正/60负、6严格事件，不能与Core432合并。以下顺序还包含观测生成方式变化：

| 旧96冻结阶段 | TP/FP/FN | 全事件 | 误报段 / 时长 |
| --- | --- | --- | --- |
| 原wide raw proxy | 12/5/24 | 3/6 | 2 / 1.0s |
| 原wide NFO | 7/5/29 | 2/6 | 3 / 1.0s |
| 原Depth Pro + global ToF | 0/0/36 | 0/6 | 0 / 0s |
| nominal45 raw | 30/19/6 | 5/6 | 6 / 3.8s |
| Calibration | 30/15/6 | 5/6 | 4 / 3.0s |
| 同样本RGB / noRGB | 30/10/6；30/13/6 | 各5/6 | 4 / 2.0s；6 / 2.6s |

nominal45改变8×8角布局并从原深度重新生成观测，**不是同输入上的算法提升**；覆盖图像采样31,416→10,384，不能只报告TP提升。
其4cm中心杆与薄板均已被检测；后四臂6FN全来自same-zone小前景。
旧同样本RGB优势是拟合机制证据（13个OUTSIDE监督样本均来自同一布局），不能比后续Core迁移失败更有说服力。
来源：[旧对照](BA_CAMERA_CORRIDOR_20260919.md)、[FOV](TOF_FOV45_20260920.md)、[校准](TOF_CORRIDOR_CALIBRATION_20260920.md)、[RGB拟合](TOF_LATERAL_ATTRIBUTION_20260920.md)。

当前[return-lineage](RETURN_LINEAGE_RESULTS_20260920.md)进一步澄清：公开单return没有选中目标，不等于私有模拟候选池没有目标。
Thin六个FN对应的8个目标corridor zone-records仍有7–14个目标samples，目标bin排第二，被背景选中；尚未证明改用多return会补回且无新增FP。
Core55/144正帧也出现这种竞争但全部仍报警；不得把zone损失率当作事件漏检率。
旧wide下的[cross-zone ceiling](CROSS_ZONE_ANCHOR_CEILING_20260920.md)只约束原公开输入，不是nominal45或所有传感器的信息上限。
[Zone-handoff](ZONE_HANDOFF_CEILING_RESULTS_20260920.md)与[score-factor](SCORE_FACTOR_COLLAPSE_RESULTS_20260920.md)诊断的gap是负帧；填平它们会增加误报，不能包装成主体障碍补漏。

## 6. NFO：保留定位收益，不能把像素面积改名为主体障碍

Matched同结构比较的2m mixed IoU47.11→51.04%、recall93.09→94.70%，FP536,667→470,344；它本来就是保留的组件正结果。
重新读取near-area五档，原500 test、同阈值、同分母；每档depth与NFO的near像素数一致：

| zone near-area | 已知像素 / near像素 | depth→NFO recall | depth→NFO IoU |
| --- | --- | --- | --- |
| (0,5%] | 164,248 / 3,509 | 76.00→66.32% | 3.15→4.01% |
| (5,10%] | 100,130 / 7,633 | 84.10→75.50% | 9.64→11.70% |
| (10,20%] | 173,116 / 26,168 | 90.20→87.44% | 18.53→20.34% |
| (20,50%] | 393,251 / 132,690 | 89.60→91.48% | 36.67→38.21% |
| (50,100%) | 504,517 / 379,891 | 94.85→96.97% | 74.48→76.30% |

中大面积两档recall、FP、IoU均改善，但其剩余背景pixel FPR仍为70.99%/82.57%。五档加总与原mixed的TP/FP/FN/TN完全一致。
>20%不是物理Core，≤20%不是独立细杆；zone面积随距离、投影、表面组成变化。此组缺少可用的物理主体事件/首报/误报段标签，这些指标NOT_EVALUABLE。
2m结论不自动覆盖全部距离：1m各档召回增加但FP率也增加；3m的20–50%档召回下降约0.99pp。
来源：[matched](BA_NFO_MATCHED_20260919.md)、[area](BA_NFO_AREA_DIAGNOSTIC_20260919.md)、[native audit](BA_NFO_NATIVE_SUPPORT_AUDIT_20260919.md)。

| 旧配方 | 新主体目标下仍须保留的限制 |
| --- | --- |
| [RGB-only](BA_NFO_RGB_CONTROL_20260919.md) | far-small recall97.89%但IoU6.35%、precision6.36%、FPR96.26%；full-image FPR94.27%，非仅薄物体困难 |
| [Hybrid](BA_NFO_HYBRID_20260919.md) | mixed IoU51.04→49.34%、recall94.70→93.28%；中大面积也退化，不构成主体升级 |
| [ZCR](BA_NFO_ZCR_20260919.md) | pure-far FPR约99.97%；原全局阈值联合约束无解结论仍在其范围内成立 |
| [Joint32](BA_NFO_JOINTFIT_20260919.md) / [Late32](BA_NFO_LATEFUSION_20260919.md) | 小集拟合可保留；[500帧迁移](BA_NFO_FROZEN_TRANSFER_20260919.md)mixed recall95.00→87.80/87.25%，不是仅极端杆类拖累 |
| [Fullsupport](BA_NFO_FULLSUPPORT_20260919.md) | mixed recall95.05→90.40%，full IoU61.60→59.78%；六家族召回均退，原负结论保留 |
| [Zone readout](BA_NFO_ZONE_READOUT_20260919.md) | 有精度取舍，但全图0 rescuedTP/1,878 lostTP；[局部分数排序](BA_NFO_CONDITIONAL_20260919.md)或oracle阈值不是已执行告警补回 |

## 7. 本次改变的判断与唯一优先问题

- **更应重视：** 旧A*主体性能、Raw/Single简单表示、public-positive v2的单配置HEAD响应收益、matched NFO中大near-area定位能力。它们各在原输入和证据范围内有效。
- **降低主体优先级：** 主要补杆类却损失主体帧的residual；零主体增益且增误报的tail；未启用的pixel分支。保留代码/证据不等于继续训练。
- **仍不升级：** 当前RGB/noRGB具体无损迁移角色、S1迁移、RGB-only/ZCR背景泛滥、NFO广泛迁移退化。不能通过新分层把它们追认成原实验成功。
- **当前唯一优先问题：** 保持Calibration为两传感器基线，减少完整主体布局中的outside及pre-entry误报段/时长，并在下一次比较前说明允许的事件/延迟代价。本审查没有足够证据指定一个新算法，也不自动启动后继实验。

当前两传感器Core的误报与旧四传感器A*的低FP并不矛盾：输入、几何、场景、观测代理和工作点不同。
下一决策应继承已有证据中有用的部分，不能把旧A*数值贴到新任务，也不能要求所有新方法先恢复每一帧挑战误差。

## 8. 核查、复现与交付

标准库脚本：[audit_existing_strata_20260920.py](audit_existing_strata_20260920.py)。只读取保存决策/标签/汇总；CPU标记为TASK_NOT_GPU_SUITABLE。
复核22份保存JSON：4份prediction seals，逐帧ID/标签身份，三层完整分母，原TP/FP/FN，原误报段/事件计数，Core逐事件首报/入界时已有报警/left-censored，以及NFO四距离五档计数。
所有输入在读取前后SHA256相同；脚本和输入hash写入receipt。此处不是重新验证原采集物理几何、所有源图像或模型权重。
独立只读检查发现并修正了“前一帧报警但entry当前未报也计为持续报警”的审查脚本错误，最终Core逐事件字段与原结果逐项相符。

最终本地结果：`artifacts.local/work/existing-strata-audit-20260920/recount-final.json`。
文件含每层完整计数、事件ID/时间、主体逐帧得失、UNKNOWN相关字段、全部输入hash；文档中的主表可由它复核。
重现时指定一个新输出文件，脚本拒绝覆盖旧receipt：

```powershell
python research/active/dtr-r0/nearfield/audit_existing_strata_20260920.py --output artifacts.local/work/existing-strata-audit-20260920/recount-check.json
```

本次没有新算法终点，不新增/重写实验登记与继承；已有ledger303问题也没有被绕过或修复。
仅任务拥有的过时临时汇总可清理，原模型、协议、预测和失败证据保留。没有后台服务、GPU训练进程或付费资源。
