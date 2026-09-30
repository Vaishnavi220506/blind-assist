# 覆盖、可分性与新方法比较图表（2026-10-01）

两张图已从现有结果生成，PNG 可插入文稿，SVG 保留矢量图形与可编辑文字。另备有新试点逐种子比较模板和接入程序。本轮无训练、推理、阈值调整或新实验，只整理已使用的仿真 Development；没有替换原报告或提升证据等级。

## 覆盖与退化

![覆盖与性能](../../../../artifacts.local/work/cnh-evidence-figures-20261001/coverage_performance.png)

左列为绝对 AUC，中列为冻结工作点 BER，右列为相对 FULL 的 AUC 损失与原扫描单位重采样 95% 区间。随机与同侧近距条件分开；实线在覆盖内，虚线为外推。同样的区间距离对应不同评估面积，不能解释全部退化。区间只含评估抽样，不含训练波动；曲线不能当作设备在线可读取的覆盖距离。

来源：[覆盖扫描](CNH_COVERAGE_SWEEP_20260930.md)。每格 144 个单位，每组 P=144、N=432；HEAD/BODY 指标平均，模型为三种子集成。

[矢量 SVG](../../../../artifacts.local/work/cnh-evidence-figures-20261001/coverage_performance.svg)

## 可分性诊断

![可分性分栏](../../../../artifacts.local/work/cnh-evidence-figures-20261001/separability_panels.png)

上排是表示探针，下排是模型输出。三个边距分别展示，误差线为 24 单位的 Q25–Q75 分布，不是置信区间。1.5 cm 来自严格跨墙配对，logit 固定较高表示占据；4.5/12 cm 的旧诊断跨墙重新抽样目标，logit 按每对参考差翻转，使用斜线柱标记。不同表示和实验不连成“信息损失链”，也不由中位数接近推出等效。

来源：[浅边界](CNH_SHALLOW_BOUNDARY_20261001.md) · [旧表示诊断](CNH_REPRESENTATION_SEPARABILITY_20260930.md)。每条件 24 单位、每场景 12 次噪声采样。特权方向知道候选位置，不是可达上限；浅边界大墙外侧目标有 11/24 与墙重叠，转弯历史标签会变化。

[矢量 SVG](../../../../artifacts.local/work/cnh-evidence-figures-20261001/separability_panels.svg) · [图中数值](../../../../artifacts.local/work/cnh-evidence-figures-20261001/separability_chart_data.json)

## 新方法收益与代价

[比较表](../../../../artifacts.local/work/cnh-evidence-figures-20261001/method_comparison.md) 已填入 FULL 历史扫描的十格 AUC、BER、FN、FP、FNR、FPR。历史参照不能代替同轮重训基线；新方法栏保留待填，没有编造试点结果。

[JSON 接入模板](../../../../artifacts.local/work/cnh-evidence-figures-20261001/seed_metrics.template.json) 每条记录对应模型、种子、区域、HEAD/BODY，包含实际单位列表、P/N/FN/FP/AUC、阈值来源及可选成本。试点方案自行确定区域和校准规则，本工具不追加通过门槛。运行时核对基线与方法的单位集合、计数和种子集合，输出逐种子差、均值与范围，并保留“少漏但多报”的代价。不同区域 AUC 不混成总体 AUC；描述性种子范围不作为训练置信区间。

复现图表：

```powershell
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' research/active/dtr-r0/nearfield/cnh_evidence_figures.py
```

接入真实新试点时，在模板中填入原始汇总，再加 `--trial-results <已填好的JSON绝对路径>`。程序额外生成 `trial_summary.json` 和 `trial_comparison.md`，不训练模型或访问新的评估数据。

核对：逐查询账本重新计算的 100 个模型／组／格指标与扫描结果一致；50 个表示统计量的中位数及四分位与逐对记录一致。已查看两张 PNG 的版面。接入功能用明确标注的合成测试记录检查了差值计算、误报代价保留，并拒绝同数量不同单位或不同种子集合；这些测试数字不是研究结果。

[数值检查](../../../../artifacts.local/work/cnh-evidence-figures-20261001/checks.json) · [接入检查](../../../../artifacts.local/work/cnh-evidence-figures-20261001/importer_acceptance.json) · [生成脚本](cnh_evidence_figures.py)
