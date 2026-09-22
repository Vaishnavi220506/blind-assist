# LOCAL 冻结迁移前诊断：局部支持与外观的混合假设

2026-09-22，用户授权的只读诊断，已完成一次 governed run。
**建议检验一个机制：微小边界侵入与近边界错开在区域支持汇总后相似，
LOCAL 又依赖支持带中的外观，因而可能把背景颜色与近物回波组成错误局部证据。**
这是待检验假设，尚未定位模型的因果依赖，也没有形成过滤器或新候选。

保存的两个新增 FP 都由 **HEAD 中心 query 4** 触发，虽同属
`body_suspended_solid_g05_outside`；其分数不是紧贴旧阈值的数值误差。
相比之下，丢失的 RAW TP 只有很少的 possible 支持，没有 definite 支持。
两类现象值得在新场景中同时保留，不能只针对两个 FP 设计有利对照。

## 输入、完整分母与冻结条件

输入是 `ba-inherit-spatial-20260922-run` 的原封存特征、预测、选择、逐帧结果、
指标及哈希文件，以及原 prepared 的公开 RGB、ToF、identities 和 materialization。
没有载入模型 pickle，没有重新训练、推断、选择阈值或读取标签 NPZ。
原目录及文件在诊断前后逐个检查哈希，未修改。

先检查并保存字段/数组 schema：原 1,728 帧的 RAW/LOCAL 特征均为
`[1728, 6, 961]`；两臂前 910 列完全相同，RAW 后 51 列全零。
本报告只分析原 evaluation 的全部 576 帧、3,456 个保存 query 分数，
覆盖 16 个布局、48 个完整 clip。已有标签属于已消费的 controlled Development。
图示为 outcome 定义的诊断样本，不能获得新确认身份。

| 完整分母或结果组 | 帧数 |
| --- | ---: |
| 全部 evaluation | 576 |
| 正 / 负标签 | 256 / 320 |
| A 漏掉的正例 | 63 |
| LOCAL 相对 A 救回 TP / 仍漏 FN | 30 / 33 |
| A 沉默负例 | 314 |
| LOCAL 新增 FP / 未新增 FP 的 A 沉默负例 | 2 / 312 |
| LOCAL 相对 RAW 得到 / 丢掉 TP | 24 / 1 |

冻结 cutoff：LOCAL `0.7959699076137833`，RAW `0.776778260888226`。
原 A 分数、OR 决策和 UNKNOWN 全部保持。UNKNOWN 为 487 / 576；
两个新增 FP 均在原 UNKNOWN 内，沉默不是观察到空闲。
原 A / A OR RAW / A OR LOCAL 的 TP/FP/FN 分别为
`193/6/63`、`200/12/56`、`223/8/33`，本诊断没有创造新任务效果。

30 个救回 TP 分布在 14 / 16 个布局：29 BOUNDARY、1 INSIDE，
BODY/HEAD 各 15；四类分别为 body_protruding_plane 8、
body_suspended_solid 7、head_hanging_plane 10、head_horizontal 5。
最大分数来自 HEAD 中心的有 22 帧，BODY 中心有 8 帧；这是查询分数来源，
不等于 22 个正确 HEAD 定位。本诊断没有以帧级正标签证明层级归属。

## 三个关键帧与公开特征

下表中的末尾索引属于原完整身份；三个 winning query 都是 4。
颜色在 `[0,1]`，区间均值来自原区域回波，不能解释为该 RGB 像素的测得深度。
pixel fraction 的分母是全部区域 footprint 像素，既不是全图，也不是仅有效回波像素。

| 原帧身份 | LOCAL / RAW 分数 | A 分数 | definite / possible pixel fraction | possible 平均 range |
| --- | --- | ---: | --- | ---: |
| body_suspended_solid_g05_outside_03 | 0.923389 / 0.599212 | 0.295598 | 0.001736 / 0.009421 | 2.374379 m |
| body_suspended_solid_g05_outside_09 | 0.914901 / 0.645864 | 0.269180 | 0.003636 / 0.010248 | 2.321445 m |
| head_hanging_plane_g08_boundary_02 | 0.065136 / 0.826609 | 0.053993 | 0 / 0.003140 | 2.839655 m |

所有身份都带 `query_occupancy_` 前缀。两个 FP 的 LOCAL margin 分别为
`+0.127419`、`+0.118931`；lost TP 为 `-0.730834`。
两个 FP 的 BODY 中心分数反而只有 `0.445804`、`0.344826`。
因此不能把其来源笼统描述为 BODY 分支阈值附近抖动。

两 FP 的 possible-band RGB 均值近似 `[0.584, 0.467, 0.350]`，
相对 outside-band 的均值差约 `[0.285, 0.212, 0.140]`。
公开画面有暖色砖墙和深绿悬物，均值与暖色外观一致；没有像素归属标注，
**不能直接断言模型在使用砖墙，或某一回波实际上来自哪个表面**。
同样，这里的 `definite` 是整区间投影假设的几何分带名，不是物体所有权证明。

30 个 rescued TP 的 possible fraction 为 `0.002975–0.016033`，
中位数 `0.009174`；possible 平均 range 为 `2.162113–2.878772 m`，
中位数 `2.332957 m`。两 FP 在这两个范围内；其 RGB 三通道及相对 outside
的三通道差也均落在 rescued TP 的取值范围内。rescued 分数为
`0.911055–0.998738`，也与两 FP 重叠。
这不证明整个特征空间不可分，更不能说明用这两个负例设计的规则会泛化。
没有执行可分性训练、特征重要度归因、阈值扫描或新决策评价。

## 同布局、同采样序号的已有对照

| 组与帧 | INSIDE LOCAL / RAW | BOUNDARY LOCAL / RAW | OUTSIDE LOCAL / RAW |
| --- | --- | --- | --- |
| body_suspended_solid_g05, 03 | 0.999026 / 0.790835 | 0.980850 / 0.767340 | 0.923389 / 0.599212 |
| body_suspended_solid_g05, 09 | 0.999126 / 0.847389 | 0.960588 / 0.817913 | 0.914901 / 0.645864 |
| head_hanging_plane_g08, 02 | 0.833855 / 0.813924 | 0.065136 / 0.826609 | 0.050640 / 0.716111 |

前两组的 BOUNDARY 都是旧 LOCAL 救回的正例，而 OUTSIDE 是本次关注的 FP。
第 09 帧的 BOUNDARY/OUTSIDE 在 winning HEAD query 中有相同
definite fraction `0.00363636` 和 possible fraction `0.01024793`，
possible 平均 range 仅差约 `0.0000236 m`。该差值是汇总数的接近程度，
不是 ToF 的物理精度或两个完整传感器输入完全相同的声明。

lost TP 的 BOUNDARY/OUTSIDE 同样无 definite、possible fraction 同为
`0.00314050`，possible 平均 range 都约 `2.839655 m`；LOCAL 两者均低，
RAW 则在旧阈值下区别了这两个帧。INSIDE 的 possible fraction 为 `0.03090909`，
LOCAL 分数恢复到 `0.833855`。这支持把小侵入/小支持作为挑战，但还不能证明
面积是决定原因：几何、像素内容、所有区域特征和两模型拟合都可能共同起作用。

## 一个可证伪的迁移假设

假设 H：整区域回波区间被赋给区域内所有像素后，边界内外的小差异在局部
汇总特征中减弱；LOCAL 的部分收益和错误依赖于该支持带内的外观对比。
暖色背景与近处目标回波相容的组合可能保持 OUTSIDE 高分，而小面积的真侵入
可能只留下背景相近的汇总值而丢失原 RAW 的区分信号。
H 同时预言外观敏感性和小几何差异敏感性，尚非已经建立的解释。

最小有信息的冻结迁移对照是新基础布局的 **外观 × 几何成对组合**：

- 在物体几何、相机和时间序列相同的情况下，仅替换背景的材质/色彩；
  实际核验公开 ToF、查询几何与标签未变，再观察冻结 LOCAL/RAW 的分数及
  告警变化。材质名称本身不能替代这些不变性检查。
- 在同一外观下，用原尺寸物体做 BOUNDARY 与近边界 OUTSIDE 成对位置，
  保留少量侵入和完全错开的完整 clips，检验固定 LOCAL 是否稳定区分两者。
  同时包括悬物与薄片式 HEAD 边界例，避免只复刻两个 FP 的有利形态。

若几何不变的外观对足以改变 LOCAL 的救回/误报，且新 OUTSIDE 代价集中于
相同支持结构，H 获得受限支持；若外观对稳定而边界差异被可靠区分，H 受挫。
只有外观改动却同时改变 ToF/可见几何时，不能据该对照判定外观原因。
实际迁移预算、完整分母和保留条件由根代理的冻结协议确定；本诊断不添加
阈值、专家、过滤器、采集批次或后继训练。

## 执行与证据

```powershell
pwsh -NoProfile -File tools/ba.ps1 run research-ue -RunSpec artifacts.local/evidence/ba-local-transfer-diagnostic-20260922/run-spec.json
```

run ID 为 `local-transfer-diagnostic-20260922-v1`，route 为 `dtr-local-transfer`。
已有 derived-evaluator 准入及 prepared 精确子路径合同足够，未改政策。
UE preflight、master/fabric verification 与命令均 PASS，native input 为 0；
来源族仍为 `ue-query-occupancy-20260922`，没有恢复 fresh 身份。

输出根：`artifacts.local/evidence/ba-local-transfer-diagnostic-20260922/`。
`run/` 内保存 schema、输入/源码哈希、全部 576 个逐帧诊断记录、每个原始组
完整身份、全部 61 项命名特征的分布、两中心查询特征、完整六 query 分数、
三个关键帧的九个同组对照、RGB 预览、backend、summary 和输出封存。
`frame-diagnostic.json` 中的分组来自保存结果，仅供分析，不参与模型推断。
运行规格与 console 在输出根；journal 位于
`artifacts.local/evidence/resource-fabric/runs/dtr-local-transfer/local-transfer-diagnostic-20260922-v1.json`。

源码语法检查及运行中的 schema、特征共享、封存、索引、固定 cutoff、逐帧 OR、
UNKNOWN、完整分母、原比较身份和输入未变断言全部通过。
这是保存结果诊断核验，不是再次验证渲染器或 HGB 训练实现。
CPU / NumPy 2.4.4，backend 原因 `CPU_TASK_CLASS_SCALAR_SCORING`；
诊断科学命令计时约 1.24 s，不含 governed-entry 开销。
没有持续进程、worker 或付费资源，没有运行失败或机械重试。

结果仅支持决定下一次冻结成对检验的重点，不改变 LOCAL 的 COMPONENT、
RAW 的 NEGATIVE_CONTROL、A 基线或原误报段成本；没有新迁移性能、真实硬件、
自然分布、物理返回归属或安全结论。中央状态、登记和 Git 由根代理处理。
