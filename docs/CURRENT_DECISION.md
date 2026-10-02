# 当前研究决定

更新：2026-10-02。主线：盲杖互补的前视障碍感知。冻结v5与回放已完成；用户新授权持续推进ToF相关突破，开展Development机制探索。

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`（历史保留）；ToF 阶段：`V5_FROZEN_COMPLETE / HARDWARE_DEFERRED`。

## 当前决定

- **以三级真值评估序列报警。** 10月2日用户确定：伸入身体走廊必须报，身体外0–10cm擦身报警可接受、不计误报，更远或其他高度带才是清晰负例。停止在同批15/52条边缘上增加H3单帧测距归属臂；静态mask扩展复核额外37条0补回/8丢失（`f08029e6`，`NOT_SUPPORTED`），撤下其优先定位。方向误差扫描的整体冻结判读仍为 `NOT_PATH_LIMITED_AT_<=1DEG`。待Claude外扩标签结果推送，再用既有94000–94095序列评估固定误停预算下的及时停率；计划、分母和区间见[避障当前页](../research/active/dtr-r0/CURRENT.md)及[实验日志](../research/active/dtr-r0/RUNS.md)。已消费Development，不新渲染、不改写旧判据。
- **本轮用户授权的 v5 冻结复现通过，保留 NN 加固定五分数平滑 A2。** HEAD/BODY AP 0.8444/0.7914；相对 S2 配对95%区间均高于零，两组均64/64单位胜出。A3补回6个强信号漏检，但guard实际误报未通过预设要求，不采用。见[v5结果](../research/active/dtr-r0/nearfield/CNH_V5_RESULTS_20260928.md)。不扩大记忆网络、不追加调参；用户设备暂不方便，交付历史真实输入回放，现场演示延后。02段整段有物体、无负帧，不能判定误报或选择性；主失败对照改为04恢复背景中HEAD左/中均128/128触发。这里“空”仅指无新增物体，64cm柜面仍在名义查询范围内，不能当真实应用假警率；输入定义与查询语义尚未分离，迁移未成立。
- **接受 v4 执行偏差并披露。** 六个主检验及 A1/A2 保留为“正式结果，带已披露偏差”；不写成完全符合冻结流程，不事后修改协议。接受理由及恢复时点见[v4结果追加决定](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V4_RESULTS_20260926.md)。
- v1/v2 原失败保留；修复 v2、位置/质量/软先验诊断均为已消费 Development，不追认为正式证据。相机必要性、ToF物理上限与真实效果均未建立。
- 手机保留 A 基线及原首页→手动开始 A+LOCAL→结束返回首页；UNKNOWN 不等于无障碍。City、保护test、新UE采集和硬件第二阶段仍暂停；已有RGB/深度Development缓存可用于已授权的ToF归属探索。

## 保留证据

|证据|关键数字|边界|
|---|---|---|
|v3 / v4 主检验|63 / 64 audit单位，六项均成立；S2−B1-R HEAD/BODY：+0.099/+0.079、+0.092/+0.078|受控仿真相对增益|
|v4 A1/A2|BODY近事件1990；没报减少5.1–5.4pp。HEAD/BODY近事件1936/1990；单帧及时优势7.4–9.2 / 4.8–7.4pp|相同calib预算，audit实际假警不等|
|候选与软先验|修复v2：32calib/63audit；中等组合AUC .9保留HEAD/BODY增益37.1%/17.0%|消费Development；非通用候选规格|

完整主张、分母、来源提交与禁用措辞见[论文主张台账](../research/active/dtr-r0/THESIS_CLAIMS_20260927.md)。

## 未决与入口

RGB逼近选择性首轮失败保留；后续局部边缘小试发现理想正确距离可改善净距，但实际粗格对象/表面归属未解决。产品报警工作点仍未决定；短模拟序列及“序列×查询盒”假警不能换算真实提醒负担。真实计数/串扰/安装标定待硬件。确认畅通距离仅作≥10cm、ρ≥0.5、最坏摆放的附录辅助地图。

[公开数据检索](../artifacts.local/work/tof-real-histogram-search-20260927/REPORT.md)：已核实LCSPCData（TMF8820）真实直方图与部分真值的文件目录；THDR3K（L8CH）入口仍未核验。未下载数据；后续文件审计/验证另行决定，不直接迁移为L8CH标定。

[路线当前页](../research/active/dtr-r0/CURRENT.md) · [v3结果](../research/active/dtr-r0/nearfield/CNH_TRACK_A_SCALE_V3_RESULTS_20260926.md) · [软先验诊断](../research/active/dtr-r0/nearfield/CNH_SOFT_PRIOR_DEV_20260927.md) · [项目入口](PROJECT_STATE.md)

[封口前全文](operations/snapshots/CURRENT_DECISION_20260927_PRE_TOF_CLOSE.md)逐字节保存；旧快照中的待决状态不覆盖本页。[v5更新前全文](operations/snapshots/CURRENT_DECISION_20260928_PRE_V5.md)保留；后续当前页变更由Git历史保存。
