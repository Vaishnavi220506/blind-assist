# 真实行人视角 RGB 数据适用性审查（2026-09-27）

**建议：SANPO-Real 优先进入小规模标注覆盖核查，尚不能直接验收候选质量；SideGuide 为需申请的备选，EgoWalk 不宜为本目标下载。** 本轮只查元数据和获批的 1 帧样本，未跑模型、筛全库或启动融合实验。

|候选|许可、获取与规模|深度、细障碍标注与内参|三指标可用性／1–2 m 通道内数量|
|---|---|---|---|
|**[SANPO-Real](https://github.com/google-research-datasets/sanpo_dataset)**|数据 CC BY 4.0，代码 Apache-2.0；公开 GCS，无需账号，已实取。701 session、617,408 立体帧对；237 session 有 112,768 分割帧，其中 18,787 人工帧；Real+Synthetic 总下载约 6 TB。|头/胸前步行视角；ZED SDK 稀疏深度＋CREStereo 估深，后者裁限 0–80 m，**不是准确性保证**。杆/标牌/树有实例；护栏/扶手为语义区域，树枝无独立类。内参、位姿和逐帧人工/传播标记可得。|召回、假候选/帧、区单位角误差均**条件可测**：需人工帧、有效深度及目标完整性复核。近距小物体数**未知**，公开距离统计不是该交集。|
|[SideGuide](https://github.com/ChelseaGH/sidewalk_prototype_AI_Hub)|作者声明允许商业/非商业研发并保留版权；国际用户填表等批准，AI Hub 限韩国国民申请。实际发放条款待核实，未申请。论文 35 万框、10 万实例多边形、18 万立体对；[平台](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=189)深度子集记 17 万。|ZED 立体部分是轮椅视角；GA-net 视差＋置信度，非独立深度真值；有效米制范围未知。有杆、路桩、标牌、树干、路障；按文件夹提供 calibration，需处理裁图。|三项**条件可测**，但掩膜/深度实际交集与对齐未核实；1–2 m 数量、最小包字节数未知。不能等同头戴步行域。|
|[EgoWalk](https://huggingface.co/datasets/EgoWalk/trajectories)|公开 trajectories 卡/API 标 MIT、无门槛，268 GB、1,168,593 行；原始记录 2.91 TB 需接受访问条件，未操作。|[论文 v2](https://arxiv.org/html/2505.21282v2)：50 小时胸前 ZED 2，SDK 深度、相机参数；3 万余自动可通行掩膜。有效距离范围未知，缺完整物体实例/全景标注。|现成标签**不能直接测三项**，需新增物体人工标注；1–2 m 细障碍数未知。|

**项目记录与最小样本：**历史 `SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md`（Git `f01a0072`）已有官方 train／CC BY 4.0 记录；旧 canonical 路径本轮未找到，当前 BA-NFO 本地集为 Synthetic，不冒充真实样本。获批下载官方 train 的 `d3CKbaNbOPz-ZjLPUd6qD6wSNC3OKQxL/camera_chest/left/000000` 及配套元数据，共 **7 文件、6,062,266 B**，逐文件 MD5 对云端一致，SHA-256 留档。人工标记 `HUMAN_ANNOTATED`；RGB/三通道掩膜 `(1242,2208,3)`，CREStereo `(720,1280)`，ZED `(1242,2208)`；深度实际为带高宽头的 gzip float16，不能按 README 的 NPZ 直接读。掩膜 R=类别、256G+B=实例；内参齐全，RGB/掩膜/深度预览无明显整体错位，未验证米制精度。样本、[来源/哈希](../../../../artifacts.local/datasets/sanpo-real-suitability-20260927/PROVENANCE.json)与[结构回执](../../../../artifacts.local/datasets/sanpo-real-suitability-20260927/STRUCTURE_RECEIPT.json)仅存忽略目录。

**下一步：**先定义“小物体尺寸、1–2 m 距离口径、重力/身体通道和实例匹配”，再另行授权少量官方 train 人工帧的覆盖计数，分别报告目标实例数、帧数及独立 session 数。优先核查细目标的双目深度和缺标；无效深度、未标树枝为 UNKNOWN，不能算无障碍或假候选。用内参转射线后按固定虚拟 ToF 的视场、坐标轴和区边界折算；仅评估共同视场，不能整图等分 8×8。

**边界：**[SANPO 论文](https://arxiv.org/html/2309.12172v2)指出传播标签有细杆/树失败；需使用修正的 `fixed_camera_poses.csv`，并校正深度分辨率对应内参。单帧不估计全库近距覆盖。数据相机 ≠ Atom、无配对 ToF，三指标仅描述已审核目标上的视觉候选；与[已消费 v2 工程目标](CNH_SOFT_PRIOR_DEV_20260927.md)比较须统一候选真伪口径并考虑联合误差和持续性，不能直接得到融合收益，也不能以单模型未达标否定相机引导路线。
