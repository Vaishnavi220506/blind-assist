# UE 数据复用入口

打开本地 [可搜索分类目录](../../artifacts.local/evidence/ue-reuse/current/index.html)。
另有 [CSV](../../artifacts.local/evidence/ue-reuse/current/inventory.csv) 和
[完整 JSON](../../artifacts.local/evidence/ue-reuse/current/inventory.json)。
原文件保持原位，索引通过现有 `asset_catalog` 登记稳定 locator、元数据身份和文件清单。
这是一份用途导航，不是新的实验结论或数据权限总账；证据状态沿用总账，缺少依据保留 unknown。

## 实验入口已经接入

UE 实验通过 `tools/ba.ps1 run research-ue` 执行。默认是已消费 Core 432 帧的
ToF 回归；提供 `-RunSpec` 则执行声明的研究命令。两种方式均强制走
`asset_runtime` 的 UE 准入检查。已有 DTR/L10 governed run 若声明 UE 索引中的输入，
或命令指向 UE/nearfield 研究脚本，也会自动触发检查。

```powershell
pwsh -NoProfile -File tools/ba.ps1 run research-ue -RunId ue-core-regression-NEW_ID
pwsh -NoProfile -File tools/ba.ps1 run research-ue -RunSpec artifacts.local/evidence/MY_RUN/run-spec.json
```

使用配置好的 research Python（需有 NumPy/OpenCV）；可用 `-Python` 指定环境。
默认回归不启动 UE、不训练模型、不改阈值；调用现有 `score_frame / decide`，
逐帧核验观测哈希，封存新预测后才打开旧预测作一致性比较。

运行规格在原 `blindassist-asset-run-v1` 中增加：

```json
{
  "reuse": {"mode": "regression", "query": "core transfer observations tof calibration"},
  "inputs": [{
    "alias": "observations",
    "asset": "work/ba-core-transfer-20260920",
    "relative_path": "observations",
    "role": "observation",
    "purpose": "consumed-development-regression"
  }]
}
```

其余 `id / route / question / evaluator / command / outputs` 字段沿用
[资产运行器](README.md)。命令使用 `{{input:observations}}` 等绑定路径。
mode 支持 regression、diagnostic、development、training；本入口不授予 fresh/final 权限。

实际执行顺序：

1. 按 query 查询本地分类索引和总账中新产生的派生结果，将候选、选择、来源族与政策哈希写入 journal。
2. 按 [准入策略](../../data/ue-reuse-policy.json) 检查路径范围、观测/配置/evaluator 角色、
   已消费或保留状态；检查失败时，不启动子进程、不新增消费记录。
3. 解析实际子路径、记录每个输入别名及身份，执行原研究命令。
4. 登记结果、输出哈希和输入→输出派生关系，并将来源族、evaluator 角色限制传给后续复用。

目录级 unknown 可以由有来源文档的精确准入条目获得本次 Development 使用许可，
原总账状态不会因此被升级。来源族数也不等于已证明相互独立的数据集数。
Spatial BCE 整目录含保留 test，不能靠填写 `split=train` 解锁；须先建立独立 train/dev
子集及可审计契约。其他未准入目录、尚未声明角色的缓存，保留候选可见但阻止直接执行，
按原来源文档补齐具体契约，不因目录名称猜权限。

这不是操作系统访问沙箱；它校验 governed run 的声明。历史上绕过 `ba.ps1` / runtime
直接调用的研究脚本尚未全部迁移，不能声称它们自动受控。新 UE 研究使用上述入口；
新数据或适配器在同一改动中补齐精确输入契约，避免整理完成后再次失去来源记录。

2026-09-22 集成验收：Core 432/432 帧与历史 calibrated 决策完全一致，原阈值保持不变；
输入消费、派生边和 master/fabric 校验均通过。这只证明流程及回归一致性，不是算法提升。
回执位于 `artifacts.local/evidence/resource-fabric/runs/ue-reuse/`，实际结果位于
`artifacts.local/evidence/ue-reuse-ue-reuse-integration-20260922-v2/result.json`。

## 先按任务选数据

以下规模来自所链接的已有研究文档，不是此次重新解码、验收或去重的结果。
路径相对 `artifacts.local/`，本机是否存在以生成清单为准。

| 用途 | 数据入口 | 复用说明 |
| --- | --- | --- |
| 修改场景、重新采集 | `unreal/BlindAssistStreetLab`、`unreal/BlindAssistObstacleLab`、`unreal/CitySample` | 工程与素材，不能作为已采集帧数；许可按原始来源 |
| 城市静态诊断 | `nearfield/city-pcg-20260908/worker-scale500-v1` | 历史 1500 帧；相机位姿有限，HEAD 覆盖不完整 |
| BODY/HEAD 表征和几何适配 | `work/body-query-5000-20260909/dataset-v1`、`work/body-query-10000-20260909/final-dataset-v2` | 历史 5000/10000 帧；静态查询，不是完整事件序列；旧标签不能直接改名为当前走廊标签 |
| 接近过程回归 | `work/ba-core-transfer-20260920`、`work/ba-core-workpoint-transfer-20260920`、`work/ba-core-hold-validation-20260920` | 各 432 帧/36 clips；已消费 Development |
| 完整事件及时序 | `work/ba-full-event-transfer-20260920` | 576 帧/24 clips，接近、停留、撤退 |
| 空间训练和开发集诊断 | `work/ba-spatial-bce-20260920` | 2880 帧中 1728 train、576 dev 已使用；另 576 test/8 groups 仍保留，不能自动加入训练 |
| 近边界迁移回归 | `work/ba-spatial-complement-transfer-20260921` | 1152 帧/48 clips/16 layouts，已消费 |
| 几何覆盖与失败分析 | `work/ba-data-coverage-20260921` | 历史 3456 帧；保留原实验失败结论与适用范围 |
| 固定输入、闭环与传感器排错 | 分类目录中的“闭环回放与传感器诊断” | 先查相应 protocol/manifest；固定回放不能代表改变运动后的反事实性能 |

## 防止错误复用

- NFO 的 500 UE 帧来自 BODY-query 5k，不是新增独立数据。projection-stress 也复用既有 1152 帧。
- 相同源数据的特征、预测、模型、预览、审核结果不增加独立样本数。此次只建立元数据身份，未做内容去重，不删除任何重复候选。
- BODY-query 的原几何不能直接证明完整场景阴性；旧审计中的 lateral 候选不能代替厘米级近边界阴性，保留 UNKNOWN。
- RGB/传感器观测与 native depth、world pose、对象身份、标签等 evaluator-only 信息分别使用；目录中同时存在不代表都可输入模型。
- 已消费数据可做明确披露的 Development、诊断及回归，不能恢复 fresh 身份。失败、重试和 partial 目录须查原始完成回执后选取。
- 分类依据目录和文档引用，是导航标签；某些工作目录含真实/公共/UE 混合来源，列在“混合来源待核验”，不自动宣称纯 UE。

## 刷新与程序复用

在仓库根执行 `python tools/data/ue_reuse_inventory.py`，更新 HTML/CSV/JSON 与现有资产总账。
扫描本机 `unreal/`、`nearfield/` 的直接子目录，以及 Git 跟踪 UE/nearfield 文档引用的
`work/` 目录；不连接副机、不镜像远程数据、不读取图片或标签内容、不改写 payload。
缺失引用列入 JSON 的 `missing_references`，并非已证实数据丢失，可能是历史或计划路径。
文件数是磁盘文件数而非帧数；大小是逻辑字节，不能作为独立数据量或实际占用。
扫描为时间点快照，活跃目录可能变化；`vanished` 和 `skipped_reparse` 明示扫描限制。

选定资产后，通过已有接口留下消费记录：

```powershell
python tools/data/asset_catalog.py resolve work/ba-core-transfer-20260920 --consumer YOUR_RUN_ID --purpose development-regression
```

## 依据

- [已有数据规模与保留集](../../research/active/dtr-r0/nearfield/SPATIAL_DATA_REUSE_INVENTORY_20260921.md)
- [几何与 payload 适配边界](../../research/active/dtr-r0/nearfield/EXISTING_CORRIDOR_DATA_RESULTS_20260921.md)
- [City 数据审计](../../research/active/dtr-r0/nearfield/CITY_DATA_AUDIT_20260908.md)
- [空间迁移](../../research/active/dtr-r0/nearfield/SPATIAL_COMPLEMENT_TRANSFER_RESULTS_20260921.md)
- [扩展几何覆盖](../../research/active/dtr-r0/nearfield/DATA_COVERAGE_RESULTS_20260921.md)
