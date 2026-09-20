# City Sample 高清机制演示：范围与交付核查

2026-09-20。工程演示扩展，保留原算法、阈值、1296帧评估证据与已有结论。新增场景只有观测与冻结策略响应，没有评估真值，不构成新效果确认。

## 完成状态

完成六区域、八段、192帧原生高清采集及页面交付。单帧HD smoke另存，不混入192帧。六份HD收据全部PASS；原生配对、共姿态/FOV、规格与源地图完整性、资源健康和任务进程释放均通过。逐段查看八张预定中间帧，材质、加载和构图正常。来源总审计为 `source-completion-summary.json`，导出哈希为 `export-receipt.json`。

页面验收通过：默认八卡画廊、1920×1080图片、原图入口、逐帧与手动选区联动、24采样时间线、2倍速连续八段到末帧自动停止、原保持补帧导览以及仅三个评估集合进入结果页。1440像素桌面与568像素窄窗已实际查看，无横向页面溢出；浏览器无脚本警告或错误。修复了从靠下的场景卡片进入回放时沿用旧滚动位置的问题。数据检查覆盖原1296帧/27组指标及新增192帧/8段的尺寸、未标注隔离、独立保持历史，全部PASS。

直接file入口仍只有静态检查；浏览器功能验收使用本地HTTP，未绕过已有file导航限制。Big City `big01`有一次引擎启动断言，0帧退出，完整保留在 `attempts/big01-startup-01`；同设置一次重试通过，没有覆盖已产帧或按结果换场景。

## 场景与采样

预先保存的 `plan.json` 包含Big City五处位置与Small City一处位置，8段×24采样，共192帧；单帧smoke另计。位置是两张原生City Sample地图中的不同区域，不是六个独立城市。

| 区域 | 片段ID | 预定动作 |
| --- | --- | --- |
| Big City / plaza | plaza_02、plaza_03 | 接近后退开；横向经过 |
| Big City / boulevard | boulevard_01、boulevard_04 | 接近后退开；横向经过 |
| Big City / residential | residential_02 | 横向经过 |
| Big City / big01 | big01_02 | 接近后退开 |
| Big City / big04 | big04_03 | 横向经过 |
| Small City / smallcity | smallcity_04 | 接近后退开 |

每个姿态设置后等待场景稳定再采样，标签间隔0.2秒，片段时间为0.0–4.6秒。这是预定姿态的准静态序列，不是连续引擎运动、实时硬件或帧率测量。相机高度参考保存的首个竖直碰撞命中点上方1.7米；该命中不保证是可行走地面，完整路径可行性没有由此得到认证。

beauty画面为1920×1080原生渲染，不是640×360上采样。传感器RGB及原生轴向深度保持640×360；三者记录并核对相同实际相机姿态和100°水平视场。展示分辨率与传感器输入分辨率分开，8×8代理、几何评分与 `CoreAlertPolicy` 不变。RGB不进入当前告警策略。

## 数据和收据

本机证据根为 `E:\linnan\linnan\artifacts.local\work\ba-city-showcase-20260920\`，经既有junction存储于F盘。以下路径相对此根：

- `plan.json`、`spec.json`、`protocol.json`：全部片段、来源镜位、时间和固定预算；协议不允许按告警结果筛选场景或改变策略。
- `specs/<region>.json`、`snapshot-seal.json`、`source/city_pcg_capture.py`：封存场景规格与专用采集快照；共享采集器和源地图不改写。
- `<region>/appearance/*.png`：1080p beauty；`model/sample/*.png`及`evaluator/native/*.npy`：传感器RGB/原生深度。
- `<region>/receipt.json`、`render-resource-health.json`、`process-release.json`、`hd-source-receipt.json`：采集姿态、源完整性、已知资源错误检查、进程释放和HD核查。

页面通过独立 `extra-data.js` 装载 `illustrative` 集合，评估truth为null；不显示TP/FP/TN/FN、准确率或误报统计，不加入原1296帧的27组指标。UNKNOWN与实际提醒均保留。视觉检查用于图像、材质、视角和演示可读性，不认证检测效果。

## 采集入口与交付检查

下面是已封存协议下的工程入口，不是让使用者重新执行的播放步骤。已有输出不应覆盖或重复生成；普通演示只需双击 `launch_demo.cmd`。

```text
python -B research/active/dtr-r0/nearfield/core_alert_demo/city_showcase_spec.py
python -B research/active/dtr-r0/nearfield/core_alert_demo/city_showcase_capture.py freeze
python -B research/active/dtr-r0/nearfield/core_alert_demo/city_showcase_capture.py capture --region smoke
python -B research/active/dtr-r0/nearfield/core_alert_demo/city_showcase_capture.py capture --region plaza
```

其余区域参数为 `boulevard`、`residential`、`big01`、`big04`、`smallcity`。`capture`完成后调用同入口的audit；独立检查入口为 `audit --region <region>`，收据使用排他写入保护。工程运行需要既有UE/Python环境，不是可移植离线播放的依赖。

完成全部区域后，`build_showcase.py` 从原生深度执行已有代理与冻结策略，并生成独立 `extra-data.js`；随后运行 `build_site.py` 装配页面。导出前检查原1296帧数据、策略与三个几何/代理源码的历史哈希。本机已用环境为 `E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe -B`，包含现有 NumPy/Pillow；未新增依赖。页面验收用 `node core_alert_demo/test_presentation.cjs`（相对于nearfield目录）核对原有27组指标、未标注隔离、192张PNG尺寸及跨片段保持复位。

冻结后仅为源码复现补齐 `city_showcase_spec.py` 的 `plan.json` 写入逻辑，因此当前builder哈希与原protocol记录不同；已保存的规格、计划、预算与采集设置未重写。早期Willow采集因用户转向City Sample而停止，其99帧和中断记录保留在 `ba-core-showcase-20260920/attempts/attempt-01-planter-scale/`；未执行的源码修订另存 `attempts/unexecuted-city-superseded-source/`，未加入本页面。

最终核查须记录实际片段/帧数、HD尺寸与共姿态/FOV、源地图完整性、资源健康、图像检查、浏览器操作和原1296帧指标复核，并保留失败/中断收据。任务进程退出后保留项目共享缓存；不把演示画面更清晰解释成真实硬件或自然场景能力提升。
