# CNH H3 学习读出演示：工程回放与实时入口

本轮设备不便接入，按用户更新后的范围交付程序和历史真实输入回放。
**不把回放称为实时测试，不提供真实传感器准确率或物理事件端到端延迟。**

入口：[cnh_h3_live_demo.py](host/cnh_h3_live_demo.py)。输出和运行收据位于
`artifacts.local/work/cnh-h3-live-demo-20260928/`，最终含 v5 阈值的历史真实输入视频为
[replay-v5-final/demo.mp4](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-final/demo.mp4)。
先前未加载阈值的 `replay/` 单独保留，不覆盖为最终结果。
显示明确标注“模拟训练模型、真实传感器输入、无定量结论”和 `REPLAY`。
相机—ToF 没有可用物理配准，因此并排显示相机和原始 8×8 分区热力图，
不把分区框伪装成物体像素位置。色标按帧缩放，格内数字是 max-bin z4，不是概率或距离。

## 处理定义

- 冻结的三个 NN checkpoint 平均 logits；输入顺序为 `log1p` 压缩后的 z4、z1。
- 从另一段明确静止背景记录估计均值和连续 k=1..4 帧和的逐通道方差。
  直接测量时间和的方差保留时间相关影响，但这是固定场景工程归一化，不是可靠物理噪声或 SNR 标定。
  不以独立通道相加近似真实 S2 窗口方差，**真实输入 A3 暂不可用**。
- 背景及推理都按主机接收时间因果采样到 5 Hz 网格：只取网格时刻前最新已接收帧，
  不插值、不重复一个传感器帧；缺帧/重置或空网格打断历史。传感器静止，输运为恒等。
- A2 使用最近至多五个 NN logits，权重逐帧减半后归一化，startup 同样归一化；
  测试与 `cnh_learned_memory_fusion.causal_ewma(alpha=.5, window=5)` 逐元素完全一致。
- 六个查询盒支持来自原几何函数、tau=.75 和名义 -10° 安装角；未测人体姿态。
  这与仿真中的输入分布有差异，不能由此展示片宣称避障有效。
- 仿真 calib 阈值仅用于展示触发。文件必须给出 `A2_thresholds` 的 HEAD/BODY、`source` 和 `scope`；
  缺阈值时只显示分数，不虚构报警。无论 v5 是否支持 A3，真实 S2 的未标定状态不被覆盖。

## 运行

在仓库根执行（无需新模型训练）。从本机 `config/local.toml` 读取已配置的
`research_python`，下方读取配置使用 Python 3.11+：

```powershell
$py = python -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("config/local.toml").read_text(encoding="utf-8"))["research_python"])'
$capture = 'artifacts.local/hardware-bringup/captures/simple-20260927T164234Z-cc4d0e'
$models = 0..2 | ForEach-Object { "artifacts.local/work/cnh-learned-readout-20260928/seeds/model_seed$_.pt" }
& $py research/active/hardware-bringup/host/cnh_h3_live_demo.py `
  --replay "$capture/02-object-attempt2" `
  --background "$capture/01-background-attempt1/tof/frames.jsonl" `
  --models $models --seconds 30 `
  --thresholds artifacts.local/work/cnh-track-a-v5-20260928/data/analysis/demo_thresholds.json `
  --out artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-final
```

上述 `--thresholds` 使用最终 v5 calib 阈值。每次输出 `report.json`、
`background_model.npz`、逐帧 `inference.jsonl`、`preview.jpg` 和 `demo.mp4`。
视频 CFR 仅用于编码，重复视频帧不计为新传感器帧。

后续设备可用时，把 `--replay` 换成明确 XIAO 的 `--port COMx`，
提供当前静止装置、当前环境的独立背景采集文件及明确 `--camera <URL或索引>`。
不要把旧场景背景文件用于新环境。串口默认与 H3 固件一致为 115200；不刷固件、不自动扫描网络。
串口分支使用 `pyserial`（当前 Torch 环境未安装，本轮没有为不可用设备安装）；
视频窗按 ESC 结束，串口、相机、编码器均在 finally 释放。
网格区间由下一次接收关闭，额外缓冲包含在真实运行的主机接收至显示延迟内。
真实采集与相机读取分支尚未验证，不宣称本轮完成了实时连接。

只有提供带验证证据的 `--registration` 才叠加到相机：JSON 包含
`verified: true`、`evidence` 和 `zone_polygons_normalized`（64×4×2，坐标在0..1）。
该文件应对应同一固定装置；不能用名义 FOV 或当前粗配准搜索冒充标定。

## 检查与边界

最终回放已完整解码为 **300 帧、10 fps、30.0 秒**，并实际查看第15秒画面。
冻结 v5 calib 阈值为 HEAD −0.4040601254、BODY −0.0805260986，选择 A2。
没有根据真实录制调参。**135 个预热后采样帧中，六个查询盒各135次触发**；
11个预热帧没有触发。持续触发意味着这段视频只能证明输入、推理、显示管线运行，
不能据此证明真实障碍定位、报警有效性或误报已受控。

| 工程测量 | 本次结果与分母 |
| --- | --- |
| 原始接收/用于推理 | 读取155帧（含闭合采样网格的后续帧）；146个唯一传感器序号用于30秒视频 |
| 原始帧率/使用帧率 | 5.109 / 4.866 Hz；按接收时间因果取5 Hz网格，空网格不重复旧观察 |
| 回放处理吞吐 | 146帧 / 4.831秒 = 30.22 fps；包含推理、渲染、编码，不是现场帧率 |
| 每帧处理、渲染、编码时间 | 中位数27.51 ms、P95 32.84 ms；不是物理事件端到端延迟 |
| 相机接收时间配对 | 146/146帧；接收时间差绝对值P95为63 ms，非曝光同步 |
| 背景归一化样本 | 143个采样帧；k=1/2/3/4连续和方差窗口143/136/131/126个 |

完整证据：[运行收据](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-final/report.json)、
[视频解码与触发计数](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-final/video_validation.json)、
[预热后画面](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-final/preview-postwarmup.jpg)。
本轮没有现场相机、串口或已标定叠加测试；没有遗留任务进程。

[聚焦测试](host/test_cnh_h3_live_demo.py)覆盖冻结平滑逐值一致、因果采样、背景时间相关/缺帧、
拒绝未验证配准。回放视频同时检查时长、可解码性和画面标签。
`report.json` 分开报告实际原始帧率、使用帧率、编码帧率及处理/渲染/编码耗时；
回放吞吐不是真实端到端帧率。相机最大150ms主机接收时间近邻配对，仍不是曝光同步。
