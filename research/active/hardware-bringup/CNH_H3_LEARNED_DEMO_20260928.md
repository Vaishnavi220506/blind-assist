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

### 可选 sim-floor 输入对齐检查

默认仍为 `--input-mode empirical`，上述历史回放不变。可选
`--input-mode sim-floor --sim-floor-fields <显式字段.npz>` 仅提供与仿真特征计算对齐的代码路径，
**不从真实背景均值/方差猜测仿真参数，不据此认定真实输入已对齐**。
本轮真实录制没有所需字段，因此没有执行真实 sim-floor 回放。

NPZ 使用 `allow_pickle=False`，必须包含：

| 字段 | 约定 |
| --- | --- |
| `schema` / `source` | 标量字符串 `cnh.sim-floor.v1` / 非空来源说明 |
| `seq` | 唯一整数序号 `[N]`，与输入传感器序号精确对应 |
| `bias` | 有限数值 `[8,8,16]`，显式仿真串扰项；不是整段场景背景 |
| `ambient` | 非负有限数值 `[N,8,8]`，与原仿真 `16*ambient + max(bias,0)` 定义、单位一致 |
| `T_Q_tof` | 合法刚体变换 `[N,4,4]`，对应每个seq；不由图像或背景猜测 |

缺文件、字段、对应序号或合法数值时，输出 `not_available.json`，状态 `NOT_AVAILABLE`、
`fallback=false`；不退回 empirical。离线回放在启动推理前检查全段序号。
该分支直接调用冻结 `cnh_learned_features.sequence_features`，每步只保留最近至多4帧、
使用恒等输运和显式查询变换，保留原函数的float32算术及查询支持。
进入NN前也与冻结特征文件一致：**z4/z1先float16量化，再转float32，最后sign*log1p**。
返回用于检查的原始z1/z4保持原函数float32；两层数据不混为一谈。

验证使用非零bias、随帧变化的非零ambient、显式查询变换和历史重置，
z1、z4、支持掩码及量化后的NN输入均与实际冻结函数逐元素完全相同。
7项聚焦测试通过；另对原02段146帧检查默认empirical分支，NN/A2分数及历史/重置
与已保存回放逐值完全相同，见
[默认分支回归收据](../../../artifacts.local/work/cnh-h3-live-demo-20260928/sim-floor-default-regression.json)。

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
没有根据真实录制调参。**02物体段的135个预热后采样帧中，六个查询盒各135次触发**；
11个预热帧没有触发。更正解释：该30秒截取全程有物体，缺少负例，
**不能从02持续触发推出误报高或缺乏选择性**，也不能据此证明真实定位或报警有效。

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

### 恢复背景及运动段对照

补充回放保持脚本、三个模型、v5阈值和01背景完全不变，没有重训、调参或重新采集。
**04恢复背景（无新增物体）才是当前背景抑制未闭环的主要失败对照：
左/中HEAD查询在128个可报警帧中均持续触发128次。** 03运动段仅用于对照分数触发，
没有逐查询真值，不能据此计算真实检出率。下表分母均排除每次历史重置后的预热帧；
HEAD依次为查询0/2/4，BODY为1/3/5，对应左/中/右。

| 记录 | 可报警帧/全部唯一输入 | HEAD三盒触发数 | BODY三盒触发数 | 解码视频 |
| --- | --- | --- | --- | --- |
| 04恢复背景 | 128 / 144 | 128、128、14（各分母128） | 14、4、15（各分母128） | 298帧，29.8秒 |
| 03物体运动 | 133 / 145 | 133、133、126（各分母133） | 128、130、130（各分母133） | 296帧，29.6秒 |

04/03的16/12个预热帧均无报警；原记录较短，未补造到30秒。
两段视频均完整解码，并实际检查第15秒画面，保留REPLAY、未配准及无定量结论标签。
计数与来源见[对照汇总](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-controls/control_summary.json)，
[04视频](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-controls/04-restored-attempt4/demo.mp4)及
[04收据](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-controls/04-restored-attempt4/report.json)，
[03视频](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-controls/03-movement-attempt3/demo.mp4)及
[03收据](../../../artifacts.local/work/cnh-h3-live-demo-20260928/replay-v5-controls/03-movement-attempt3/report.json)。

原因尚未分离：真实输入减去整段背景均值并使用实测时间和方差，仿真训练则减串扰项、
使用包含环境光的噪声底；二者并不天然同分布。另需区分任务语义：这里的“空”仅是无新增手持物体，
台架原有柜面实测约0.635m（约64cm），距离本身仍在名义0.3–3m查询范围内；
其精确逐盒覆盖未标定。因此04既不能直接变成真实应用假警率，也不能以柜面存在为由宣称触发正确。
当前证据说明背景归一化、查询语义和模拟训练分数的衔接仍须验证，不能归因到其中单一因素。

[聚焦测试](host/test_cnh_h3_live_demo.py)覆盖冻结平滑逐值一致、因果采样、背景时间相关/缺帧、
拒绝未验证配准。回放视频同时检查时长、可解码性和画面标签。
`report.json` 分开报告实际原始帧率、使用帧率、编码帧率及处理/渲染/编码耗时；
回放吞吐不是真实端到端帧率。相机最大150ms主机接收时间近邻配对，仍不是曝光同步。
