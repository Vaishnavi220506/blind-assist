# CNH 近峰 / 后峰对照回放

这是已保存记录和判断的可视化工具，不是新实验、实时检测器或新证据。
用户暂不方便做改变背景的实物对照，所以先完善电脑端回放。

观测台首页新增“查看 CNH 近峰 / 后峰对照回放”，地址：
`http://127.0.0.1:8766/cnh-review`。页面可脱离硬件使用，也可直接打开生成的
`artifacts.local/hardware-bringup/cnh-replay-20260921/index.html`。
相机图片、数据和脚本全部内嵌，没有网络依赖。

## 查看方法

1. 点击“背景 / 近物 / 半遮挡 / 移开书”切换四段，播放或拖动时间轴逐帧看。
2. 点击 4×4 网格选择任意原始区域。默认 Z6 是展示示例，全部 16 区都可查看。
3. 看完整有符号 CNH 和后段局部放大图；参考曲线是前两秒的中位模板，不是同一时刻。
4. 下方状态直接读取保存的近端/后峰/双证据判断、固定门槛及幅值外推标记。
   参考和校准帧显示“未评分”，不显示成未检出；参考不合格区域显示“不可评估”。
   标量 UNKNOWN 与 CNH 状态分别显示，不用后峰替换原始距离。

回放包含每段中间 4 秒，总共 83 帧 ToF；准备动作和阶段间等待被跳过。
572 个已保存的区域帧判断逐项保留，其余区域/时刻不补算。它不是完整录像的连续时间轴。
相机选择接收时间最近的帧，允许在 ToF 接收之前或之后，最大差 500 ms，并在画面下
显示有符号差值。这是离线对照，不是在线因果输入，也不是曝光同步。
本页规则区别于普通观测台回放的“取游标之前最后一帧”。网格没有叠到相机上，
原始区号不等于已标定的画面位置。

## 生成与身份检查

```powershell
& artifacts.local/hardware-bringup/venv/Scripts/python.exe -B `
  research/active/hardware-bringup/host/cnh_replay.py `
  --run artifacts.local/hardware-bringup/paired/dashboard-20260921T144239Z-8a8beb `
  --report artifacts.local/hardware-bringup/cnh-late-echo-20260921-v2/report.json `
  --previous-report artifacts.local/hardware-bringup/cnh-components-20260921-v1/report.json `
  --capture-protocol research/active/hardware-bringup/cnh-components-protocol.json `
  --output artifacts.local/hardware-bringup/cnh-replay-20260921
```

输出目录必须不存在，已有输出不覆盖。原始流、帧索引、点击时间窗、旧报告身份及
所用 JPEG 哈希都须与保存记录匹配。导出器不调用分类器、不修改评分报告或源文件。
漏掉评分、帧数改变、重复身份和状态不匹配均拒绝导出。`manifest.json` 保存来源和
页面、模板、导出器哈希，便于确认显示内容的身份。

研究结论仍以 [弱后峰报告](CNH_LATE_ECHO_20260921.md) 为准：7/11 区域的局部探索结果，
240/242 半遮挡区域观测涉及幅值范围外推，不能称为普遍双目标检测或测距恢复。
本次不重新登记算法实验，也不修补既有中央账本错误。

## 本次验证

导出成功：83/83 帧有通过 JPEG 哈希检查的配图，最大绝对主机接收时间差为 94 ms；
572 条保存判断无遗漏。独立页面约 2.44 MB，生成清单记录来源及页面 SHA256。
3 项导出器聚焦测试、7 项既有观测台测试通过。浏览器连接器当时不可用，改用本机
Chromium 无头浏览器验证 HTTP 入口与直接文件打开；四阶段、全 16 区状态、
UNKNOWN 与 CNH 正证据并存、纯近物/恢复控制、拖动/单步/播放/暂停/末尾停止及
390 px 窄屏无横向溢出均通过，无页面 JavaScript 异常。桌面及窄屏截图已查看。
详细回执和截图保存在输出目录 `browser-check.json`、`desktop.png`、`mobile.png`。

新增回放是只读 GET 页面，既有观测台仅在确认未录制后重启，并载入原记录回放；
没有打开硬件串口。测试用浏览器已关闭，本机 8766 观测台保留供使用；从首页
“关闭本机服务”可释放它。
