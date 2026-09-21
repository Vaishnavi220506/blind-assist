# 双路连续采集验收，2026-09-21

结论：本次请求 120 秒的双路录制通过有界传输检查；此前偶发字节损坏没有复现，
但未修复、未确定根因。本轮到此停止，不追加采集，不调整固件或测距参数。

## 已有损坏帧追查

来源：`paired/dashboard-20260921T122758Z-6b65b2/tof`，相对于
`artifacts.local/hardware-bringup/`。第 78 行 seq=23941，6022 字节，原始偏移
476514–482536（末端不含）。原始串口内容含 `174483040,,100094576` 和
`-172890624,-]]`，缺数字的形态已在 `raw.bin` 中存在。

该行有完整 JSON 起始标识、末尾字段与换行，之后还有有效 seq=23942；它不是文件
首尾的未完成帧。接收块记录显示该行恰好对应一个 6022 字节读取块。用 1 字节、
16384 字节和整文件分块回放均复现 77 有效帧、1 条 malformed_line、1 帧缺口；
可重复的 `transport_audit.py` 又使用 7/16384/整文件分块验证，全部与原存帧对象一致。
原始文件哈希和接收块覆盖也一致。因此不是 UI 或解析分块引入的损坏；设备输出、
USB 传输和系统接收之间仍无法进一步归因。固件逐项 Serial.print 的返回值未检查，
这是可查的方向，不是已经确认的原因。本次没有据此修改或刷写固件。

## 一次连续录制

来源：`paired/dashboard-20260921T123740Z-ff4b51`。本机 COM11 Atom / COM5 XIAO，
通过观测台有界采集接口启动，请求 120 秒，自然完成，两个子采集器退出码均为 0。
未要求用户移动设备，场景没有独立测量或约束。

| 项目 | Atom 相机 | ToF/CNH |
|---|---:|---:|
| 有效帧 | 2373 张 VGA JPEG | 613 帧，4×4×24 |
| 首末主机接收跨度 | 117.000 秒 | 119.734 秒 |
| 主机接收平均帧率 | 20.274 fps | 5.111 Hz |
| 帧间隔中位数 / P95 | 32 / 110 ms | 203 / 219 ms |
| 最大主机接收间隔 | 312 ms | 235 ms |
| 损坏 / 序号缺帧 / 重置 | 0 / 0 / 0 | 0 / 0 / 0 |

相机采集器既有策略会为最后一次请求预留完整 3 秒超时预算，因此约 117 秒停止发
新请求；不能声称相机覆盖了完整 120 秒。此次相机平均帧率低于此前约 25–28 fps
短测，尚未定位差异原因，不宣称固定帧率。没有超过观测台 1500 ms 提示门槛的
帧间接收停顿；这不是实时同步或延迟保证。

采集时全部 JPEG 通过解码检查，事后核对全部图像文件 SHA256；两路原始串口哈希
均与采集摘要一致。ToF 接收块完整覆盖原始文件，三种分块重放的 613 个 sensor
对象均与已保存 JSONL 一致。所有 manifest/summary 均已收尾。

无采集子进程残留；COM5 和 COM11 均通过 Windows 独占打开并立即关闭检查，没有
发送串口数据或设置控制线。详情保留为 `port-release-check.json`。本机观测台服务
留作回放，可通过页面“关闭本机服务”释放；服务不会自动重新采集。

```text
camera serial SHA256 0c8473f6a621bfe6dc4e41287ae46a2ff27817a4e60d6e00cfd2fd4219a6a0d5
tof raw SHA256       cb7f26594ee8e4eb9b7ead5360f8a4a4ad04dfea6cd07de42e37709784933e31
```

复核命令（输出必须使用尚不存在的文件名）：

```powershell
& artifacts.local/hardware-bringup/venv/Scripts/python.exe -B `
  research/active/hardware-bringup/host/transport_audit.py `
  --run artifacts.local/hardware-bringup/paired/dashboard-20260921T123740Z-ff4b51 `
  --output artifacts.local/hardware-bringup/paired/dashboard-20260921T123740Z-ff4b51/new-audit.json
```

两段均已生成 `transport-audit.json`，旧损坏段未通过，新段通过。结果只覆盖本次
短时传输与证据完整性，不证明长期可靠性、测距精度或避障效果。近零/UNKNOWN
仍保留。后续候选为一次原始 zone 与相机左右上下方向对应检查，本轮未启动。
