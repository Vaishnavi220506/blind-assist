# 实机采集记录约定 v1

传输暂为 USB CDC newline-delimited JSON。Wi-Fi 和手机协议尚未接入。
波特率参数为 115200；hardware USB CDC 的实际吞吐不等同于物理 UART 波特率。

## 测距帧

`type=frame`，`seq` 是当前固件启动后的序号；`ms` 是 MCU 毫秒计数；
`stream` 是驱动流计数。`rows/cols` 为 8×8，`distance_mm`、`target_status`、
`nb_target` 各 64 项，数组顺序是驱动原始 zone 顺序，尚未做相机投影标定。
当前每区输出一个目标。不允许据此排除第二个目标。

## CNH 帧

`type=cnh_frame`，`rows=4`、`cols=4`、`bins=24`。
距离、状态、目标数各 16 项；`hist_raw`、`hist_scaler` 各 16×24；
`ambient_raw`、`ambient_scaler` 各 16 项；`cnh_header` 保留 5 个 uint32。
配置为 start_bin=0、sub_sample=4、integration=20 ms、requested_rate=5 Hz。

归一化依据 [UM3183 Rev 7 §5.7](references/README.md)：`raw / (2 ** scaler)`，
主机用 `ldexp(raw, -scaler)`，同时适用于 histogram 和 ambient。
用户资料包 Example_12 使用 `raw / (2 << scaler)`，比手册值小一半；本路线不沿用
该示例公式。原始整数和缩放值永远保留，实机幅值与当前驱动适用性仍待验证。
绘图横轴使用 bin 编号，不把它直接解释成经过标定的测距值。
基本 bin 常数约 37.5348 mm，本配置作 4:1 分组；这不是测距精度指标。

## 有效性与时间

### Atom USB JPEG

固件 `atom-usb-jpeg-v3-otg` 接收 ASCII `CAPTURE\n`，返回一行 JSON：
`type=jpeg`、`seq`、`device_readout_us`、`width=640`、`height=480`、`bytes=N`；
随后恰好 N 字节 JPEG，再跟一个换行。主机一次仅发出一个请求。
`device_readout_us` 是取得帧缓冲之后的 MCU 时间，不是曝光时间。
主机另外记录请求、头部接收、JPEG 最后字节接收的 monotonic 时间。
这只建立单设备传输记录，没有建立与 XIAO 的时间映射。
相机使用 USB-OTG CDC，主机加 `--usb-otg` 开启 DTR；它不控制相机曝光或硬件 GPIO。
非 JPEG 的启动/错误行、异常字节和未完整收到的尾部保留在原始日志中。

### ToF 有效性

展示距离要求目标数 > 0、status=5、distance_mm > 0；其余显示 UNKNOWN。
这是保守查看规则，原始状态 9 等其他结果并未从日志删除。
CNH 全零、非有限值、尺寸不符、帧序号跳变、设备重启应在摘要中可见。
主机 monotonic 接收时间只描述主机观察，不代表曝光时间或完成跨设备同步。

## 证据边界

初始化 `step` 和失败 `error` 也属于原始记录，不能只留下成功帧。
保存固件 SHA256、配置、采集持续时间和摆放条件；输出目录不可覆盖旧采集。
离线回放与构造测试不计作硬件新采样。实机读出与校准状态见 [CURRENT.md](CURRENT.md)。
采集工具的 `UNVERIFIED_ON_HARDWARE` 是准备期固定标签，不随帧出现自动晋级；
实际硬件证据以场景、原始日志和人工检查记录为准，不能用该字段判断是否收到了 CNH。

## 并行采集与离线对应

`pair_capture.py` 启动两个既有采集器，分别保留在 `camera/`、`tof/`，根目录保存
端口身份、调用参数、源码哈希、起止时间、退出码和子摘要。限时结束后关闭串口；
异常时仅终止本次启动的子进程。相机为末次请求预留超时，实际采集窗口可比 ToF 短。

`pair_inspect.py` 在两路主机接收时间范围的交集中寻找最近相机记录，输出
`pairs.jsonl`，关系固定为 `RECEIPT_NEAREST_ONLY_UNCALIBRATED`。等距取较早图片。
区间外数据、坏记录和 UNKNOWN 不删除；另外保存带排除原因的原记录副本。
ToF 时间来自包含行尾的 USB 读取块，相机时间来自 JPEG 接收完成；两路缓冲和曝光
延迟未知，因此接收时间差不是传感器同步误差。区域原始次序不能直接叠加到相机像素。
