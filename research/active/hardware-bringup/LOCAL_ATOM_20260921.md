# AtomS3R-M12 本机独立相机读出

2026-09-21 完成 USB JPEG 请求式读出。最终固件 `atom-usb-jpeg-v3-otg`，
640×480 JPEG，USB-OTG CDC；未启用 Wi-Fi、ToF 或避障判决。

## 已验证

- 芯片 ESP32-S3-PICO-1，8 MB Flash / 8 MB PSRAM；初始化日志 PSRAM 为 8388608 bytes，
  相机初始化返回 0，传感器 PID 13920（0x3660）。
- 20 秒采集预算内，实际运行 17.046 秒，472 张 JPEG 全部成功解码并验证尺寸。
  工具为下一帧预留完整 3 秒超时预算，所以没有强行采满 20 秒。
- 接收速率 27.706 fps；已请求图片的序号 1–472 连续，无传输/解码错误。
  这是本次短采集的接收速率，不是长期稳定性或完整传感器曝光帧率。
- 固件编译通过：程序 448626 bytes，静态 RAM 62980 bytes；上传哈希校验通过。
- 主机 13 项 mock 测试通过，覆盖分片/批量读取、超时、错误 JPEG、不可覆盖证据、
  序号异常、关闭端口和两种 USB 控制线模式。无 Android 或模拟运行路径修改。

![本机 Atom 样张](../../../artifacts.local/hardware-bringup/atom-local/capture-20260921-usb-otg-vga/frame-000235-seq-0000000236.jpg)

场景由用户摆放，样张能看到桌面和物体；未检查焦距、曝光、色彩或相机标定误差。
JPEG 只是本地开发采集证据，不随源码上传。

## 已保留的失败与修复

| 固件 / 接收端 | 完整图片 | 停止原因 |
|---|---:|---|
| v1，硬件 CDC，逐字节头部读取 | 37 | 下一张 JPEG 尾部超时 |
| v2，分块发送/扩大 TX 缓冲/flush | 25 | 下一张 JPEG 尾部仍超时 |
| v2，主机批量读取＋Windows 1 MB RX 请求 | 124 | 速率提高，但下一张尾部仍超时 |
| v3，USB-OTG CDC＋主机批量读取 | 472 | 正常达到本轮采集边界 |

前三段字节流、事件、部分 JPEG、摘要及 v1/v2 固件副本均保留。
不能把修复仅归因于主机 RX 大小；最终通过的配置切换了 USB 实现。
本轮没有继续修改 Arduino core 或推断其确切底层缺陷。

## 使用与恢复

本机当前 Atom 为 COM11，XIAO 为 COM5；端口可能随插口或 USB 模式改变，运行前应枚举身份。
原 Atom 是旧 `atoms3r_m12_tof4m_stream_r18_direct_camera_buffer`，报告缺少单区 ToF4M。
其 8 MB Flash 已完整备份，SHA256 为
`d1778ae36d0a39dc6441045aa6f04130446819d937997392153efd5620f9db36`。
备份可能包含原网络配置，仅放在本地忽略目录，禁止随源码发布。

```powershell
$py = 'artifacts.local/hardware-bringup/venv/Scripts/python.exe'
& $py -B research/active/hardware-bringup/host/atom_capture.py --port COM11 --usb-otg --seconds 20 --output '<新artifact目录>'
```

`--usb-otg` 在打开前设置 DTR=True、RTS=False，供 TinyUSB CDC 建立连接；
默认硬件 CDC 模式仍为 False/False。工具不会扫描其他相机、改 Wi-Fi 或刷写固件。
原始串口流、每张 JPEG 与 SHA256、设备读出时间、主机请求/接收时间均保存。
设备读出时间不是曝光时间，尚未与 XIAO 同步。

恢复旧固件仅在确实需要时执行：确认目标为 Atom，必要时按官方方法进入下载模式，
再用 esptool 将本地 `original-flash-8mb-20260921.bin` 写回地址 0，校验后复位。
不能把恢复文件写到 XIAO，也不能把旧固件名称当作当前固件身份。

原件、上传校验日志与恢复副本见
[`atom-local/`](../../../artifacts.local/hardware-bringup/atom-local/)，
每次采集的相对路径和哈希见 [Atom 证据索引](atom-evidence-index.json)。
引脚与电源控制依据 [M5Stack 官方相机示例](https://github.com/m5stack/M5AtomS3/tree/main/examples/Basics/camera)。
本轮限时采集已结束、COM11 已关闭，未留后台相机采集进程；设备保持 v3 固件。
