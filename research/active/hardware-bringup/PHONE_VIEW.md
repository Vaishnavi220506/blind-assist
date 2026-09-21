# 手机观测台（USB 中转）

本阶段让安卓手机浏览器查看已有相机、ToF 网格与 CNH。链路为：

```text
Atom + XIAO --USB--> 电脑观测台 --手机 USB / ADB reverse--> 手机浏览器
```

电脑仍负责采集、解码与保存；手机显示并控制同一观测台。它不是手机独立采集、
无线传输、边缘推理或 Android App 的多区 ToF 适配。两块板保持现有固件，模拟路线不变。

## 打开和断开

先启动 [本机观测台](DASHBOARD.md)。手机开启 USB 调试，使用数据线接电脑，解锁后
在手机上允许 USB 调试。以下命令从仓库根运行：

```powershell
pwsh -File research/active/hardware-bringup/phone-view.ps1 -CheckOnly
pwsh -File research/active/hardware-bringup/phone-view.ps1
# 多台手机时明确选择；serial 由 adb devices -l 获取。
pwsh -File research/active/hardware-bringup/phone-view.ps1 -Serial '<serial>'
# 不再使用时，仅移除本脚本记录为自建的映射。
pwsh -File research/active/hardware-bringup/phone-view.ps1 -Serial '<serial>' -Disconnect
```

脚本验证设备授权和本机服务，建立同端口 `tcp:8766 -> tcp:8766` 反向转发，然后请求
手机浏览器打开 `http://127.0.0.1:8766/`。非默认端口用 `-Port` 与观测台保持一致；
ADB 不在已知位置时传 `-AdbPath '<adb.exe>'`。现有不同目标的映射不会被覆盖。
记录在 `artifacts.local/hardware-bringup/phone-view/`；只记录串号、端口、时间和映射
归属，不保存调试私钥。脚本成功仅表示发起浏览器打开，不能替代手机渲染/更新验证。

无需开放局域网监听、防火墙或安装 APK。手机页面的录制/停止等控制作用于电脑端；
电脑和手机共用采集状态，回放游标是各页面自己的。断开手机 USB 后页面无法继续
取得数据；已开始的电脑端有界录制仍会按原时长完成。断开转发不停止观测台。

## 验证与边界

验收先用已保存记录检查相机、16 区域与 CNH，再做一次有界实时查看，确认两路帧号
更新、错误可见；关闭手机连接后应显示连接失败而非继续冒充实时数据。
数据年龄继续表示电脑主机接收时间意义，不是手机端端到端延迟或曝光同步。

2026-09-21 手机实测已完成：Samsung SM-S9280（S24 Ultra），Android 16 / API 36，
三星浏览器。初次 `unauthorized` 在重连并由用户允许调试后解除；未更换调试密钥、
安装 APK、修改防火墙或刷写固件。

- 历史记录相机画面可见；本轮未在手机上完整验收回放时间轴操作。
- 一次 40 秒实时录制 `dashboard-20260921T130845Z-860901`：电脑端保存 821 张
  640×480 JPEG 和 204 帧 CNH，两采集器退出码 0，无损坏记录、序号缺帧或重排。
  相机接收跨度 37.032 秒、约 22.17 fps；这些是电脑采集统计，不是手机显示帧率。
- 手机截图可见相机帧号 6634 → 6944；ToF 帧号 36365 → 36440，4×4 原始网格、
  UNKNOWN 和 24 bin CNH 正常显示。CNH 可见不等于其距离轴或幅值已标定。
- 移除脚本自建转发后，手机显示“连接或读取失败”“连接中断 · 帧已过期”，
  相机旧图变暗。随后恢复转发。此处测的是 ADB 通道断开，未拔 USB 线。
- PowerShell 语法检查与路线 `git diff --check` 通过；实际走过未授权拒绝、
  只读检查、建立、已有映射检查、断开和重新建立。手机截图属于工程诊断证据，
  不是 Android App 的正式仪器测试。

原始流 SHA256：

```text
camera 708248c88969f38ad486610b89ee89f3246a7d71edebfda75f382449153f1148
tof    24e2ed211cd106cc459c6d645eb7eff7dbb81ad0959d9fcbe6b8ba32bce80aef
```

截图、原始无障碍树和哈希清单保留在 `artifacts.local/hardware-bringup/phone-view/`
的 `replay-check`、`live-a`、`live-b`、`live-c`、`disconnected` 与 `recovered` 目录。
无障碍导出中网页树未随截图同步更新，本轮实际画面与帧号判断采用截图；不以树文本
冒充实时更新证明。采集已自动结束；观测台与手机转发保留供使用，释放方式见上方。
