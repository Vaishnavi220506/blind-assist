# 离线准备验证，2026-09-21

本轮不打开串口、不连接远程设备、不刷写固件。

## 已完成

- prepare.ps1 在独立本地目录建立 Python 3.12 环境，导入 pyserial/esptool 成功，
  实际依赖版本保存在 `artifacts.local/hardware-bringup/environment-freeze.txt`。
- Arduino CLI 1.5.1 + ESP32 core 3.3.8 编译三份项目内源码：

| 固件 | 程序占用 | 静态 RAM | 本轮实机验证 |
|---|---:|---:|---|
| i2c_probe | 332840 bytes | 23480 bytes | 无 |
| tof_reader | 427968 bytes | 34168 bytes | 无；已有上一轮测距证据 |
| tof_cnh | 429768 bytes | 40488 bytes | 无；上一轮握手失败 |

每份构建保存 compile.log、依赖/源码哈希和 build-manifest.json；未执行上传。

- 18 项 host unittest 通过，包括 UNKNOWN、形状/类型错误、断帧、重复、重启、
  signed scaler、原始字节保留、输出不可覆盖、串口异常/中断释放、固件哈希、
  全零 CNH，以及不依赖 pyserial 的离线 CLI。
- 两段已有实机日志回放得到 81 / 103 帧，缺失序号均为 0；后一段中心中位数 594 mm。
- 已有 CNH 失败日志得到 0 帧、6 个设备错误记录，退出码 2；没有空集成功声明。
- 16 宫格 CNH 绘图通过合成输入检查；图中明确标记 SYNTHETIC / NOT HARDWARE。
- 离线资料库收录 14 份官方资料、2 份 RTrobot 供应商资料；原件和检索正文均有哈希。
- 核对 UM3183 §5.7 后，CNH 缩放按 `raw / 2**scaler`，修正供应商示例的两倍差异；
  原始数据不变，当前没有成功实机 CNH 数据可据此重算或宣称验证。

回放与合成测试保存在 `artifacts.local/hardware-bringup/host-validation-20260921-173752/`。
修正公式后的合成回放与图位于其中的 `synthetic-cnh-plot-um3183/`；旧图保留，
新旧回放的原始字节 SHA256 相同，只有归一化结果改变。

## 项目级检查边界

文档索引检查通过。结构检查识别新增独立路线，但仍报告三项既有超长文档：
`docs/PROJECT_STATE.md`、`docs/CURRENT_DECISION.md`、`research/active/dtr-r0/CURRENT.md`。
这些是本轮开始前的其他工作，本轮未修改其内容，也未为此清理模拟路线。
未运行 Android 全量构建或模拟评测，因为本轮不修改它们的运行路径。
