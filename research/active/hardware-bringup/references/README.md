# 实机硬件参考资料

这份资料库属于 [独立实机路线](../README.md)。截至 2026-09-21，已下载并验证
14 份官方来源核心资料，另归档用户提供的 2 份 RTrobot 转接板资料。
下载不等于逐页审查，也不证明手中板卡的型号后缀、修订版或电路与资料完全一致。

原始 PDF / HTML 和可搜索 `.txt` 正文保存在忽略的
[`artifacts.local/hardware-bringup/references/`](../../../../artifacts.local/hardware-bringup/references/)。
Git 仅保存本索引与 [inventory.json](inventory.json)：包含官方来源 URL、获取时间、
相对路径、SHA-256、文件大小、PDF 页数、已观察版本和检查范围。复制项目时，若需要
离线阅读，应同时携带该 artifact 子目录；原文版权仍属资料发布方。

## 按任务查资料

| 要做什么 | 优先打开 |
| --- | --- |
| Atom 烧录、相机引脚、USB / Wi-Fi、供电 | Atom 产品指南；主板与扩展板原理图 |
| XIAO BOOT / RESET、USB、基础板引脚与供电 | Seeed 入门指南、引脚复用指南；不要套用 Plus 或 Sense 扩展板专属功能 |
| ToF 状态码、CNH 配置、bin 数值解释 | UM3183 第 5–6 节；VL53L8CH 数据手册 |
| ToF 视场、阵列方向、外壳窗口 | UM3183 §2.2；AN5894；AN5939 |
| 蓝板 VCC、I²C 电平、稳压与排针 | RTrobot 板级原理图，再对照 ST 芯片供电要求；不能只看芯片手册推断转接板供电 |
| 相机寄存器 / IMU 数据 | OV3660 / BMI270 数据手册 |

## 官方资料

以下相对文件名均位于 artifact 参考资料目录中。每项也有同名 `.txt`，方便本地全文搜索。

| ID | 离线原文 | 来源 / 已观察版本 |
| --- | --- | --- |
| `atom-guide` | [m5stack/atoms3r-m12-guide.pdf](../../../../artifacts.local/hardware-bringup/references/m5stack/atoms3r-m12-guide.pdf) | [M5Stack 产品页](https://docs.m5stack.com/en/core/AtomS3R-M12) 官方 PDF 导出；11 页 |
| `atom-main-schematic` | [m5stack/atoms3r-m12-main-schematic.pdf](../../../../artifacts.local/hardware-bringup/references/m5stack/atoms3r-m12-main-schematic.pdf) | 同一官方产品页的主板电路；1 页 |
| `atom-ext-schematic` | [m5stack/atoms3r-m12-ext-schematic.pdf](../../../../artifacts.local/hardware-bringup/references/m5stack/atoms3r-m12-ext-schematic.pdf) | 同一官方产品页的扩展板电路；1 页 |
| `ov3660` | [m5stack/ov3660-datasheet.pdf](../../../../artifacts.local/hardware-bringup/references/m5stack/ov3660-datasheet.pdf) | OmniVision 文档，M5Stack 官方托管；2011-05-13 preliminary specification |
| `bmi270` | [m5stack/bmi270-datasheet.pdf](../../../../artifacts.local/hardware-bringup/references/m5stack/bmi270-datasheet.pdf) | Bosch 文档，M5Stack 官方托管；Rev 1.3 / November 2020 |
| `xiao-guide` | [seeed/xiao-esp32s3-getting-started.html](../../../../artifacts.local/hardware-bringup/references/seeed/xiao-esp32s3-getting-started.html) | [Seeed 入门、引脚与供电指南](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)；页面标注 2026-05-14 更新 |
| `xiao-pin-guide` | [seeed/xiao-esp32s3-pin-multiplexing.html](../../../../artifacts.local/hardware-bringup/references/seeed/xiao-esp32s3-pin-multiplexing.html) | [Seeed 引脚复用指南](https://wiki.seeedstudio.com/xiao_esp32s3_pin_multiplexing/) |
| `xiao-schematic` | [seeed/xiao-esp32s3-v1.4-schematic.pdf](../../../../artifacts.local/hardware-bringup/references/seeed/xiao-esp32s3-v1.4-schematic.pdf) | Seeed 基础板资源链接；文件名与内部修订信息不一致，见下方 |
| `esp32s3` | [espressif/esp32-s3-datasheet.pdf](../../../../artifacts.local/hardware-bringup/references/espressif/esp32-s3-datasheet.pdf) | [Espressif 官方 PDF](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)；Version 2.2 |
| `vl53l8ch` | [st/vl53l8ch-datasheet.pdf](../../../../artifacts.local/hardware-bringup/references/st/vl53l8ch-datasheet.pdf) | [ST 官方 PDF](https://www.st.com/resource/en/datasheet/vl53l8ch.pdf)；DS14310 Rev 9 / July 2025 |
| `vl53l8cx` | [st/vl53l8cx-datasheet.pdf](../../../../artifacts.local/hardware-bringup/references/st/vl53l8cx-datasheet.pdf) | [ST 官方 PDF](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)；DS14161 Rev 12 / July 2025；保留用于型号差异核对 |
| `um3183` | [st/um3183-cnh-user-manual.pdf](../../../../artifacts.local/hardware-bringup/references/st/um3183-cnh-user-manual.pdf) | ST VL53L7CH / L8CH ULD 与 CNH 手册；Rev 7 / June 2026；24 页 |
| `an5939` | [st/an5939-cover-glass.pdf](../../../../artifacts.local/hardware-bringup/references/st/an5939-cover-glass.pdf) | ST VL53L8 盖板设计说明；Rev 3 / March 2025；18 页 |
| `an5894` | [st/an5894-fields-of-view.pdf](../../../../artifacts.local/hardware-bringup/references/st/an5894-fields-of-view.pdf) | ST ToF 视场说明；Rev 2 / March 2024；14 页 |

Seeed 当前“基础板 Schematic”链接文件名含 `v1.4`，PDF 内部标题却是
`XIAO ESP32-S3-Sense`、`Rev V1.3`、`2026-02-10`，并引用 `V1.5.kicad_sch`。
这里完整保留原文件并记录差异，不据此认定用户手中板卡修订号。实际基础板与 Sense
共用主板的哪些部分适用，仍须结合实物和具体电路核对。

网页快照保留 HTML 正文，但没有镜像外部 CSS、脚本和图片，离线版不保证原站排版。
同名 `.txt` 可直接离线检索；PDF 原件保留图、表及原始布局。PDF 提取文字也可能遗漏
图形中的接线信息，电源和引脚判断应打开原图核对。

## 用户提供的转接板资料

| 离线原文 | 来源及边界 |
| --- | --- |
| [supplier-rtrobot/VL53L8CX_Schematic.pdf](../../../../artifacts.local/hardware-bringup/references/supplier-rtrobot/VL53L8CX_Schematic.pdf) | 用户资料包 `OA-5_6(VL53L7_8CX)/Database/vl53l8ch/`；图纸标注 RTrobot、VL53LxCH、2024-01-28；不是 ST 评估板原理图 |
| [supplier-rtrobot/VL53L8CX_AssemblyDrawing.pdf](../../../../artifacts.local/hardware-bringup/references/supplier-rtrobot/VL53L8CX_AssemblyDrawing.pdf) | 同一用户资料包；图片型装配图，无可用文本层；本轮仅验证 PDF 可解析 |

供应商文档没有经验证的公开下载 URL，清单明确记录为用户提供，不冒充芯片厂官方文件。
`VL53L8CX` 文件名、内部 `VL53LxCH` 标注与已使用的 `VL53LMZ` 驱动不能各自单独证明
实物后缀；资料适用性与已读到的数据能力应分别记录。

## CNH 阅读要点与检索

UM3183 §5 明确区分原始 bin 起点、输出 bin 数和合并数；环境光有独立值，不能把 CNH
数组直接当作毫米距离。CNH 值由有符号整数与缩放量组合，内存预算还包括环境光和头部。
请对照当前固件所用 ULD 版本解释数据，不自动用新手册替换旧驱动的定义。

已核对一处示例与手册差异：UM3183 Rev 7 §5.7（PDF 第 17 页）给出的数值是
`raw / (2 ** scaler)`，而用户资料包的 `ESP32/VL53LxCH/main/examples/Example_12_cnh_data.c`
第 218、223 行使用 `raw / (2 << scaler)`。当 `scaler` 非负且位移有效时，后者分母
多一倍；负数位移也不应直接照搬。主机显示按手册指数缩放解释，采集文件必须保留
`hist_raw` / `hist_scaler` 和 `ambient_raw` / `ambient_scaler`，使结果可按明确版本重算。
这项数值解释不改变固件原始采集值，也不将旧示例换算结果静默覆盖。

头部大小需要区分结构层次。当前 [锁定驱动](../vendor-lock.json) 的
`vl53lmz_plugin_cnh.c` 定义持久区头部 `CNH_PER_HEADER_BYTES = 5*4`（20 字节），
另有每个缓冲区头部 `CNH_PER_BUFFER_HEADER_BYTES = 2*4`（8 字节）；地址函数先跳过
5 个 word，再从所选缓冲区的 `p[2]` 取直方图起点。单缓冲结构合计 28 字节，因此
不能把它简单写成“旧版 20 / 新手册 28”的矛盾。Rev 7 §5.8 的 28 字节概述不能替代
具体版本的 ping/pong、可选块和 4 字节对齐规则；实际布局使用锁定插件的
`vl53lmz_cnh_get_block_addresses()` 解析，不硬编码更改头部，不因新手册自动升级驱动。

在仓库根目录执行，例如：

```powershell
rg -n 'CNH|start_bin|sub_sample|6160|value format' artifacts.local/hardware-bringup/references/st/um3183-cnh-user-manual.txt
rg -n 'Power Pins|Battery Usage|D4|D5' artifacts.local/hardware-bringup/references/seeed/*.txt
```

本轮检查覆盖所有原件的文件头、PDF 解析或 HTML 正文、大小与 SHA-256；额外查看了
版本页、官方入口、UM3183 的方向/CNH 段落，以及 Seeed 修订信息差异。
其余页面仅自动提取供后续按需阅读，未宣称完成全量电气审查。
