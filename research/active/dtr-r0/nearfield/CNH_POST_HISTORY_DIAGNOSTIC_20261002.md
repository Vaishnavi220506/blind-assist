# HIST之后的残留错误：描述性诊断

原判读 FRAME_HISTORY_NOT_ESTABLISHED_DEV 保留。所有31条浅擦碰和939条外侧清晰查询保留在分母中。

| unit/config/query | side/mode/context | .9m跨越帧 | 三臂首报帧 M3/AGG/HIST | 截止前余量 M3/AGG/HIST |
|---|---|---|---|---|
|96004/3/BODY|negative_x/1/none|14→15|未报/未报/未报|-0.2108/-0.2689/-0.2978|
|96031/36/BODY|positive_x/1/panel|13→14|14/14/14|-0.3132/-0.1420/-0.1610|
|96057/26/BODY|negative_x/0/panel|13→14|未报/未报/未报|-1.6450/-1.7270/-1.7236|
|96073/11/BODY|negative_x/1/none|13→14|未报/未报/未报|-2.0391/-2.2955/-2.2867|
|96082/35/HEAD|negative_x/1/panel|14→15|未报/未报/未报|-2.9950/-3.2899/-3.2905|

| 外侧清晰比较 | 新增首停 | 移除首停 |
|---|---:|---:|
|AGG_minus_M3|4|2|
|HIST_minus_M3|6|2|
|HIST_minus_AGG|2|0|

## 全分母分组

|集合/维度/值|n|M3/AGG/HIST首停|M3/AGG/HIST及时|
|---|---:|---|---|
|all_shallow/motion/0|5|4/4/4|4/4/4|
|all_shallow/motion/1|14|11/11/11|10/10/10|
|all_shallow/motion/2|12|12/12/12|12/12/12|
|all_shallow/side/negative_x|17|13/13/13|13/13/13|
|all_shallow/side/positive_x|14|14/14/14|13/13/13|
|all_shallow/condition/none|14|12/12/12|12/12/12|
|all_shallow/condition/panel|17|15/15/15|14/14/14|
|all_shallow/query/BODY|18|15/15/15|14/14/14|
|all_shallow/query/HEAD|13|12/12/12|12/12/12|
|all_outside/motion/0|397|29/29/30|29/29/30|
|all_outside/motion/1|363|35/34/35|34/33/34|
|all_outside/motion/2|179|5/8/8|5/8/8|
|all_outside/side/negative_x|384|41/41/42|40/40/41|
|all_outside/side/positive_x|555|28/30/31|28/30/31|
|all_outside/condition/none|503|41/41/41|40/40/40|
|all_outside/condition/panel|436|28/30/32|28/30/32|
|all_outside/query/BODY|446|35/38/39|35/38/39|
|all_outside/query/HEAD|493|34/33/34|33/32/33|

逐条增删、截止点、首报距离、分数余量及完整13帧分数见 diagnostic.json。未报警不能写成从未观测到；分数也不能归因给目标或背景。所有几何与模式标签仅用于本次事后描述。

## 可解释到的机制边界

三臂漏掉完全相同的5条：4条观察窗内一直未报，1条迟报。迟报96031/36/BODY都在frame14才报警，此时目标前沿0.7948m，比0.9m截止点晚0.1316s。96004/3/BODY与该迟报例的HIST原始logit在截止前曾高于固定平滑阈值，但平滑输出仍低于阈值；这表明有可见分数峰被时间平均压低，不等于已经证明换读出能满足误停预算。

其余3条截止前HIST平滑最大余量为−1.7236、−2.2867、−3.2905，原始分数也没达到现平滑阈值。96073/11在截止后分数上升，整窗最大余量收窄到−0.6119，仍未报。只凭这些分数，不能区分传感器没有观测、表征失效或分类器抑制。

HIST相对M3增加6条外侧误停、移除2条；增加者5条panel、1条none，均名义外侧12.3–17.9cm。首停时实际目标前沿1.17–3.14m，目标中心相对行进轴约12.5–22.5°；这是几何描述，不是传感器视轴或目标归因。6条新增的整窗最大超阈余量仅0.0028–0.0914，其中5条末帧又降回阈值以下。相对AGG新增的两条均panel，余量仅0.0096、0.0028，没有移除任何外侧误停。

因此残留错误同时包含时间平滑边界与较强的截止前低分；逐帧历史并未改变浅漏停集合。外侧增删主要是阈值附近的小幅分数变化，不能据此断言更丰富的历史完全没有信息，也不能把panel或motion元数据直接做部署门控。

## 结论与证据入口

这是对保留结果的事后描述，没有新判据、新候选、新训练或新推理。三臂未改变浅擦碰漏停集合；外侧误停增删集中在阈值附近，不能据此识别物理不可观测或目标/背景因果来源。原 FRAME_HISTORY_NOT_ESTABLISHED_DEV 判读不变。

- 原结果：[CNH_FRAME_HISTORY_RESULTS_20261002.md](CNH_FRAME_HISTORY_RESULTS_20261002.md)。
- 可复现脚本：[cnh_post_history_diagnostic.py](cnh_post_history_diagnostic.py)。
- 本地载荷：`artifacts.local/work/cnh-post-history-diagnostic-20261002/diagnostic.json`（全部31浅/939外侧分母、逐条事件、分数及输入哈希）。
- 运行：`E:/codex-tools/bin/blindassist-research-gpu.cmd -B research/active/dtr-r0/nearfield/cnh_post_history_diagnostic.py`。此命令仅做缓存CPU统计，不调用模型或GPU。
