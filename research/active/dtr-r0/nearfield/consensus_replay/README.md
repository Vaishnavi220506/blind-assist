# A 与一致性补报：离线对照回放

完整回放已消费合成评估的 16 布局、48 片段、1152 帧。保留 A 为默认方案，实验对照为 `A_current OR (B_control_current AND N_current)`，随后执行不可递归的一帧保持，片段间重置。

## 使用

完整解压交付 ZIP，打开 `index.html`，Windows 也可双击 `launch.cmd`。页面不依赖网络、服务器、Python 或模型；图片和数据均随包提供。浏览器若限制本地文件，可用任意仅本机绑定的静态文件服务预览此文件夹。

点击“看补漏”“看误报”“看仍漏事件”可定位相关帧。可以播放、暂停、逐帧跳转和切换主画面方案；下方同时保留两种方案的告警、事件内首次报警、误报段和采样时长。上方完整数据统计不随筛选变化。Core/Boundary、HEAD/BODY 和真值仅用于浏览与评估展示。

UNKNOWN 独立保留，可以与补报或保持后的告警并存。事件检出表示相交窗口内至少一帧报警，不代表完整事件覆盖。误报时长为样本数乘 0.2 秒。片段不连续拼接为一次真实行走。

## 证据

这是封存预测及实际保存 RGB 的工程回放，没有新推理、训练、阈值修改或独立验证。Core 保持后从 159/25/17 变为 172/26/4（正确报警/误报/漏报），事件 15/16→16/16。Boundary 从 11/5/165 变为 62/10/114，但新增误报成本超预算，未整体晋级。不能作为设备、部署或安全效果证据。

源数据位于仓库 `artifacts.local/work/ba-data-coverage-20260921` 与 `ba-branch-disagreement-20260921`。随包 `build-receipt.json` 记录源哈希、图片身份及 JPEG 显示副本哈希；JPEG 仅缩放至最多 640×360 并编码，没有补画。

## 从源码重建

在仓库根目录使用安装 Pillow 的 Python：

```powershell
python research/active/dtr-r0/nearfield/consensus_replay/build_replay.py --output artifacts.local/work/ba-consensus-replay-rebuild/site
python -m unittest discover -s research/active/dtr-r0/nearfield/consensus_replay -p test_build_replay.py
node research/active/dtr-r0/nearfield/consensus_replay/test_app.cjs
```

输出目录必须不存在；构建器拒绝覆盖。原始封存数据必须在本机存在。生成数据和图片只进入忽略的 artifacts 树，不入 Git。
