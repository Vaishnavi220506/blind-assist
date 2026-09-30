# Forward perception run log

| Date | Code version | Configuration changes | Metrics (units/denominators) | Conclusion / result link |
| --- | --- | --- | --- | --- |
| 2026-10-01 | `9432fc2b` + 重跑控制修复 | 移除 PLAN/freeze 哈希锁，每次运行写独立目录 | 运行控制夹具：连续启动 2/2、失败隔离 1/1，旧输出保留；未运行 GPU 诊断 | 可重跑；报告精简，[验证记录](../../../artifacts.local/work/cnh-shallow-boundary-rerun-check-20261001/verification.json) |
| 2026-10-01 | `ff1d1e99` + `cnh_plane_residual{,_pilot}.py`，执行哈希见request | 用户授权物理平面分解小试；复用86000–86011，末帧3背景×6压力×内外目标；保留原始/墙/signed残差，无训练 | 12单位432目标状态，270增量非零；名义侧墙/大面板背景L2降低55.1%/44.8%，增量投影保留中位96.9%/96.7%；噪声拟合低于80%为29/270；墙占据控制仍可被解释93.9%；3个聚焦检查通过 | 联合通道值得继续；残差单独输入不采用。已消费Development机制证据，无LOFO BER、硬件或推广结论。[结果](../../../artifacts.local/work/cnh-plane-residual-20261001-pilot/REPORT.md) |
