# Forward perception run log

| Date | Code version | Configuration changes | Metrics (units/denominators) | Conclusion / result link |
| --- | --- | --- | --- | --- |
| 2026-10-01 | `9432fc2b` + 重跑控制修复 | 移除 PLAN/freeze 哈希锁，每次运行写独立目录 | 运行控制夹具：连续启动 2/2、失败隔离 1/1，旧输出保留；未运行 GPU 诊断 | 可重跑；报告精简，[验证记录](../../../artifacts.local/work/cnh-shallow-boundary-rerun-check-20261001/verification.json) |
