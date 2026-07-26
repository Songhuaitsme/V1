# DQN 调度项目目录导航

本仓库同时保留旧版固定动作 DQN 和当前 v1 候选调度系统。源码已按代际和职责分类：

```text
shared/                 两代共用的配置、拓扑、价格、任务生成和基础设施
legacy/                 旧版 TensorFlow/Keras：节点动作 + WAIT
v1/                     当前 PyTorch Candidate-DQN 与正式调度链路
  accounting/           能耗、成本、绿电预测和完成账本
  domain/               任务、SLA、候选和预约契约
  evaluation_v1/        正式三阶段评估、指标和配对统计
  learning/             Candidate-DQN、特征、奖励和经验回放
  scheduler/            候选枚举、策略、资源日历和在线调度
  simulation/           状态机和离散事件引擎
tests/
  shared/               共享基础设施测试
  legacy/               旧版系统测试
  v1/                   当前 v1 系统测试
artifacts/
  legacy/               旧模型、TensorBoard、训练 CSV 和图表
  v1/                   v1 评估报告及后续模型输出
文档/                    需求、设计、验收与操作指南
```

## 常用命令

从仓库根目录执行：

```powershell
# 全量测试
py -3.9 -m unittest discover -s tests -t . -v

# v1 训练规模预检
py -3.9 -m v1.train_v1 --steps 200 --seed 9 --preflight-only

# v1 冻结策略评估
py -3.9 -m v1.evaluate_v1 --policy earliest_feasible

# 旧版轻量训练
py -3.9 -m legacy.train --lightweight
```

当前正式系统入口是 `v1/train_v1.py` 和 `v1/evaluate_v1.py`；`legacy/` 仅用于旧实验复现和兼容测试。
