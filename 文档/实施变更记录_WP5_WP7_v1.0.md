# v1.0实施变更记录：WP5、WP7与最终审计

日期：2026-07-18  
需求版本：1.0  
算法版本：1.0

## 1. WP5：Candidate-DQN

新增`learning/`正式学习链路：

- 固定18维候选特征及特征schema hash；
- 共享点式`Q(global_state, candidate)`网络，可处理可变候选数量；
- 掩码、确定平局和全负Q值选择；
- Double-DQN目标只读取下一时刻可行候选集合；
- 回放保存被选候选特征及下一可行候选特征矩阵；
- 估计即时奖励与完成/失败/到期实现修正分离；
- 折扣使用物理`elapsed_seconds`；
- checkpoint强校验模型版本、候选版本和feature hash；
- `train_v1.py`提供独立正式训练入口。

学习策略没有WAIT或Reject动作。候选非空时必须选择一个候选；候选为空由确定性调度层处理为Pending或Expired。

## 2. WP7：正式评估

新增`evaluation_v1/`：

- 到达阶段、接纳结算阶段、执行排空阶段三段式评估；
- 只在Completed后写实现账本；
- 输出TaskEvaluationRecord、SchedulingDecision、SeedMetrics和AccountingReport；
- 记录选中候选、同偏好最早反事实、主动等待及等待收益向量；
- 主动等待为空时条件均值、P95及正收益率为NOT_APPLICABLE；
- 零到达、零预约、零完成CPU小时、零绿能机会和非正baseline分母均使用显式状态；
- 禁止正式输出NaN/Infinity；
- 支持seed配对t区间、可复现配对bootstrap、相对变化和pilot功效样本量；
- 质量门槛区分PASS、FAIL和DIAGNOSTIC_ONLY；少于10种子不能形成正式通过结论。

## 3. 最终审计增强

- 新增`audit_v1.py`运行时物理不变量扫描；正式评估在每个事件批次末执行；
- 新增`traceability_audit.py`静态追踪审计；
- 新增显式实验性候选压缩模块，保留最早、成本、绿能、均衡及Pareto代表，并报告遗漏率和遗憾；
- 正式完整枚举与近似模式使用不同`CandidateMode`，结果不可混用；
- 新增正式JSON规范化序列化和非有限数定位拒绝；
- 新增19项跨模块契约验收测试。

## 4. 验证证据

- 全量单元/集成/契约测试：198项通过；
- v1.0正式目标测试ID：258/258；
- 需求ID：50/50；
- 算法不变量：19/19；
- 正式评估入口冒烟：通过；
- 正式训练入口冒烟：通过。

## 5. 明确未改内容

- 未改写或重算`evaluation/baseline`中的旧结果；
- 未把旧5种子结果升级为v1.0正式baseline；
- 未冻结pilot参数：迟到权重、学习率、DQN隐藏层宽度、正式种子量仍应由后续实验确定；
- 未删除旧调度代码，旧测试继续作为兼容回归存在。

