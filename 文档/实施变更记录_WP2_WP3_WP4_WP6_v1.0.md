# v1.0实施变更记录：WP-2、WP-3、WP-4、WP-6

状态：已实现并通过当前自动化测试；尚未切换旧训练入口，正式baseline尚未生成。

## 1. WP-2：传输、预约与生命周期

- `domain/reservations.py`：冻结半开区间、路径、预约请求和预约实体。
- `scheduler/transmission.py`：固定带宽流水线传输；数据只串行化一次，逐跳只累加传播时延；远程传输JIT结束于计算开始。
- `scheduler/resource_calendar.py`：CPU/BW真实峰值检查、不可变快照、版本冲突、全资源原子提交、失败回滚和完成/故障释放。
- `simulation/state_machine.py`、`simulation/event_engine.py`：显式任务状态、同时间戳结束先于开始、完成后释放和幂等推进。

## 2. WP-3：候选、队列与在线编排

- `domain/candidates.py`：冻结v1.0候选字段、确定性ID和候选到预约请求的直接映射。
- `scheduler/candidate_generator.py`：按节点×路径×完整全局时间网格枚举，不采样、不Top-K；执行SLA、预测覆盖、CPU和带宽硬过滤。
- `scheduler/queue_manager.py`：固定EDF复合排序、队列容量拒绝、Pending物理事件重激活和闭截止到期。
- `scheduler/reservation_manager.py`：版本冲突有限重试；真实失败不复用提交重试。
- `scheduler/v1_scheduler.py`：同批逐任务生成候选、选择、提交并立即更新快照；本地即时开算；未来到达任务不提前调度。
- 静态绝对不可承载任务确定性进入`Rejected(STATICALLY_UNSERVICEABLE)`；候选非空时策略必须选择，不存在Reject/WAIT动作。

## 3. WP-6：物理预测和实现账本

- `accounting/forecast.py`：严格分段常值电价/绿电预测，缺口明确失败，绿电禁止负数或非有限值。
- `accounting/energy.py`：线性CPU增量功率；仿真时间到小时；完整区间MWh和元积分；负外生电价保留。
- 候选边际系统成本按节点账单加入前后之差计算；不冒充最终逐任务账。
- 最终成本和绿电按同节点、同时间并发任务功率比例分摊，跨任务可加且与预约顺序无关。
- 零绿电供给时吸纳变化为`NOT_APPLICABLE(null)`，候选模型值为中性0并携带机会标志。
- `accounting/ledger.py`：完成事件才写实现账；追加幂等；完整结算后冻结。

## 4. WP-4：确定性策略与目标

- `scheduler/policies.py`：Earliest、LowestCost、HighestGreen、EqualWeight均只从非空物理候选选择。
- `scheduler/objectives.py`：成本与吸纳使用训练前固定尺度；等权主项为0.5/0.5；balance和Soft/Flexible线性迟到分项独立；生成稳定策略ID和Pareto前沿。
- 不使用测试候选集合的min/max重算归一化尺度。

## 5. 验证状态

- `test_scheduler_foundations.py`：31项基础预约/事件测试。
- `test_candidate_queue.py`：24项完整候选、队列和Pending测试。
- `test_v1_scheduler.py`：15项端到端调度与目标测试。
- `test_accounting_v1.py`：12项功率、成本、绿电和账本测试。
- 本记录生成前最近一次全量回归：141项通过；新增账本/目标测试已分别通过，下一阶段结束后再次执行全量回归。

## 6. 尚未完成

- WP-7正式三阶段评估、指标Schema与配对统计。
- WP-5候选式DQN、按物理时间折扣和回放契约。
- 旧`train.py`、`evaluate.py`主入口到v1编排器的最终切换及正式baseline重跑。
