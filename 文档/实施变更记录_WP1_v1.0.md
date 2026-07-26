# v1.0实施变更记录：WP-1单位与领域对象

变更ID：`WP1-20260717-01`  
状态：已完成并通过本工作包测试  
授权依据：用户明确指令“按v1.0开始修改代码，先实施第一批基础改造：时间单位、任务Schema、SLA策略和迁移，并完成对应测试；暂不修改DQN。”

## 1. 需求与边界

- 工作包：WP-1单位与领域对象；
- 需求ID：R-01、R-02、R-03、R-17、R-18、R-47、R-48；
- 业务依据：F-01、F-02；
- 明确不修改：DQN/GNN结构、候选Q、回放、训练超参数、候选完整枚举、资源日历、baseline CSV和模型文件。

本批采用`legacy_shadow`接入：新任务先构造并校验不可变v1.0 `TaskSpec`，再生成现有训练入口可读取的显式legacy视图。legacy视图中的`latency_limit`继续保留旧入口含义，v1.0偏好与绝对限制通过独立字段提供。因此本批不宣称现有`network_env.py`已经实现v1.0候选窗口。

## 2. 修改文件与接口

| 文件 | 修改 |
| --- | --- |
| `domain/units.py` | 新增仿真时间、秒、小时、十进制MB、CPU工作量和电价单位转换；非法值显式失败 |
| `domain/models.py` | 新增`SlaType`、冻结`TaskSpec`、独立`TaskRuntime`、指标状态和旧任务迁移适配器 |
| `domain/sla.py` | 新增启动可行性、Soft/Flexible迟到比例以及Hard不适用模型编码 |
| `domain/__init__.py` | 暴露WP-1公共接口 |
| `config.py` | 冻结七个`"1.0"`版本字段；增加显式时间/数据单位；运行模式保持`legacy/approximate` |
| `task_manager.py` | 任务模板`None`改为`Flexible`；生成任务时先校验`TaskSpec`，再输出`legacy_shadow`视图 |
| `network_env.py` | 仅同步运行时类型名称和既有第三类SLA特征位置；未重写候选或奖励机制 |
| `time_conversion_calculator.py` | 改用统一时间转换器和有限值校验 |
| `test_domain_foundations.py` | 新增UNIT-001至UNIT-006、TASK-001至TASK-005、SLA-001至SLA-015及迁移/影子接入测试 |
| `test_cpu_occupancy_model.py` | 旧`None`运行时测试改为`Flexible`，数值行为保持不变 |

## 3. Schema与迁移结果

- 规范运行时SLA枚举：`Hard | Soft | Flexible`；
- Soft：`preferred_start_limit_sim=L`，`latest_start_limit_sim=1.2L`；
- Flexible：`preferred_start_limit_sim=L`，`latest_start_limit_sim=1.5L`；
- 旧`None`只有携带有效正数`latency_limit=L`时才能迁移；否则返回字段级错误；
- Hard迟到指标为`NOT_APPLICABLE`；模型影子编码为数值0并携带`applicable=false`；
- `TRAFFIC_DAY_DURATION_IN_SIM=288`派生`SIM_SECONDS_PER_UNIT=300`，`SCHEDULING_CYCLE=0.005`等于1.5秒；
- 新生成任务ID由管理器内单调序列组成，避免同周期随机碰撞。

## 4. 验证证据

执行环境：Python 3.9.20。

```text
py -3.9 -m unittest -v test_domain_foundations.py
Ran 33 tests
OK

py -3.9 -m unittest discover -v
Ran 75 tests
OK
```

回归基线为修改前42/42通过；修改后原测试与新增测试合计75/75通过。`dqn_agent.py`和`gnn_agent.py`未修改。

另执行一次无训练、无文件输出的入口烟雾检查：`TaskManager`生成的v1.0影子任务能够被现有`NetworkEnvironment.get_global_state()`和`evaluate_schedule_candidates()`读取；状态维度保持413，候选生成正常完成。这只证明兼容入口未断裂，不代表候选语义已经升级为v1.0。

## 5. 下一Gate

WP-1完成不代表v1.0调度机制完成。当前仍为：

- `SCHEDULER_ENGINE="legacy"`；
- `CANDIDATE_MODE="approximate"`；
- 当前候选窗口、JIT传输、区间资源日历和原子预约仍待WP-2/WP-3实施；
- 不应基于本批代码重新训练或生成正式baseline。

进入WP-2前应单独确认“日历/传输/状态机”修改范围。
