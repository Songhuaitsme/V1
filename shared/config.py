# ================= v1.0 元数据与兼容运行模式 =================
REQUIREMENTS_VERSION = "1.0"
ALGORITHM_VERSION = "1.0"
TASK_SCHEMA_VERSION = "1.0"
CANDIDATE_SCHEMA_VERSION = "1.0"
MODEL_SCHEMA_VERSION = "1.0"
METRIC_SCHEMA_VERSION = "1.0"
AGGREGATION_SCHEMA_VERSION = "1.0"

# WP-1只建立v1.0领域基础；现有训练/候选入口仍是legacy诊断模式。
SCHEDULER_ENGINE = "v1"
# v1.0 scheduler settings are independent from the legacy runtime switches.
# Complete enumeration is the frozen acceptance baseline; approximate search is
# an explicitly separate experimental mode.
V1_CANDIDATE_MODE = "layered_pool"
CANDIDATE_MODE = V1_CANDIDATE_MODE
V1_ABLATION_VARIANT = "reference"
V1_ACTIVE_WAIT_ENABLED = True
V1_CANDIDATE_PATH_K = 1
V1_TIME_TOLERANCE = 1e-9
# Bounded production candidate search. ``complete`` remains available as the
# exact audit baseline. Pool sizes adapt to the task SLA.
V1_CANDIDATE_POOL_MAX_BY_SLA = {
    "Hard": 128,
    "Soft": 256,
    "Flexible": 512,
}
V1_CANDIDATE_POOL_NODE_LIMIT_BY_SLA = {
    "Hard": 8,
    "Soft": 12,
    "Flexible": 16,
}
V1_CANDIDATE_POOL_TIME_SAMPLES_BY_SLA = {
    "Hard": 16,
    "Soft": 24,
    "Flexible": 32,
}
MAX_COMMIT_ATTEMPTS_PER_DECISION = 3
V1_DETERMINISTIC_POLICY = "earliest_feasible"
V1_OBJECTIVE_COST_WEIGHT = 0.5
V1_OBJECTIVE_GREEN_WEIGHT = 0.5
V1_OBJECTIVE_BALANCE_WEIGHT = 0.1
# Frozen from the complete-candidate, candidate-weighted calibration reservoir
# over seeds 1,2,4,7,9 (13,000,920 candidates; 2026-07-22).
V1_COST_REFERENCE_YUAN = 12682.284175062196
V1_COST_SCALE_YUAN = 287513.2364706354
V1_GREEN_ABSORPTION_DELTA_SCALE = 1.0
# Frozen paired-seed ablation (seeds 2 and 4, complete candidates, 2026-07-22):
# (0.5, 0.25) raised mean Soft/Flexible preferred-on-time rate from 0.75
# to 1.0 with unchanged completion/expiry/failure; (1.0, 0.5) added no benefit.
V1_SOFT_TARDINESS_WEIGHT = 0.5
V1_FLEXIBLE_TARDINESS_WEIGHT = 0.25
V1_GAMMA_PER_SECOND = 0.999
V1_DISCOUNT_MODE = "physical_time"
V1_DECISION_GAMMA = 0.95
V1_REWARD_ESTIMATE_ENABLED = True
V1_REWARD_REALIZATION_CORRECTION_ENABLED = True
V1_REWARD_TERMINAL_PENALTIES_ENABLED = True
V1_DISABLED_CANDIDATE_FEATURE_GROUPS = ()
V1_DQN_USE_GLOBAL_STATE = True
V1_DQN_DOUBLE_DQN = True
V1_TARIFF_MODE = "tou_uniform"
V1_CANDIDATE_DQN_HIDDEN_DIM = 128
V1_FORECAST_STEP_SIM = 2.0
V1_FORECAST_HORIZON_SIM = 5000.0
V1_TRAINING_POLICY = "candidate_dqn"
V1_COMPLETION_OUTCOME_REWARD = 0.0
V1_EXPIRATION_PENALTY = -1.0
V1_FAILURE_PENALTY = -1.0
V1_MAX_FORECAST_LOOKAHEAD_SIM = 4000.0
V1_TARGET_UPDATE_INTERVAL = 1000
V1_CANDIDATE_CHUNK_SIZE = 4096
V1_REPLAY_MIN_SIZE = 128
V1_TRAIN_UPDATES_PER_TRANSITION = 1
V1_CHECKPOINT_INTERVAL_CYCLES = 10000
V1_LOG_INTERVAL_CYCLES = 1000
V1_PREFLIGHT_MAX_TOTAL_SLOTS = 1000000000
V1_PREFLIGHT_MAX_SLOTS_PER_TASK = 20000000
# Includes both the two-pass policy selection and expected replay bootstrap
# regeneration.  This is a work-volume guard, not a candidate-semantics cap.
V1_PREFLIGHT_MAX_ESTIMATED_CANDIDATE_VISITS = 2000000000

# ================= 拓扑与基础资源配置 =================
DEFAULT_LINK_BANDWIDTH = 50000
DEFAULT_NODE_CPU = 400
FIBER_PROPAGATION_SPEED_KM_PER_S = 200000.0
LOCAL_LINK_DISTANCE_KM = 20.0
BACKBONE_DISTANCE_KM_BY_TIER_GAP = {
    0: 300.0,
    1: 900.0,
    2: 1800.0
}

# ================= 价格与计费策略配置 =================
ENABLE_DYNAMIC_PRICING = True
# Independent ablation switches for the dynamic-pricing components. Keep all
# enabled to preserve the original pricing behavior. Regional pricing also
# requires USE_UNIFORM_BASE_ELECTRICITY_PRICE = False.
ENABLE_CPU_UTILIZATION_MARKUP = True
ENABLE_TOU_PRICING = True
ENABLE_REGION_BASE_ELECTRICITY_PRICE = True
ENABLE_GREEN_SUBSIDY = True
ENABLE_CARBON_TAX = True
PRICE_ALPHA = 2.0
PRICE_BETA = 3.0
PRICE_TIER_MULTIPLIERS = {1: 0.85, 2: 1.0, 3: 1.2}
ELECTRICITY_LOAD_PER_YUAN_MW = 1.0
BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW = 1.30   
USE_UNIFORM_BASE_ELECTRICITY_PRICE = True
UNIFORM_BASE_ELECTRICITY_PRICE_YUAN_PER_MW = BASELINE_ELECTRICITY_PRICE_YUAN_PER_MW
MIN_CPU_PRICE = 1e-6
PRICE_LOOKAHEAD_STEP = 0.5
MAX_LOOKAHEAD_TIME = 20.0
TRAIN_EVERY = 4
CHECKPOINT_INTERVAL = 5000
PRICE_NORMALIZATION_FACTOR = 0.05

# 地区基础电价，单位可理解为 元 / MW / 单位仿真时间。
# 这里采用相对真实的仿真设定：负荷中心价格高，新能源富集地区价格低。
REGION_BASE_ELECTRICITY_PRICE = {
    'A': 0.86, 'B': 0.82, 'C': 0.78, 'D': 0.88, 'E': 0.84, 'F': 0.80,
    'J': 0.68, 'K': 0.62, 'L': 0.66, 'M': 0.64,
    'G': 0.50, 'H': 0.54, 'I': 0.56
}

# 分时电价倍率：先取地区基础价，再乘当前时段倍率。
# 谷段约 0.55，平段约 1.0，峰段约 1.45，尖峰约 1.65。
TOU_PRICE_HOURS = [
    0.0, 6.0, 8.0, 11.0, 13.0, 17.0, 19.0, 21.0, 23.0, 24.0
]
TOU_PRICE_MULTIPLIERS = [
    0.55, 0.70, 1.00, 1.35, 1.05, 1.30, 1.65, 1.25, 0.75, 0.55
]

# ================= 强化学习与训练配置 =================
# 600000 cycles ~= 10.42 simulated traffic days.
MAX_STEPS = 600000
BATCH_SIZE = 128
MEMORY_CAPACITY = 50000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
# Decay is applied per successful replay update, not per scheduling cycle.
# This reaches EPSILON_MIN after about 12000 replay updates, matching the
# shortened MAX_STEPS ratio from the previous 3000000-cycle schedule.
EPSILON_DECAY = 0.99975
LEARNING_RATE = 0.0001
USE_GNN_AGENT = False
GNN_HIDDEN_DIM = 128
GNN_EMBED_DIM = 128
GNN_DUELING_DIM = 128
GNN_NODE_FEATURE_DIM = 21

# ================= 环境预热配置 =================
ENABLE_ENV_WARMUP = True
ENV_WARMUP_CYCLES = 2000
WARMUP_FILL_REPLAY = False
WARMUP_RECORD_METRICS = False
WARMUP_POLICY = "random_valid_compute"
RANDOMIZE_INITIAL_GLOBAL_TIME = False

# ================= Dueling Double DQN + PER =================
DUELING_HIDDEN_DIM = 256
PER_ENABLED = True
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = 12000
PER_EPSILON = 1e-5
PER_FAILURE_PRIORITY_BOOST = 4.0
PER_LOW_CECI_PRIORITY_BOOST = 2.0
PER_CONSTRAINT_PRIORITY_BOOST = 3.0

# DQN 全局状态维度：
# task features 3 + time features 2 + compute nodes 51 * node_feature_dim 8 = 413
INPUT_DIM = 413
NODE_NUM = 51

# ================= 周期性调度配置 =================
GLOBAL_TIME_STEP_DURATION = 0.005
SCHEDULING_CYCLE = 0.005   #调度周期
BASE_TASKS_PER_SECOND = 25    #每秒生成任务数 正比于系统规模和负载水平，需根据实际情况调整
MAX_QUEUE_LENGTH = 500
MAX_TASKS_PER_CYCLE = 200

# Capacity-aware task generation.
# The long-run offered CPU-time load is normalized by total system CPU capacity.
# The external arrivals in one scheduling cycle are capped by a multiplier of
# per-cycle CPU-time supply: total capacity * scheduling cycle.
ENABLE_CAPACITY_AWARE_TASK_GENERATION = True
TASK_LOAD_TARGET_UTILIZATION = 0.7
TASK_PEAK_LOAD_MULTIPLIER = 1.1

# ================= 业务时间模拟配置 =================
TRAFFIC_DAY_DURATION_IN_SIM = 1440.0/5
SIM_SECONDS_PER_UNIT = 300.0
# 1仿真单位 = 86400 / 288 = 300秒
# 1个cycle = 0.005 * 300 = 1.5秒

# ================= 任务属性 =================
TASK_DATA_SIZE_RANGE = (50, 200)
TASK_CPU_DEMAND = (10, 300)
TASK_DURATION_MEAN = 25
TASK_DURATION_STD = 15
TASK_LATENCY_LIMIT_RANGE = (0.1, 25)    #暂时用处不大
TASK_BW_DEMAND = (10, 50)

# v1.0显式单位名；旧键暂时只供legacy训练入口读取。
TASK_DATA_SIZE_MB_RANGE = TASK_DATA_SIZE_RANGE
TASK_BANDWIDTH_DEMAND_MBPS_RANGE = TASK_BW_DEMAND

# 四类任务的基础占比。最终采样概率还会叠加时间潮汐和区域修正。
TASK_TYPE_BASE_RATIO = {
    "Realtime_Service": 0.3,
    "Interactive_Query": 0.35,
    "Data_Intensive": 0.20,
    "Model_Training": 0.15
}

# ================= 重试机制 =================
MAX_RETRIES = 5
MAX_RETRIES_BY_SLA = {
    "Hard": 1,
    "Soft": 3,
    "Flexible": 5,
}
RETRY_PRIORITY_WEIGHT = 5.0

# ================= WAIT 动作惩罚配置 =================
WAIT_PENALTY_PARAMS = {
    "Hard": {"base": 3.0, "urgency": 8.0, "retry": 4.0, "queue": 2.0},
    "Soft": {"base": 0.8, "urgency": 3.0, "retry": 2.0, "queue": 1.0},
    "Flexible": {"base": 0.2, "urgency": 0.8, "retry": 1.0, "queue": 0.5},
}

# ================= WAIT 收益估计配置 =================
WAIT_GAIN_COST_WEIGHT = 1.0
WAIT_GAIN_GREEN_WEIGHT = 0.5
WAIT_GAIN_BALANCE_WEIGHT = 0.1
WAIT_GAIN_LATENCY_WEIGHT = 0.2
# None means estimate WAIT gain across the remaining SLA delay window.
WAIT_GAIN_LOOKAHEAD_CYCLES = None
WAIT_GAIN_MAX_LOOKAHEAD_SAMPLES = 30
WAIT_GAIN_THRESHOLD = 0.0
WAIT_GAIN_PENALTY_WEIGHT = 0.1

CHECKPOINT_DIR = r"D:\project\DQN\basic\artifacts\legacy\logs\DQN_CHECKPOINT"


# ================= 绿电与碳排放计价配置 =================
# 节点基础功耗换算 (假设每个 CPU 单位消耗 10KW)
# 这样 5000 容量的节点满载时，功耗为 50 MW，符合中大型数据中心特征
CPU_POWER_UNIT_MW = 0.01
# v1.0 canonical physical name.  The legacy alias remains read-only during
# migration so historical diagnostics keep their original behavior.
INCREMENTAL_CPU_POWER_MW_PER_CPU = CPU_POWER_UNIT_MW
LEGACY_ELECTRICITY_TARIFF_INPUT_UNIT = "yuan_per_kwh"

# 绿电补贴系数 (当出现"弃风弃光"时，极限折扣率，0.8 表示最低可至 2 折)
GREEN_SUBSIDY_RATE = 0.8

# 碳排放惩罚税率 (当绿电耗尽，必须购买电网火电时，加收的溢价比例)
CARBON_TAX_RATE = 0.5

# ================= 算电协同评价指标配置 =================
CECI_GREEN_MATCH_WEIGHT = 0.35
CECI_GREEN_ABSORPTION_WEIGHT = 0.65
CECI_GREEN_WEIGHT = 0.8
CECI_COST_WEIGHT = 0.1
CECI_BALANCE_WEIGHT = 0.1

# ================= 绿电优先奖励配置 =================
ENABLE_SYSTEM_GREEN_REWARD = True
GREEN_MATCH_REWARD_SCALE = 4.0
GREEN_ABSORPTION_DELTA_REWARD_SCALE = 120.0
GREEN_WASTE_PENALTY_SCALE = 3.0
GREEN_LOAD_COVERAGE_REWARD_SCALE = 3.0
GREEN_ABSORPTION_DELTA_CLIP = 0.05

ENABLE_REBALANCED_SUCCESS_REWARD = False
SUCCESS_REWARD_HARD = 15.0
SUCCESS_REWARD_SOFT = 10.0
SUCCESS_REWARD_FLEXIBLE = 10.0
REBALANCED_SUCCESS_REWARD_HARD = 10.0
REBALANCED_SUCCESS_REWARD_SOFT = 7.0
REBALANCED_SUCCESS_REWARD_FLEXIBLE = 5.0

# ================= 成本稳定性奖励配置 =================
ENABLE_COST_SPIKE_PENALTY = False

# 根据第一阶段 lightweight 指标校准：周期 p50 的平均值约为 0.0137。
COST_PER_CPU_TIME_THRESHOLD = 0.0137
COST_RATIO_SPIKE_THRESHOLD = 0.85
COST_SPIKE_PENALTY_SCALE = 8.0
COST_SPIKE_PENALTY_CLIP = 10.0

# ================= Constrained RL / Lagrangian reward =================
ENABLE_CONSTRAINED_RL = True
LAGRANGE_LR = 0.05
LAGRANGE_MAX = 10.0
CONSTRAINT_TARGET_SLA_VIOLATION = 0.02
CONSTRAINT_TARGET_DROP = 0.03
CONSTRAINT_TARGET_COST_OVER_BUDGET = 0.05
CONSTRAINT_TARGET_OVERLOAD = 0.05
CONSTRAINT_COST_BUDGET_RATIO = 1.3
CONSTRAINT_OVERLOAD_THRESHOLD = 0.85
INITIAL_LAMBDA_SLA = 1.0
INITIAL_LAMBDA_DROP = 1.0
INITIAL_LAMBDA_COST = 0.5
INITIAL_LAMBDA_OVERLOAD = 0.5
