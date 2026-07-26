import numpy as np
import matplotlib.pyplot as plt
import math
# ==================== 新增：环境配置 ====================
# 1. 解决中文显示问题（以黑体为例，Windows系统通用）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ==================== 从 task_manager.py 复现的核心函数 ====================
def gaussian(x, mu, sigma):
    """高斯分布密度（未归一化，用于波形叠加）"""
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def sigmoid(x, k=3.3):
    """S型平滑函数，防止溢出"""
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def get_tidal_multiplier(task_type, sim_hr):
    """完全复刻 TaskManager._get_tidal_multiplier 的逻辑"""
    if task_type == "Realtime_Service":
        return 0.3 + 0.8 * gaussian(sim_hr, 9.0, 1.5) + 0.9 * gaussian(sim_hr, 20.0, 2.0)
    elif task_type == "Interactive_Query":
        rise = sigmoid(sim_hr - 9.0, k=3.3)
        fall = sigmoid(sim_hr - 18.0, k=3.3)
        return 0.2 + 1.0 * (rise - fall)
    elif task_type == "Data_Intensive":
        return 0.1 + 1.5 * gaussian(sim_hr, 2.0, 2.0)
    elif task_type == "Model_Training":
        return 0.4 + 0.4 * gaussian(sim_hr, 4.0, 3.0)
    else:
        return 1.0

# ==================== 区域权重缩放（取自 generate_tasks） ====================
def get_region_adjusted_weight(task_type, region, tidal_weight):
    """
    东部不缩放，中部、西部对实时任务分别缩放 0.5 和 0.3。
    其他任务类型不缩放。
    """
    if task_type == "Realtime_Service":
        if region == 'West':
            return tidal_weight * 0.3
        elif region == 'Middle':
            return tidal_weight * 0.5
    # 东部及其他类型不做缩放
    return tidal_weight

# ==================== 图1：一天内的潮汐波动曲线 ====================
hours = np.linspace(0, 24, 500)
task_types = ["Realtime_Service", "Interactive_Query", "Data_Intensive", "Model_Training"]
task_labels = ["实时响应服务", "交互式查询", "数据密集型", "模型训练型"]
colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

plt.figure(figsize=(12, 5))
for t, lbl, col in zip(task_types, task_labels, colors):
    y = [get_tidal_multiplier(t, h) for h in hours]
    plt.plot(hours, y, label=lbl, color=col, linewidth=2)

# --- 关键修改点 ---
plt.xlim(0, 24)  # 强制横轴范围为 0 到 24，消除两端空档
plt.xticks(np.arange(0, 25, 2)) # 设置刻度，让坐标轴更整齐
# ----------------

plt.xlabel("时间", fontsize=12)
plt.ylabel("潮汐波动系数", fontsize=12)
plt.title("四种任务类型潮汐波动", fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 图2：某一时刻三个区域的任务类型生成权重（柱状图） ====================
# 选择你感兴趣的时间点（例如上午 9:00）
selected_hour = 9.0

regions = ['East', 'Middle', 'West']
region_labels = ['东部 (East)', '中部 (Middle)', '西部 (West)']
# 各区域基础生成比例（与 _get_dynamic_task_rate 一致）
region_base_ratio = {'East': 0.50, 'Middle': 0.30, 'West': 0.20}

# 计算每个区域每种任务类型的“调整后权重”
data_matrix = []  # 行：区域，列：任务类型
for reg in regions:
    weights = []
    for t in task_types:
        raw_w = get_tidal_multiplier(t, selected_hour)
        adj_w = get_region_adjusted_weight(t, reg, raw_w)
        weights.append(adj_w)
    data_matrix.append(weights)

# 绘制分组柱状图
x = np.arange(len(task_types))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
for i, (reg, reg_lbl) in enumerate(zip(regions, region_labels)):
    bars = ax.bar(x + i * width, data_matrix[i], width,
                   label=f'{reg_lbl} (比例 {region_base_ratio[reg]:.2f})',
                   color=plt.cm.tab10(i))
    # 标注数值
    for bar, val in zip(bars, data_matrix[i]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x + width)
ax.set_xticklabels(task_labels, fontsize=11)
ax.set_ylabel('生成权重 (未归一化)', fontsize=12)
ax.set_title(f'不同区域任务类型权重对比 (sim_hr = {selected_hour:.1f})', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()