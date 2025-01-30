#%% figure 1
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
# Data
states = ['State1', 'State2', 'State3', 'State4', 'State5']
# ai_performance = [5.68, 65.77, 18.04, 6.96, 1.40]
ai_performance = [8.72, 59.52, 18.40, 10.44, 2.92]

human_performance = [15.80, 4.00, 21.70, 12.00, 46.50]

# Bar positions
x = np.arange(len(states))
width = 0.4

# Create figure and axis
fig, ax = plt.subplots(figsize=(8,5))

# Bars with black edges
ai_bars = ax.bar(x - width/2, ai_performance, width, label='LLMs Average Performance', color='#4F81BD', edgecolor='black')
human_bars = ax.bar(x + width/2, human_performance, width, label='Human Performance', color='#A6C6F2', edgecolor='black')

# Annotations
for bar in ai_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height}', ha='center', va='bottom', fontsize=10)

for bar in human_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height}', ha='center', va='bottom', fontsize=10)

# Add percentage sign to Y-axis
def add_percent(y, _):
    return f'{y:.0f}%'

ax.yaxis.set_major_formatter(FuncFormatter(add_percent))

# Adjust Y-axis range to include 70%
ax.set_ylim(0, 70)

# Adjust X-axis ticks and labels
ax.set_xticks(x)  # Set tick positions
ax.set_xticklabels(states)  # Set custom tick labels

# Adjust labels
# ax.set_xlabel('State', fontsize=13, labelpad=10)  # X-axis label
ax.set_ylabel('Percentage', fontsize=13, labelpad=20, rotation=0, ha='center')  # Y-axis label

# Manually adjust Y-axis label position
ax.yaxis.set_label_coords(-0.05, 1.03)  # Adjust Y-axis label position (horizontal, vertical)

# Manually adjust X-axis label position
ax.xaxis.set_label_coords(0.5, -0.1)  # Adjust X-axis label position (horizontal, vertical)

# Remove all extra spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Add an arrow to the positive direction of Y-axis
ax.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction',
            textcoords='axes fraction', arrowprops=dict(facecolor='black', width=0.5, headwidth=6, headlength=8))

# Add an arrow to the positive direction of X-axis
ax.annotate('', xy=(0.975, 0), xytext=(0, 0), xycoords='axes fraction',
            textcoords='axes fraction', arrowprops=dict(facecolor='black', width=0.5, headwidth=6, headlength=8))

# Set legend without a frame
ax.legend(fontsize=12, frameon=False, loc='upper right')  # No border for legend

# Enlarge tick label sizes for both axes
ax.tick_params(axis='x', labelsize=12)  # Increase X-axis tick label size
ax.tick_params(axis='y', labelsize=12)  # Increase Y-axis tick label size

# Save and show plot
plt.tight_layout()
plt.savefig(r'F:\zjy\柱状图\performance_comparison_1.png',dpi=800)
plt.show()




#%% figure 2
#%% figure 2
import numpy as np
import matplotlib.pyplot as plt

# 数据
states = ['State1', 'State2', 'State3', 'State4', 'State5']
labels = ['Gemini1.5Flash', 'GPT 4o mini', 'GPT 4o', 'Kimi', 'Ernie Bot']
data = np.array([
    [10.60, 69.40, 11.60, 8.40, 0.00],
    [9.00, 40.60, 25.20, 14.60, 10.60],
    [8.60, 62.60, 20.60, 8.20, 0.00],
    [3.60, 73.60, 8.20, 14.60, 0.00],
    [11.80, 51.40, 26.40, 6.40, 4.00]
])

# 针对每个模型单独计算误差（行方向）
errors = np.std(data, axis=1, ddof=1) / np.sqrt(data.shape[1])  # 每行的标准误差

# 柱状图参数
x = np.arange(len(states))
width = 0.15

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 5))

# 颜色
colors = ['#1b4f72', '#2874a6', '#3498db', '#5dade2', '#aed6f1']

# 绘制柱状图
for i in range(len(labels)):
    ax.bar(
        x + i * width - width * (len(labels) - 1) / 2, 
        data[i], 
        width, 
        label=labels[i], 
        color=colors[i], 
        edgecolor='black', 
        yerr=errors[i],  # 针对每个模型的误差
        capsize=4, 
        error_kw={'elinewidth': 1, 'color': 'black'}
    )

# 添加误差棒上的数值标注
for i in range(len(labels)):
    for j in range(len(states)):
        ax.text(
            x[j] + i * width - width * (len(labels) - 1) / 2, 
            data[i][j] + errors[i] + 1,  # 位置：柱子高度 + 对应误差 + 偏移量
            f'{errors[i]:.2f}',  # 标注每个柱子的误差
            ha='center', 
            va='bottom', 
            fontsize=8
        )

# 设置坐标轴和标签
ax.set_xticks(x)
ax.set_xticklabels(states, fontsize=14)
ax.set_yticks(np.arange(-20, 121, 20))
ax.set_yticklabels([f'{y}%' for y in np.arange(-20, 121, 20)], fontsize=14)
ax.set_ylim(-20, 120)

# 设置Y轴标签
ax.set_ylabel('Percentage', fontsize=14, labelpad=20, rotation=0, ha='center')
ax.yaxis.set_label_coords(-0.05, 1.03)

# 美化图形
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=13)

# 添加箭头
ax.annotate('', xy=(0.97, 0.14), xytext=(0, 0.14), xycoords='axes fraction',
            textcoords='axes fraction', arrowprops=dict(facecolor='black', width=0.5, headwidth=6, headlength=8))
ax.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction',
            textcoords='axes fraction', arrowprops=dict(facecolor='black', width=0.5, headwidth=6, headlength=8))

# 图例
ax.legend(fontsize=14, frameon=False, loc='upper right')

# 保存图形
plt.savefig(r'F:\zjy\柱状图\grouped_bar_chart_std', dpi=800)
plt.show()