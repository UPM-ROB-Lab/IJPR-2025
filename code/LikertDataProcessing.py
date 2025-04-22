import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Cambria 并加粗
plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['font.weight'] = 'bold'  # 全局字体加粗

# 定义文件名
file_names = [
    "1_42.xlsx", "2_45.xlsx", "3_43.xlsx", "4_41.xlsx", "5_45.xlsx",
    "6_41.xlsx", "7_43.xlsx", "8_43.xlsx", "9_45.xlsx",
    "itera_1_42.xlsx", "itera_2_45.xlsx", "itera_3_41.xlsx"
]

# 自定义小标题名称
scheme_names = [
    "Scheme 1", "Scheme 2", "Scheme 3", "Scheme 4", "Scheme 5",
    "Scheme 6", "Scheme 7", "Scheme 8", "Scheme 9",
    "Iteration 1", "Iteration 2", "Iteration 3"
]

# 存储每个表格的平均值
averages = []

# 读取每个表格并计算平均值
for file_name in file_names:
    df = pd.read_excel(file_name)
    column_averages = df.mean().tolist()  # 计算每列的平均值
    averages.append(column_averages)

# 将平均值数据整理为一个汇总表格
questions = [
    "Gearshift Lever",
    "Steering Wheel",
    "Control Buttons",
    "Seat",
    "Center Console",
    "Other Visible Areas",
    "Overall Interior Design"
]
df_averages = pd.DataFrame(averages, columns=questions, index=scheme_names)  # 使用自定义名称作为索引

# 计算每个方案的总得分
df_averages['Total Score'] = df_averages.sum(axis=1)

# 按照总得分从高到低排序
df_averages_sorted = df_averages.sort_values(by='Total Score', ascending=False)

# 绘制柱状图
fig, ax = plt.subplots(figsize=(14, 7))

# 设置柱状图的宽度
bar_width = 0.8

# 定义每个问题的颜色（新的颜色）
colors = [
    "#452a3d",  # RGB (69, 42, 61)
    "#44757a",  # RGB (68, 117, 122)
    "#b7b5a0",  # RGB (183, 181, 160)
    "#eed5b7",  # RGB (238, 213, 183)
    "#e5855d",  # RGB (229, 133, 93)
    "#dd6c4c",  # RGB (221, 108, 76)
    "#d44c3c",  # RGB (212, 76, 60)
]

# 每个表格对应一个柱子，柱子由七列数据叠加而成
bottom = np.zeros(len(df_averages_sorted))  # 用于叠加的初始值
for i, question in enumerate(questions):
    ax.bar(df_averages_sorted.index, df_averages_sorted[question], bar_width, label=question, bottom=bottom, color=colors[i])
    bottom += df_averages_sorted[question]  # 更新叠加值

# 在每个柱子上标注总得分
for i, total_score in enumerate(df_averages_sorted['Total Score']):
    ax.text(i, total_score + 0.1, f"{total_score:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# 在每个柱子下方标注有效问卷数/收集问卷数
valid_surveys = [42, 45, 43, 41, 45, 41, 43, 43, 45, 42, 45, 41]  # 有效问卷数
total_surveys = 50  # 收集问卷数
# 按照排序后的方案名称重新排列有效问卷数
valid_surveys_sorted = [valid_surveys[scheme_names.index(name)] for name in df_averages_sorted.index]
for i, valid in enumerate(valid_surveys_sorted):
    ax.text(i, -2, f"{valid}/{total_surveys}", ha='center', va='top', fontsize=10, fontweight='bold', color='black')

# 在最后一个柱子的问卷数后面标注含义
ax.text(len(valid_surveys_sorted) - 0.5, -2, "Valid questionnaires/Total questionnaires", ha='left', va='top', fontsize=10, fontweight='bold', color='black')

# 设置图表标题和标签
ax.set_title("Questionnaire Results", fontsize=16, fontweight='bold')
ax.set_xlabel("Schemes", fontsize=20, fontweight='bold', labelpad=25)
ax.set_ylabel("Mean Score", fontsize=20, fontweight='bold')
ax.legend(title="Functional Region", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'weight': 'bold'})  # 图例字体加粗

# 调整布局
plt.xticks(rotation=0, ha='center')  # 旋转x轴标签以便更好地显示
plt.tight_layout()

# 保存图表为 JPG 格式
plt.savefig("LikertResults.jpg", dpi=600, bbox_inches='tight')

# 显示图表
plt.show()