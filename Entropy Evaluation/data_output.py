import pandas as pd

# 读取CSV文件
df = pd.read_csv('E7_results_entropy/entropy.csv')

# 确保所有数值列都是浮点数类型
numeric_columns = [
    'beta_entropy_3',
    'beta_entropy_6',
    'beta_entropy_10',
    'alfa_guess_work_entropy_5',
    'alfa_guess_work_entropy_2',
    'upper_bound'
]

for col in numeric_columns:
    # 将列转换为字符串，然后转换为浮点数，处理可能的非数值数据
    df[col] = df[col].astype(str).str.strip().replace('', 'NaN').astype(float)

# 定义指标顺序
metrics = numeric_columns

# 定义所有PIN列表
all_pins = sorted(df['PIN'].unique())

# 将PIN分成两组（每组10个）
pin_groups = [all_pins[:10], all_pins[10:]]

# 生成LaTeX表格字符串
latex_tables = []

for group_idx, pin_group in enumerate(pin_groups):
    table_str = r"""
\begin{table*}
\centering
\caption{Entropy Metrics for PIN Group """ + str(group_idx + 1) + r"""}
\label{tab:entropy_group_""" + str(group_idx + 1) + r"""}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|""" + "c" * len(pin_group) + r"""}
\toprule
\textbf{Metric} & \multicolumn{""" + str(len(pin_group)) + r"""}{c}{\textbf{PIN}} \\
\cmidrule{2-""" + str(len(pin_group) + 1) + r"""}
""" + " & ".join([""] + [f"\\textbf{{{pin}}}" for pin in pin_group]) + r""" \\
\midrule
\multicolumn{""" + str(len(pin_group) + 1) + r"""}{c}{\textbf{Sit Posture}} \\
\midrule
"""

    # 添加sit姿势数据
    for metric in metrics:
        row = [metric.replace('_', r'\_')]
        for pin in pin_group:
            # 获取对应PIN和姿势的数据
            value = df[(df['PIN'] == pin) & (df['posture'] == 'sit')][metric].values[0]
            # 确保值是数值类型
            if pd.isna(value):
                row.append("N/A")
            else:
                row.append(f"{value:.2f}")
        table_str += " & ".join(row) + r" \\" + "\n"

    table_str += r"""\midrule
\multicolumn{""" + str(len(pin_group) + 1) + r"""}{c}{\textbf{Walk Posture}} \\
\midrule
"""

    # 添加walk姿势数据
    for metric in metrics:
        row = [metric.replace('_', r'\_')]
        for pin in pin_group:
            value = df[(df['PIN'] == pin) & (df['posture'] == 'walk')][metric].values[0]
            if pd.isna(value):
                row.append("N/A")
            else:
                row.append(f"{value:.2f}")
        table_str += " & ".join(row) + r" \\" + "\n"

    table_str += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""
    latex_tables.append(table_str)

# 打印所有表格字符串
for i, table in enumerate(latex_tables):
    print(f"// ========== Table Group {i + 1} ==========")
    print(table)
    print("\n\n")

# 保存到文件
with open('E7_results_entropy/latex_tables.tex', 'w') as f:
    for table in latex_tables:
        f.write(table + "\n\n")