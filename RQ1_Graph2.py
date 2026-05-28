import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

def load_all_data():
    files = [
        'single_timeperiod_eer_results_FIX_1.csv',
        'single_timeperiod_eer_results_FIX_2.csv',
        'single_timeperiod_eer_results_FIX_3.csv',
        'single_timeperiod_eer_results_FIX_4.csv'
    ]
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def categorize_eer(eer):
    if eer == 0:
        return '0'
    elif 0 < eer <= 0.1:
        return '0-10'
    elif 0.1 < eer <= 0.2:
        return '10-20'
    elif 0.2 < eer <= 0.3:
        return '20-30'
    elif 0.3 < eer <= 0.4:
        return '30-40'
    elif 0.4 < eer <= 0.5:
        return '40-50'
    else:
        return '>50'

def calculate_distribution(df, prefix):
    filtered = df[df['Pattern'].str.startswith(prefix)]
    filtered['EER_Category'] = filtered['EER'].apply(categorize_eer)
    counts = filtered['EER_Category'].value_counts().sort_index()
    return counts

def get_category_radius(category):
    radius_map = {
        '0': 0.5,
        '0-10': 0.55,
        '10-20': 0.6,
        '20-30': 0.65,
        '30-40': 0.7,
        '40-50': 0.75,
        '>50': 0.8
    }
    return radius_map.get(category, 1.0)

def plot_single_pie(dist, filename):
    colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#8E44AD', "#00C7B1", '#98FB98', '#DDA0DD']
    labels = ['0', '0-10', '10-20', '20-30', '30-40', '40-50', '>50']
    label_to_color = {label: colors[i] for i, label in enumerate(labels)}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    total = dist.sum()
    current_angle = -90
    
    max_radius = max([get_category_radius(c) for c in dist.index])
    wedges_info = []
    
    for category, count in dist.items():
        angle = (count / total) * 360
        radius = get_category_radius(category)
        color = label_to_color.get(category, '#888888')
        
        wedge = Wedge((0, 0), radius, current_angle, current_angle + angle, 
                     facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)
        
        percentage = (count / total) * 100
        mid_angle = current_angle + angle / 2
        
        wedges_info.append((radius, mid_angle, percentage))
        current_angle += angle
    
    for radius, mid_angle, percentage in wedges_info:
        mid_angle_rad = np.deg2rad(mid_angle)
        label_radius = radius * 0.75
        label_x = label_radius * np.cos(mid_angle_rad)
        label_y = label_radius * np.sin(mid_angle_rad)
        
        
        ax.text(label_x, label_y, f'{percentage:.1f}%', 
                ha='center', va='center', fontsize=28)
    
    ax.set_xlim(-max_radius * 0.8, max_radius * 0.8)
    ax.set_ylim(-max_radius * 0.8, max_radius * 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(labels))]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.1, 0.95), ncol=1, prop={'size': 24})
    
    plt.tight_layout()
    plt.savefig(f'RQ1-graphs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pie_charts(four_dist, three_dist, two_dist):
    plot_single_pie(four_dist, 'pie_four.png')
    plot_single_pie(three_dist, 'pie_three.png')
    plot_single_pie(two_dist, 'pie_two.png')

def output_data_results(four_dist, three_dist, two_dist):
    """输出详细的数据结果"""
    categories = ['0', '0-10', '10-20', '20-30', '30-40', '40-50', '>50']
    
    print("\n" + "=" * 70)
    print("EER Distribution Data Results")
    print("=" * 70)
    
    def create_full_distribution(dist):
        full_data = {}
        total = dist.sum()
        for cat in categories:
            count = dist.get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            full_data[cat] = {'count': count, 'percentage': pct}
        return full_data
    
    four_data = create_full_distribution(four_dist)
    three_data = create_full_distribution(three_dist)
    two_data = create_full_distribution(two_dist)
    
    print("\n[Four-segment Patterns]")
    print(f"{'Category':<12} {'Count':>8} {'Percentage':>12}")
    print("-" * 35)
    for cat in categories:
        info = four_data[cat]
        print(f"{cat:<12} {info['count']:>8} {info['percentage']:>11.2f}%")
    print(f"{'Total':<12} {sum(d['count'] for d in four_data.values()):>8}")
    
    print("\n[Three-segment Patterns]")
    print(f"{'Category':<12} {'Count':>8} {'Percentage':>12}")
    print("-" * 35)
    for cat in categories:
        info = three_data[cat]
        print(f"{cat:<12} {info['count']:>8} {info['percentage']:>11.2f}%")
    print(f"{'Total':<12} {sum(d['count'] for d in three_data.values()):>8}")
    
    print("\n[Two-segment Patterns]")
    print(f"{'Category':<12} {'Count':>8} {'Percentage':>12}")
    print("-" * 35)
    for cat in categories:
        info = two_data[cat]
        print(f"{cat:<12} {info['count']:>8} {info['percentage']:>11.2f}%")
    print(f"{'Total':<12} {sum(d['count'] for d in two_data.values()):>8}")
    
    print("\n" + "=" * 70)
    print("LaTeX Table Format (Copy to Paper)")
    print("=" * 70)
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{EER Distribution Across Different Pattern Types}
\\label{tab:eer_distribution}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Pattern Type} & \\textbf{0} & \\textbf{0-10} & \\textbf{10-20} & \\textbf{20-30} & \\textbf{30-40} & \\textbf{>50} \\\\
\\midrule"""
    
    def get_row(data, categories):
        row = []
        for cat in categories[:-1]:
            row.append(f"{data[cat]['percentage']:.1f}\\%")
        row.append(f"{data['>50']['percentage']:.1f}\\%")
        return " & ".join(row)
    
    latex_table += f"""
Four-segment & {get_row(four_data, categories)} \\\\
Three-segment & {get_row(three_data, categories)} \\\\
Two-segment & {get_row(two_data, categories)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    print(latex_table)
    
    csv_data = []
    for pattern, data in [('Four-segment', four_data), ('Three-segment', three_data), ('Two-segment', two_data)]:
        for cat in categories:
            csv_data.append({'Pattern': pattern, 'EER_Range': cat, 'Count': data[cat]['count'], 'Percentage': f"{data[cat]['percentage']:.2f}%"})
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv('RQ1-graphs/eer_distribution_data.csv', index=False)
    print(f"\nData saved to: RQ1-graphs/eer_distribution_data.csv")
    print("=" * 70)

def calculate_user_eer_stats(df):
    """以用户为单位，计算每个用户的EER均值和方差"""
    print("\n" + "=" * 70)
    print("User-level EER Statistics (Aggregated from All Sessions)")
    print("=" * 70)
    
    # 按用户ID分组，计算均值、方差等统计量
    user_stats = df.groupby('用户ID')['EER'].agg([
        ('Count', 'count'),
        ('Mean_EER', 'mean'),
        ('Std_EER', 'std'),
        ('Var_EER', 'var'),
        ('Min_EER', 'min'),
        ('Max_EER', 'max')
    ]).reset_index()
    
    # 按用户ID排序
    user_stats = user_stats.sort_values('用户ID').reset_index(drop=True)
    
    # 打印详细统计
    print(f"\n{'User ID':<10} {'Count':>8} {'Mean':>10} {'Std':>10} {'Var':>12} {'Min':>10} {'Max':>10}")
    print("-" * 75)
    
    for _, row in user_stats.iterrows():
        print(f"{row['用户ID']:<10} {int(row['Count']):>8} {row['Mean_EER']:>9.4f} {row['Std_EER']:>9.4f} {row['Var_EER']:>11.6f} {row['Min_EER']:>9.4f} {row['Max_EER']:>9.4f}")
    
    # 打印整体汇总
    print("\n" + "-" * 75)
    print(f"{'Total':<10} {len(user_stats):>8} {user_stats['Mean_EER'].mean():>9.4f} {user_stats['Std_EER'].mean():>9.4f} {user_stats['Var_EER'].mean():>11.6f} {user_stats['Min_EER'].min():>9.4f} {user_stats['Max_EER'].max():>9.4f}")
    
    # 保存到CSV
    user_stats.to_csv('RQ1-graphs/user_eer_stats.csv', index=False)
    print(f"\nUser-level statistics saved to: RQ1-graphs/user_eer_stats.csv")
    
    # 输出LaTeX表格格式
    print("\n" + "=" * 70)
    print("LaTeX Table Format for User Statistics")
    print("=" * 70)
    
    latex_user_table = """\\begin{table}[htbp]
\\centering
\\caption{User-level EER Statistics (Aggregated from All Sessions)}
\\label{tab:user_eer_stats}
\\begin{tabular}{lrrrrrr}
\\toprule
\\textbf{User ID} & \\textbf{Count} & \\textbf{Mean EER} & \\textbf{Std} & \\textbf{Variance} & \\textbf{Min} & \\textbf{Max} \\\\
\\midrule"""
    
    for _, row in user_stats.iterrows():
        latex_user_table += f"\n{row['用户ID']} & {int(row['Count'])} & {row['Mean_EER']:.4f} & {row['Std_EER']:.4f} & {row['Var_EER']:.6f} & {row['Min_EER']:.4f} & {row['Max_EER']:.4f} \\\\"
    
    latex_user_table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
    
    print(latex_user_table)
    print("\n" + "=" * 70)
    
    return user_stats

if __name__ == '__main__':
    df = load_all_data()
    
    # 计算用户级EER统计
    calculate_user_eer_stats(df)
    
    four_dist = calculate_distribution(df, 'four')
    three_dist = calculate_distribution(df, 'three')
    two_dist = calculate_distribution(df, 'two')
    
    output_data_results(four_dist, three_dist, two_dist)
    plot_pie_charts(four_dist, three_dist, two_dist)


# ======================================================================
# User-level EER Statistics (Aggregated from All Sessions)
# ======================================================================

# User ID       Count       Mean        Std          Var        Min        Max
# ---------------------------------------------------------------------------
# 11bb98af-9492-43c4-b7d0-1317dea626ca      190    0.0345    0.0701    0.004921    0.0000    0.3333
# 1e992ffa-436b-4b34-b34a-777d244b13dd      168    0.0251    0.0703    0.004941    0.0000    0.5000
# a89817e4-a5fb-458e-8645-4658e24bde57      192    0.0554    0.1023    0.010472    0.0000    0.6923
# eb251996-b14d-440d-ab78-ba2ae89ffce4      183    0.0417    0.0872    0.007602    0.0000    0.5000
# ee62a8c9-ad8d-4582-b333-36aa425e9fde      192    0.0471    0.0877    0.007696    0.0000    0.4583
# fe958dc9-6bc7-4e10-bfde-46f03d964824      190    0.0522    0.1098    0.012058    0.0000    0.7000

# ---------------------------------------------------------------------------
# Total             6    0.0427    0.0879    0.007948    0.0000    0.7000

# User-level statistics saved to: RQ1-graphs/user_eer_stats.csv

# ======================================================================
# LaTeX Table Format for User Statistics
# ======================================================================
# \begin{table}[htbp]
# \centering
# \caption{User-level EER Statistics (Aggregated from All Sessions)}
# \label{tab:user_eer_stats}
# \begin{tabular}{lrrrrrr}
# \toprule
# \textbf{User ID} & \textbf{Count} & \textbf{Mean EER} & \textbf{Std} & \textbf{Variance} & \textbf{Min} & \textbf{Max} \\
# \midrule
# 11bb98af-9492-43c4-b7d0-1317dea626ca & 190 & 0.0345 & 0.0701 & 0.004921 & 0.0000 & 0.3333 \\
# 1e992ffa-436b-4b34-b34a-777d244b13dd & 168 & 0.0251 & 0.0703 & 0.004941 & 0.0000 & 0.5000 \\
# a89817e4-a5fb-458e-8645-4658e24bde57 & 192 & 0.0554 & 0.1023 & 0.010472 & 0.0000 & 0.6923 \\
# eb251996-b14d-440d-ab78-ba2ae89ffce4 & 183 & 0.0417 & 0.0872 & 0.007602 & 0.0000 & 0.5000 \\
# ee62a8c9-ad8d-4582-b333-36aa425e9fde & 192 & 0.0471 & 0.0877 & 0.007696 & 0.0000 & 0.4583 \\
# fe958dc9-6bc7-4e10-bfde-46f03d964824 & 190 & 0.0522 & 0.1098 & 0.012058 & 0.0000 & 0.7000 \\
# \bottomrule
# \end{tabular}
# \end{table}

# ======================================================================
# RQ1_Graph2.py:37: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   filtered['EER_Category'] = filtered['EER'].apply(categorize_eer)
# RQ1_Graph2.py:37: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   filtered['EER_Category'] = filtered['EER'].apply(categorize_eer)
# RQ1_Graph2.py:37: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   filtered['EER_Category'] = filtered['EER'].apply(categorize_eer)

# ======================================================================
# EER Distribution Data Results
# ======================================================================

# [Four-segment Patterns]
# Category        Count   Percentage
# -----------------------------------
# 0                 265       72.60%
# 0-10                4        1.10%
# 10-20              66       18.08%
# 20-30              18        4.93%
# 30-40               8        2.19%
# 40-50               3        0.82%
# >50                 1        0.27%
# Total             365

# [Three-segment Patterns]
# Category        Count   Percentage
# -----------------------------------
# 0                 307       81.65%
# 0-10               16        4.26%
# 10-20              47       12.50%
# 20-30               4        1.06%
# 30-40               1        0.27%
# 40-50               1        0.27%
# >50                 0        0.00%
# Total             376

# [Two-segment Patterns]
# Category        Count   Percentage
# -----------------------------------
# 0                 284       75.94%
# 0-10               20        5.35%
# 10-20              54       14.44%
# 20-30              10        2.67%
# 30-40               0        0.00%
# 40-50               5        1.34%
# >50                 1        0.27%
# Total             374

# ======================================================================
# LaTeX Table Format (Copy to Paper)
# ======================================================================
# \begin{table}[htbp]
# \centering
# \caption{EER Distribution Across Different Pattern Types}
# \label{tab:eer_distribution}
# \begin{tabular}{lcccccc}
# \toprule
# \textbf{Pattern Type} & \textbf{0} & \textbf{0-10} & \textbf{10-20} & \textbf{20-30} & \textbf{30-40} & \textbf{>50} \\
# \midrule
# Four-segment & 72.6\% & 1.1\% & 18.1\% & 4.9\% & 2.2\% & 0.8\% & 0.3\% \\
# Three-segment & 81.6\% & 4.3\% & 12.5\% & 1.1\% & 0.3\% & 0.3\% & 0.0\% \\
# Two-segment & 75.9\% & 5.3\% & 14.4\% & 2.7\% & 0.0\% & 1.3\% & 0.3\% \\
# \bottomrule
# \end{tabular}
# \end{table}

# Data saved to: RQ1-graphs/eer_distribution_data.csv
# ======================================================================