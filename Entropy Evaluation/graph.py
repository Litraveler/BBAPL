import os
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 配置参数 =====================
INPUT_FILE = "E7_results_entropy/entropy.csv"

# 颜色配置
COLORS = ['#FFCCCC', '#CCFFCC', '#CCCCFF']

# 字体配置
AXIS_LABEL_FONT_SIZE = 28  # 轴名称字体大小
AXIS_TICK_FONT_SIZE = 24   # 轴刻度字体大小
LEGEND_FONT_SIZE = 24

# 纵坐标范围
Y_MAX = 10

# 熵指标列名
ENTROPY_COLS = ['beta_entropy_3', 'beta_entropy_6', 'beta_entropy_10']

# 横坐标标签（lambda with tilde）
X_LABELS = ['$\\tilde{\\lambda}_3$', '$\\tilde{\\lambda}_6$', '$\\tilde{\\lambda}_{10}$']

# 图例标签
LEGEND_LABELS = ['Four Segments', 'Three Segments', 'Two Segments']

# ===================== 数据处理函数 =====================
def load_and_process_data(file_path):
    """加载熵数据并按pattern前缀分组计算平均值"""
    if not os.path.exists(file_path):
        raise ValueError(f"文件不存在: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"读取文件: {file_path}, 数据形状: {df.shape}")
    
    # 按posture分组
    sit_df = df[df['posture'] == 'sit'].copy()
    walk_df = df[df['posture'] == 'walk'].copy()
    
    # 计算整体均值（不区分线段数量）
    sit_overall_mean = sit_df[ENTROPY_COLS].mean()
    walk_overall_mean = walk_df[ENTROPY_COLS].mean()
    
    # 定义pattern前缀分组函数
    def get_pattern_prefix(pattern):
        if pattern.startswith('four_'):
            return 'four'
        elif pattern.startswith('three_'):
            return 'three'
        elif pattern.startswith('two_'):
            return 'two'
        return 'other'
    
    # 为sit数据添加前缀标签并计算平均值
    sit_df['prefix'] = sit_df['pattern'].apply(get_pattern_prefix)
    sit_avg = sit_df[sit_df['prefix'].isin(['four', 'three', 'two'])].groupby('prefix')[ENTROPY_COLS].mean()
    
    # 为walk数据添加前缀标签并计算平均值
    walk_df['prefix'] = walk_df['pattern'].apply(get_pattern_prefix)
    walk_avg = walk_df[walk_df['prefix'].isin(['four', 'three', 'two'])].groupby('prefix')[ENTROPY_COLS].mean()
    
    # 确保顺序一致: four, three, two
    sit_avg = sit_avg.loc[['four', 'three', 'two']]
    walk_avg = walk_avg.loc[['four', 'three', 'two']]
    
    return sit_avg, walk_avg, sit_overall_mean, walk_overall_mean

# ===================== 绘图函数 =====================
def plot_entropy_bar(data, colors, output_path):
    """绘制熵值柱状图（单张图）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    n_groups = len(ENTROPY_COLS)
    n_bars = len(data)
    
    # 设置柱子宽度和间距
    bar_width = 0.25
    index = range(n_groups)
    
    # 绘制每个分组的柱子
    for i, (prefix, row) in enumerate(data.iterrows()):
        ax.bar([x + i * bar_width for x in index], row.values, width=bar_width, 
               color=colors[i], edgecolor='black', label=LEGEND_LABELS[i])
    
    # 设置横坐标
    ax.set_xticks([x + bar_width for x in index])
    ax.set_xticklabels(X_LABELS, fontsize=AXIS_TICK_FONT_SIZE)
    
    # 设置纵坐标范围
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(range(0, Y_MAX + 1, 2))
    ax.set_yticklabels([str(i) for i in range(0, Y_MAX + 1, 2)], fontsize=AXIS_TICK_FONT_SIZE)
    
    # 设置纵坐标标签
    ax.set_ylabel('Entropy', fontsize=AXIS_LABEL_FONT_SIZE)
    
    # 设置横坐标标签
    ax.set_xlabel('Lambda Entropy', fontsize=AXIS_LABEL_FONT_SIZE)
    
    # 设置图例
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    
    # 设置网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")
    
    # 关闭当前图
    plt.close()

# ===================== 主函数 =====================
def main():
    # 加载并处理数据
    sit_avg, walk_avg, sit_overall_mean, walk_overall_mean = load_and_process_data(INPUT_FILE)
    
    # 输出整体均值（不区分线段数量）
    print("\n" + "=" * 60)
    print("整体熵值均值（不区分线段数量）")
    print("=" * 60)
    
    print("\n【sit 姿态】")
    print(f"  beta_entropy_3: {sit_overall_mean['beta_entropy_3']:.4f}")
    print(f"  beta_entropy_6: {sit_overall_mean['beta_entropy_6']:.4f}")
    print(f"  beta_entropy_10: {sit_overall_mean['beta_entropy_10']:.4f}")
    
    print("\n【walk 姿态】")
    print(f"  beta_entropy_3: {walk_overall_mean['beta_entropy_3']:.4f}")
    print(f"  beta_entropy_6: {walk_overall_mean['beta_entropy_6']:.4f}")
    print(f"  beta_entropy_10: {walk_overall_mean['beta_entropy_6']:.4f}")
    
    print("\n" + "=" * 60)
    
    # 输出绘制图片时的数据（按pattern前缀分组的均值）
    print("\n" + "=" * 60)
    print("绘图数据（按pattern前缀分组）")
    print("=" * 60)
    
    print("\n【sit 姿态 - 绘图数据】")
    for prefix in ['four', 'three', 'two']:
        print(f"  {prefix}:")
        print(f"    beta_entropy_3: {sit_avg.loc[prefix, 'beta_entropy_3']:.4f}")
        print(f"    beta_entropy_6: {sit_avg.loc[prefix, 'beta_entropy_6']:.4f}")
        print(f"    beta_entropy_10: {sit_avg.loc[prefix, 'beta_entropy_10']:.4f}")
    
    print("\n【walk 姿态 - 绘图数据】")
    for prefix in ['four', 'three', 'two']:
        print(f"  {prefix}:")
        print(f"    beta_entropy_3: {walk_avg.loc[prefix, 'beta_entropy_3']:.4f}")
        print(f"    beta_entropy_6: {walk_avg.loc[prefix, 'beta_entropy_6']:.4f}")
        print(f"    beta_entropy_10: {walk_avg.loc[prefix, 'beta_entropy_10']:.4f}")
    
    print("\n" + "=" * 60)
    
    # 输出目录
    output_dir = 'E7_results_entropy'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制sit数据的图（单独保存）
    sit_output_path = os.path.join(output_dir, 'entropy_bar_plot_sit.png')
    plot_entropy_bar(sit_avg, COLORS, sit_output_path)
    
    # 绘制walk数据的图（单独保存）
    walk_output_path = os.path.join(output_dir, 'entropy_bar_plot_walk.png')
    plot_entropy_bar(walk_avg, COLORS, walk_output_path)
    
    print("\n绘图完成！")

if __name__ == "__main__":
    main()

# 【sit 姿态 - 绘图数据】
#   four:
#     beta_entropy_3: 4.3528
#     beta_entropy_6: 4.6534
#     beta_entropy_10: 5.0151
#   three:
#     beta_entropy_3: 3.3180
#     beta_entropy_6: 3.7719
#     beta_entropy_10: 4.3189
#   two:
#     beta_entropy_3: 2.7120
#     beta_entropy_6: 3.3788
#     beta_entropy_10: 4.0469

# 【walk 姿态 - 绘图数据】
#   four:
#     beta_entropy_3: 4.1049
#     beta_entropy_6: 4.4030
#     beta_entropy_10: 4.7885
#   three:
#     beta_entropy_3: 3.3611
#     beta_entropy_6: 3.7249
#     beta_entropy_10: 4.2097
#   two:
#     beta_entropy_3: 2.7635
#     beta_entropy_6: 3.3710
#     beta_entropy_10: 3.9942