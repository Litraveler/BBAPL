import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

def load_all_data():
    """加载所有四个CSV文件的数据"""
    files = [
        'single_timeperiod_eer_results_FIX_1.csv',
        'single_timeperiod_eer_results_FIX_2.csv',
        'single_timeperiod_eer_results_FIX_3.csv',
        'single_timeperiod_eer_results_FIX_4.csv'
    ]
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df['session'] = i + 1  # 添加会话标识
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def analyze_pattern_eer(df):
    """分析不同pattern（four、three、two）的EER差异性"""
    print("=" * 60)
    print("1. 不同Pattern的EER统计分析")
    print("=" * 60)
    
    # 提取不同pattern的数据
    four_eer = df[df['Pattern'].str.startswith('four')]['EER'].values
    three_eer = df[df['Pattern'].str.startswith('three')]['EER'].values
    two_eer = df[df['Pattern'].str.startswith('two')]['EER'].values
    
    print(f"four 样本数: {len(four_eer)}, 均值: {np.mean(four_eer):.4f}, 标准差: {np.std(four_eer):.4f}")
    print(f"three 样本数: {len(three_eer)}, 均值: {np.mean(three_eer):.4f}, 标准差: {np.std(three_eer):.4f}")
    print(f"two 样本数: {len(two_eer)}, 均值: {np.mean(two_eer):.4f}, 标准差: {np.std(two_eer):.4f}")
    print()
    
    # 正态性检验
    print("正态性检验 (Shapiro-Wilk):")
    stat, p_four = stats.shapiro(four_eer)
    print(f"  four: W={stat:.4f}, p={p_four:.4f} {'(正态分布)' if p_four > 0.05 else '(非正态分布)'}")
    stat, p_three = stats.shapiro(three_eer)
    print(f"  three: W={stat:.4f}, p={p_three:.4f} {'(正态分布)' if p_three > 0.05 else '(非正态分布)'}")
    stat, p_two = stats.shapiro(two_eer)
    print(f"  two: W={stat:.4f}, p={p_two:.4f} {'(正态分布)' if p_two > 0.05 else '(非正态分布)'}")
    print()
    
    # 方差齐性检验
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(four_eer, three_eer, two_eer)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    # 统计检验：使用 Kruskal-Wallis H 检验（非参数 ANOVA）
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(four_eer, three_eer, two_eer)
    print(f"  H(2) = {h_stat:.4f}, p = {p_kw:.4f}")
    
    # 计算效应量 eta-squared
    n_total = len(four_eer) + len(three_eer) + len(two_eer)
    eta_squared = h_stat / (n_total - 1)
    print(f"  效应量 η² = {eta_squared:.4f}", end="")
    if eta_squared > 0.14:
        print(" (大效应)")
    elif eta_squared > 0.06:
        print(" (中等效应)")
    else:
        print(" (小效应)")
    print()
    
    # 事后检验：使用 Dunn's test 进行两两比较
    if p_kw < 0.05:
        print("事后两两比较 (Dunn's test with Bonferroni 校正):")
        # 准备数据
        data_list = [four_eer, three_eer, two_eer]
        groups = ['four', 'three', 'two']
        
        # Dunn's test
        dunn_result = sp.posthoc_dunn(data_list, p_adjust='bonferroni')
        # 设置行和列的标签
        dunn_result.columns = groups
        dunn_result.index = groups
        print(f"  {dunn_result}")
        
        # 提取具体 p 值
        p_four_vs_three = dunn_result.loc['four', 'three']
        p_four_vs_two = dunn_result.loc['four', 'two']
        p_three_vs_two = dunn_result.loc['three', 'two']
        
        print(f"    four vs three: p = {p_four_vs_three:.4f} {'*' if p_four_vs_three < 0.05 else ''}")
        print(f"    four vs two: p = {p_four_vs_two:.4f} {'*' if p_four_vs_two < 0.05 else ''}")
        print(f"    three vs two: p = {p_three_vs_two:.4f} {'*' if p_three_vs_two < 0.05 else ''}")
        
        # 判断哪类 pattern EER 最低
        means = {'four': np.mean(four_eer), 'three': np.mean(three_eer), 'two': np.mean(two_eer)}
        min_pattern = min(means, key=means.get)
        print(f"\n  结论：不同 Pattern 的 EER 存在显著差异")
        print(f"  EER 最低的 Pattern: {min_pattern} (均值={means[min_pattern]:.4f})")
    else:
        print("  结论：不同 Pattern 的 EER 无显著差异")
    print()

def analyze_session_eer(df):
    """分析不同会话（四个文件）的EER差异性"""
    print("=" * 60)
    print("2. 不同会话的EER统计分析")
    print("=" * 60)
    
    sessions = df['session'].unique()
    session_data = [df[df['session'] == s]['EER'].values for s in sessions]
    
    for i, data in enumerate(session_data, 1):
        print(f"Session {i} 样本数: {len(data)}, 均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    print()
    
    # 正态性检验
    print("正态性检验 (Shapiro-Wilk):")
    normality = []
    for i, data in enumerate(session_data, 1):
        stat, p = stats.shapiro(data)
        is_normal = p > 0.05
        normality.append(is_normal)
        print(f"  Session {i}: W={stat:.4f}, p={p:.4f} {'(正态分布)' if is_normal else '(非正态分布)'}")
    print()
    
    # 方差齐性检验
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(*session_data)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    # 统计检验：使用 Kruskal-Wallis H 检验（非参数 ANOVA）
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(*session_data)
    print(f"  H({len(session_data)-1}) = {h_stat:.4f}, p = {p_kw:.4f}")
    
    # 计算效应量 eta-squared
    n_total = sum(len(data) for data in session_data)
    eta_squared = h_stat / (n_total - 1)
    print(f"  效应量 η² = {eta_squared:.4f}", end="")
    if eta_squared > 0.14:
        print(" (大效应)")
    elif eta_squared > 0.06:
        print(" (中等效应)")
    else:
        print(" (小效应)")
    print()
    
    # 事后检验：使用 Dunn's test 进行两两比较
    if p_kw < 0.05:
        print("事后两两比较 (Dunn's test with Bonferroni 校正):")
        # Dunn's test
        dunn_result = sp.posthoc_dunn(session_data, p_adjust='bonferroni')
        print(f"  {dunn_result}")
        
        # 提取具体 p 值
        n_sessions = len(session_data)
        has_significant_diff = False
        for i in range(n_sessions):
            for j in range(i + 1, n_sessions):
                p_val = dunn_result.iloc[i, j]
                print(f"    Session {i + 1} vs Session {j + 1}: p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
                if p_val < 0.05:
                    has_significant_diff = True
        
        if has_significant_diff:
            print("  结论：不同会话的 EER 存在显著差异")
        else:
            print("  结论：不同会话的 EER 无显著差异")
    else:
        print("  结论：不同会话的 EER 无显著差异")
    print()

def analyze_posture_eer(df):
    """分析两种Posture的EER差异性"""
    print("=" * 60)
    print("3. 不同Posture的EER统计分析")
    print("=" * 60)
    
    # 检查Posture列的取值
    print("Posture分布:")
    print(df['Posture'].value_counts())
    print()
    
    # 获取两种posture的数据
    posture_values = df['Posture'].unique()
    if len(posture_values) >= 2:
        posture1_data = df[df['Posture'] == posture_values[0]]['EER'].values
        posture2_data = df[df['Posture'] == posture_values[1]]['EER'].values
        
        print(f"{posture_values[0]} 样本数: {len(posture1_data)}, 均值: {np.mean(posture1_data):.4f}, 标准差: {np.std(posture1_data):.4f}")
        print(f"{posture_values[1]} 样本数: {len(posture2_data)}, 均值: {np.mean(posture2_data):.4f}, 标准差: {np.std(posture2_data):.4f}")
        print()
        
        # 正态性检验
        print("正态性检验 (Shapiro-Wilk):")
        stat, p1 = stats.shapiro(posture1_data)
        stat, p2 = stats.shapiro(posture2_data)
        print(f"  {posture_values[0]}: W={p1:.4f} {'(正态分布)' if p1 > 0.05 else '(非正态分布)'}")
        print(f"  {posture_values[1]}: W={p2:.4f} {'(正态分布)' if p2 > 0.05 else '(非正态分布)'}")
        print()
        
        # 方差齐性检验
        print("方差齐性检验 (Levene):")
        stat, p = stats.levene(posture1_data, posture2_data)
        print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
        print()
        
        # 统计检验
        print("组间差异检验:")
        if p1 > 0.05 and p2 > 0.05 and p > 0.05:
            t_stat, p_val = stats.ttest_ind(posture1_data, posture2_data)
            print(f"  独立样本t检验: t={t_stat:.4f}, p={p_val:.4f}")
        else:
            u_stat, p_val = stats.mannwhitneyu(posture1_data, posture2_data)
            print(f"  Mann-Whitney U检验: U={u_stat:.0f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print(f"  结论：两种Posture的EER存在显著差异 (p < 0.05)")
            if np.mean(posture1_data) < np.mean(posture2_data):
                print(f"  {posture_values[0]} 的EER显著低于 {posture_values[1]}")
            else:
                print(f"  {posture_values[1]} 的EER显著低于 {posture_values[0]}")
        else:
            print("  结论：两种Posture的EER无显著差异 (p >= 0.05)")
    else:
        print(f"  警告：仅检测到 {len(posture_values)} 种Posture，无法进行比较")
    print()

if __name__ == '__main__':
    df = load_all_data()
    print(f"总样本数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    print()
    
    analyze_pattern_eer(df)
    analyze_session_eer(df)
    analyze_posture_eer(df)


#总样本数: 1115
#数据列: ['Posture', 'Pattern', '用户ID', 'TimePeriod', 'EER', 'session']

# ============================================================
# 1. 不同Pattern的EER统计分析
# ============================================================
# four 样本数: 365, 均值: 0.0576, 标准差: 0.1047
# three 样本数: 376, 均值: 0.0288, 标准差: 0.0660
# two 样本数: 374, 均值: 0.0432, 标准差: 0.0926

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.6064, p=0.0000 (非正态分布)
#   three: W=0.4881, p=0.0000 (非正态分布)
#   two: W=0.5268, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=9.6178, p=0.0001 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 12.3993, p = 0.0020
#   效应量 η² = 0.0111 (小效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#              four     three       two
# four   1.000000  0.001331  0.350739
# three  0.001331  1.000000  0.151857
# two    0.350739  0.151857  1.000000
#     four vs three: p = 0.0013 *
#     four vs two: p = 0.3507
#     three vs two: p = 0.1519

#   结论：不同 Pattern 的 EER 存在显著差异
#   EER 最低的 Pattern: three (均值=0.0288)

# ============================================================
# 2. 不同会话的EER统计分析
# ============================================================
# Session 1 样本数: 259, 均值: 0.0484, 标准差: 0.1000
# Session 2 样本数: 284, 均值: 0.0505, 标准差: 0.0979
# Session 3 样本数: 286, 均值: 0.0391, 标准差: 0.0775
# Session 4 样本数: 286, 均值: 0.0348, 标准差: 0.0821

# 正态性检验 (Shapiro-Wilk):
#   Session 1: W=0.5501, p=0.0000 (非正态分布)
#   Session 2: W=0.5653, p=0.0000 (非正态分布)
#   Session 3: W=0.5447, p=0.0000 (非正态分布)
#   Session 4: W=0.4910, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=1.9333, p=0.1224 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(3) = 3.7511, p = 0.2896
#   效应量 η² = 0.0034 (小效应)

#   结论：不同会话的 EER 无显著差异

# ============================================================
# 3. 不同Posture的EER统计分析
# ============================================================
# Posture分布:
# sit     566
# walk    549
# Name: Posture, dtype: int64

# sit 样本数: 566, 均值: 0.0412, 标准差: 0.0936
# walk 样本数: 549, 均值: 0.0450, 标准差: 0.0858

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.0000 (非正态分布)
#   walk: W=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=0.5003, p=0.4795 (方差齐性)

# 组间差异检验:
#   Mann-Whitney U检验: U=150620, p=0.2320
#   结论：两种Posture的EER无显著差异 (p >= 0.05)
