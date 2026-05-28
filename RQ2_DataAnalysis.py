import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

def get_pattern_number(pattern):
    """从Pattern名称中提取数字部分"""
    number_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8
    }
    pattern_lower = pattern.lower()
    parts = pattern_lower.split('_')
    if len(parts) >= 2:
        suffix = parts[-1]
        if suffix in number_map:
            return number_map[suffix]
    return 0

def load_gnb_ovsm_data():
    """加载GNB和OVSVM的数据"""
    gnb_ovsm_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_2.csv', 2),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_3.csv', 3),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_1_4.csv', 4)
    ]
    all_data = []
    for file_path, session in gnb_ovsm_files:
        df = pd.read_csv(file_path)
        df['Session'] = session
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_strokepl_data():
    """加载StrokePL的数据"""
    strokepl_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_2.csv', 2),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_3.csv', 3),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_1_to_4.csv', 4)
    ]
    all_data = []
    for file_path, session in strokepl_files:
        df = pd.read_csv(file_path)
        df['Session'] = session
        df['Classifier'] = 'StrokePL'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_all_classifier_data():
    """加载所有分类器的数据"""
    gnb_ovsm_df = load_gnb_ovsm_data()
    strokepl_df = load_strokepl_data()
    return pd.concat([gnb_ovsm_df, strokepl_df], ignore_index=True)

def analyze_pattern_eer(df, classifier_name):
    """分析不同pattern（four、three、two）的EER差异性"""
    print("=" * 70)
    print(f"【{classifier_name}】1. 不同Pattern的EER统计分析")
    print("=" * 70)
    
    df_filtered = df[df['Classifier'] == classifier_name]
    
    four_eer = df_filtered[df_filtered['Pattern'].str.startswith('four')]['EER'].values
    three_eer = df_filtered[df_filtered['Pattern'].str.startswith('three')]['EER'].values
    two_eer = df_filtered[df_filtered['Pattern'].str.startswith('two')]['EER'].values
    
    print(f"four 样本数: {len(four_eer)}, 均值: {np.mean(four_eer):.4f}, 标准差: {np.std(four_eer):.4f}")
    print(f"three 样本数: {len(three_eer)}, 均值: {np.mean(three_eer):.4f}, 标准差: {np.std(three_eer):.4f}")
    print(f"two 样本数: {len(two_eer)}, 均值: {np.mean(two_eer):.4f}, 标准差: {np.std(two_eer):.4f}")
    print()
    
    print("正态性检验 (Shapiro-Wilk):")
    stat, p_four = stats.shapiro(four_eer)
    print(f"  four: W={stat:.4f}, p={p_four:.4f} {'(正态分布)' if p_four > 0.05 else '(非正态分布)'}")
    stat, p_three = stats.shapiro(three_eer)
    print(f"  three: W={stat:.4f}, p={p_three:.4f} {'(正态分布)' if p_three > 0.05 else '(非正态分布)'}")
    stat, p_two = stats.shapiro(two_eer)
    print(f"  two: W={stat:.4f}, p={p_two:.4f} {'(正态分布)' if p_two > 0.05 else '(非正态分布)'}")
    print()
    
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(four_eer, three_eer, two_eer)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(four_eer, three_eer, two_eer)
    print(f"  H(2) = {h_stat:.4f}, p = {p_kw:.4f}")
    
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
    
    if p_kw < 0.05:
        print("事后两两比较 (Dunn's test with Bonferroni 校正):")
        data_list = [four_eer, three_eer, two_eer]
        groups = ['four', 'three', 'two']
        
        dunn_result = sp.posthoc_dunn(data_list, p_adjust='bonferroni')
        dunn_result.columns = groups
        dunn_result.index = groups
        print(f"  {dunn_result}")
        
        p_four_vs_three = dunn_result.loc['four', 'three']
        p_four_vs_two = dunn_result.loc['four', 'two']
        p_three_vs_two = dunn_result.loc['three', 'two']
        
        print(f"    four vs three: p = {p_four_vs_three:.4f} {'*' if p_four_vs_three < 0.05 else ''}")
        print(f"    four vs two: p = {p_four_vs_two:.4f} {'*' if p_four_vs_two < 0.05 else ''}")
        print(f"    three vs two: p = {p_three_vs_two:.4f} {'*' if p_three_vs_two < 0.05 else ''}")
        
        print("\n  结论：不同 Pattern 的 EER 存在显著差异")
        means = {'four': np.mean(four_eer), 'three': np.mean(three_eer), 'two': np.mean(two_eer)}
        min_pattern = min(means, key=means.get)
        print(f"  EER 最低的 Pattern: {min_pattern} (均值={means[min_pattern]:.4f})")
    else:
        print("  结论：不同 Pattern 的 EER 无显著差异")
    print()

def analyze_session_eer(df, classifier_name):
    """分析不同会话的EER差异性"""
    print("=" * 70)
    print(f"【{classifier_name}】2. 不同会话的EER统计分析")
    print("=" * 70)
    
    df_filtered = df[df['Classifier'] == classifier_name]
    
    sessions = df_filtered['Session'].unique()
    session_data = [df_filtered[df_filtered['Session'] == s]['EER'].values for s in sessions]
    
    for i, data in enumerate(session_data, 1):
        print(f"Session {i} 样本数: {len(data)}, 均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    print()
    
    print("正态性检验 (Shapiro-Wilk):")
    for i, data in enumerate(session_data, 1):
        stat, p = stats.shapiro(data)
        is_normal = p > 0.05
        print(f"  Session {i}: W={stat:.4f}, p={p:.4f} {'(正态分布)' if is_normal else '(非正态分布)'}")
    print()
    
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(*session_data)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(*session_data)
    print(f"  H({len(session_data)-1}) = {h_stat:.4f}, p = {p_kw:.4f}")
    
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
    
    if p_kw < 0.05:
        print("事后两两比较 (Dunn's test with Bonferroni 校正):")
        dunn_result = sp.posthoc_dunn(session_data, p_adjust='bonferroni')
        print(f"  {dunn_result}")
        
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

def analyze_posture_eer(df, classifier_name):
    """分析两种Posture的EER差异性"""
    print("=" * 70)
    print(f"【{classifier_name}】3. 不同Posture的EER统计分析")
    print("=" * 70)
    
    df_filtered = df[df['Classifier'] == classifier_name]
    
    print("Posture分布:")
    print(df_filtered['Posture'].value_counts())
    print()
    
    posture_values = df_filtered['Posture'].unique()
    if len(posture_values) >= 2:
        posture1_data = df_filtered[df_filtered['Posture'] == posture_values[0]]['EER'].values
        posture2_data = df_filtered[df_filtered['Posture'] == posture_values[1]]['EER'].values
        
        print(f"{posture_values[0]} 样本数: {len(posture1_data)}, 均值: {np.mean(posture1_data):.4f}, 标准差: {np.std(posture1_data):.4f}")
        print(f"{posture_values[1]} 样本数: {len(posture2_data)}, 均值: {np.mean(posture2_data):.4f}, 标准差: {np.std(posture2_data):.4f}")
        print()
        
        print("正态性检验 (Shapiro-Wilk):")
        stat, p1 = stats.shapiro(posture1_data)
        stat, p2 = stats.shapiro(posture2_data)
        print(f"  {posture_values[0]}: W={p1:.4f} {'(正态分布)' if p1 > 0.05 else '(非正态分布)'}")
        print(f"  {posture_values[1]}: W={p2:.4f} {'(正态分布)' if p2 > 0.05 else '(非正态分布)'}")
        print()
        
        print("方差齐性检验 (Levene):")
        stat, p = stats.levene(posture1_data, posture2_data)
        print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
        print()
        
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

def analyze_classifier_eer(df):
    """分析三种分类器之间的EER差异性"""
    print("=" * 70)
    print("【分类器对比】三种分类器的EER统计差异分析")
    print("=" * 70)
    
    classifiers = ['GNB', 'OVSVM', 'StrokePL']
    classifier_data = []
    
    for clf in classifiers:
        clf_df = df[df['Classifier'] == clf]
        eer_values = clf_df['EER'].values
        classifier_data.append(eer_values)
        print(f"{clf} 样本数: {len(eer_values)}, 均值: {np.mean(eer_values):.4f}, 标准差: {np.std(eer_values):.4f}")
    print()
    
    print("正态性检验 (Shapiro-Wilk):")
    for i, data in enumerate(classifier_data):
        stat, p = stats.shapiro(data)
        is_normal = p > 0.05
        print(f"  {classifiers[i]}: W={stat:.4f}, p={p:.4f} {'(正态分布)' if is_normal else '(非正态分布)'}")
    print()
    
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(*classifier_data)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(*classifier_data)
    print(f"  H(2) = {h_stat:.4f}, p = {p_kw:.4f}")
    
    n_total = sum(len(data) for data in classifier_data)
    eta_squared = h_stat / (n_total - 1)
    print(f"  效应量 η² = {eta_squared:.4f}", end="")
    if eta_squared > 0.14:
        print(" (大效应)")
    elif eta_squared > 0.06:
        print(" (中等效应)")
    else:
        print(" (小效应)")
    print()
    
    if p_kw < 0.05:
        print("事后两两比较 (Dunn's test with Bonferroni 校正):")
        dunn_result = sp.posthoc_dunn(classifier_data, p_adjust='bonferroni')
        dunn_result.columns = classifiers
        dunn_result.index = classifiers
        print(f"  {dunn_result}")
        
        comparisons = []
        has_significant_diff = False
        
        for i in range(len(classifiers)):
            for j in range(i + 1, len(classifiers)):
                p_val = dunn_result.iloc[i, j]
                comparisons.append((classifiers[i], classifiers[j], p_val))
                if p_val < 0.05:
                    has_significant_diff = True
                print(f"    {classifiers[i]} vs {classifiers[j]}: p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
        
        if has_significant_diff:
            print("\n  结论：不同分类器的 EER 存在显著差异")
            means = {clf: np.mean(data) for clf, data in zip(classifiers, classifier_data)}
            sorted_means = sorted(means.items(), key=lambda x: x[1])
            print("  EER 均值排序:")
            for i, (clf, mean) in enumerate(sorted_means, 1):
                print(f"    {i}. {clf}: {mean:.4f}")
        else:
            print("\n  结论：不同分类器的 EER 无显著差异")
    else:
        print("\n  结论：不同分类器的 EER 无显著差异")
    print()

if __name__ == '__main__':
    df = load_all_classifier_data()
    print(f"总样本数: {len(df)}")
    print(f"分类器分布:")
    print(df['Classifier'].value_counts())
    print()
    
    classifiers = ['GNB', 'OVSVM', 'StrokePL']
    
    for clf in classifiers:
        analyze_pattern_eer(df, clf)
        analyze_session_eer(df, clf)
        analyze_posture_eer(df, clf)
    
    analyze_classifier_eer(df)

# 总样本数: 5321
# 分类器分布:
# GNB         2271
# OVSVM       2271
# StrokePL     779
# Name: Classifier, dtype: int64

# ======================================================================
# 【GNB】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 714, 均值: 0.4304, 标准差: 0.1578
# three 样本数: 816, 均值: 0.4337, 标准差: 0.1470
# two 样本数: 741, 均值: 0.4588, 标准差: 0.1466

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.8785, p=0.0000 (非正态分布)
#   three: W=0.8933, p=0.0000 (非正态分布)
#   two: W=0.8924, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=1.9434, p=0.1435 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 13.0947, p = 0.0014
#   效应量 η² = 0.0058 (小效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#              four     three       two
# four   1.000000  1.000000  0.007847
# three  1.000000  1.000000  0.003367
# two    0.007847  0.003367  1.000000
#     four vs three: p = 1.0000
#     four vs two: p = 0.0078 *
#     three vs two: p = 0.0034 *

#   结论：不同 Pattern 的 EER 存在显著差异
#   EER 最低的 Pattern: four (均值=0.4304)

# ======================================================================
# 【GNB】2. 不同会话的EER统计分析
# ======================================================================
# Session 1 样本数: 757, 均值: 0.4328, 标准差: 0.1516
# Session 2 样本数: 757, 均值: 0.4443, 标准差: 0.1517
# Session 3 样本数: 757, 均值: 0.4456, 标准差: 0.1490

# 正态性检验 (Shapiro-Wilk):
#   Session 1: W=0.8898, p=0.0000 (非正态分布)
#   Session 2: W=0.8745, p=0.0000 (非正态分布)
#   Session 3: W=0.8983, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=0.6786, p=0.5074 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 2.6134, p = 0.2707
#   效应量 η² = 0.0012 (小效应)

#   结论：不同会话的 EER 无显著差异

# ======================================================================
# 【GNB】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# sit     1200
# walk    1071
# Name: Posture, dtype: int64

# sit 样本数: 1200, 均值: 0.4650, 标准差: 0.1325
# walk 样本数: 1071, 均值: 0.4138, 标准差: 0.1650

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.0000 (非正态分布)
#   walk: W=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=98.9806, p=0.0000 (方差不齐)

# 组间差异检验:
#   Mann-Whitney U检验: U=759977, p=0.0000
#   结论：两种Posture的EER存在显著差异 (p < 0.05)
#   walk 的EER显著低于 sit

# ======================================================================
# 【OVSVM】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 714, 均值: 0.2510, 标准差: 0.2513
# three 样本数: 816, 均值: 0.2457, 标准差: 0.2628
# two 样本数: 741, 均值: 0.2777, 标准差: 0.2641

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.8494, p=0.0000 (非正态分布)
#   three: W=0.8388, p=0.0000 (非正态分布)
#   two: W=0.8826, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=3.6519, p=0.0261 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 8.3684, p = 0.0152
#   效应量 η² = 0.0037 (小效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#              four     three       two
# four   1.000000  0.941888  0.220500
# three  0.941888  1.000000  0.012454
# two    0.220500  0.012454  1.000000
#     four vs three: p = 0.9419
#     four vs two: p = 0.2205
#     three vs two: p = 0.0125 *

#   结论：不同 Pattern 的 EER 存在显著差异
#   EER 最低的 Pattern: three (均值=0.2457)

# ======================================================================
# 【OVSVM】2. 不同会话的EER统计分析
# ======================================================================
# Session 1 样本数: 757, 均值: 0.2486, 标准差: 0.2546
# Session 2 样本数: 757, 均值: 0.2590, 标准差: 0.2626
# Session 3 样本数: 757, 均值: 0.2658, 标准差: 0.2626

# 正态性检验 (Shapiro-Wilk):
#   Session 1: W=0.8576, p=0.0000 (非正态分布)
#   Session 2: W=0.8539, p=0.0000 (非正态分布)
#   Session 3: W=0.8613, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=0.0232, p=0.9771 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 1.8044, p = 0.4057
#   效应量 η² = 0.0008 (小效应)

#   结论：不同会话的 EER 无显著差异

# ======================================================================
# 【OVSVM】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# sit     1200
# walk    1071
# Name: Posture, dtype: int64

# sit 样本数: 1200, 均值: 0.2829, 标准差: 0.2850
# walk 样本数: 1071, 均值: 0.2297, 标准差: 0.2257

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.0000 (非正态分布)
#   walk: W=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=32.4392, p=0.0000 (方差不齐)

# 组间差异检验:
#   Mann-Whitney U检验: U=690423, p=0.0019
#   结论：两种Posture的EER存在显著差异 (p < 0.05)
#   walk 的EER显著低于 sit

# ======================================================================
# 【StrokePL】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 257, 均值: 0.1568, 标准差: 0.2023
# three 样本数: 264, 均值: 0.1579, 标准差: 0.1892
# two 样本数: 258, 均值: 0.1797, 标准差: 0.2109

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.7738, p=0.0000 (非正态分布)
#   three: W=0.8112, p=0.0000 (非正态分布)
#   two: W=0.8192, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=1.0428, p=0.3530 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 1.6819, p = 0.4313
#   效应量 η² = 0.0022 (小效应)

#   结论：不同 Pattern 的 EER 无显著差异

# ======================================================================
# 【StrokePL】2. 不同会话的EER统计分析
# ======================================================================
# Session 1 样本数: 259, 均值: 0.1620, 标准差: 0.2072
# Session 2 样本数: 260, 均值: 0.1559, 标准差: 0.1916
# Session 3 样本数: 260, 均值: 0.1764, 标准差: 0.2039

# 正态性检验 (Shapiro-Wilk):
#   Session 1: W=0.7815, p=0.0000 (非正态分布)
#   Session 2: W=0.7928, p=0.0000 (非正态分布)
#   Session 3: W=0.8291, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=0.9976, p=0.3692 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 1.4837, p = 0.4762
#   效应量 η² = 0.0019 (小效应)

#   结论：不同会话的 EER 无显著差异

# ======================================================================
# 【StrokePL】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# sit     426
# walk    353
# Name: Posture, dtype: int64

# sit 样本数: 426, 均值: 0.2295, 标准差: 0.2243
# walk 样本数: 353, 均值: 0.0866, 标准差: 0.1320

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.0000 (非正态分布)
#   walk: W=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=83.7770, p=0.0000 (方差不齐)

# 组间差异检验:
#   Mann-Whitney U检验: U=103742, p=0.0000
#   结论：两种Posture的EER存在显著差异 (p < 0.05)
#   walk 的EER显著低于 sit

# ======================================================================
# 【分类器对比】三种分类器的EER统计差异分析
# ======================================================================
# GNB 样本数: 2271, 均值: 0.4409, 标准差: 0.1509
# OVSVM 样本数: 2271, 均值: 0.2578, 标准差: 0.2601
# StrokePL 样本数: 779, 均值: 0.1648, 标准差: 0.2012

# 正态性检验 (Shapiro-Wilk):
#   GNB: W=0.8894, p=0.0000 (非正态分布)
#   OVSVM: W=0.8587, p=0.0000 (非正态分布)
#   StrokePL: W=0.8030, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=207.2663, p=0.0000 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 1357.3940, p = 0.0000
#   效应量 η² = 0.2551 (大效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#                       GNB          OVSVM       StrokePL
# GNB        1.000000e+00  2.404528e-206  2.403718e-197
# OVSVM     2.404528e-206   1.000000e+00   2.059155e-15
# StrokePL  2.403718e-197   2.059155e-15   1.000000e+00
#     GNB vs OVSVM: p = 0.0000 *
#     GNB vs StrokePL: p = 0.0000 *
#     OVSVM vs StrokePL: p = 0.0000 *

#   结论：不同分类器的 EER 存在显著差异
#   EER 均值排序:
#     1. StrokePL: 0.1648
#     2. OVSVM: 0.2578
#     3. GNB: 0.4409