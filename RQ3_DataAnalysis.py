import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

def load_strokepl_data():
    strokepl_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_2_to_3.csv', '2-3'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_3_to_4.csv', '3-4'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/timeperiod_level_eer_results_2-3_to_4.csv', '2&3-4')
    ]
    all_data = []
    for file_path, combo in strokepl_files:
        df = pd.read_csv(file_path)
        df['Combo'] = combo
        df['Classifier'] = 'StrokePL'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_gnb_ovsm_data():
    gnb_ovsm_files = [
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_2_3.csv', '2-3'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_3_4.csv', '3-4'),
        ('e:/【论文撰写】/投稿/StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices/Codes/GaussianNaiveBayesC/fusion_recognition_results_23_4.csv', '2&3-4')
    ]
    all_data = []
    for file_path, combo in gnb_ovsm_files:
        df = pd.read_csv(file_path)
        df['Combo'] = combo
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_all_classifier_data():
    gnb_ovsm_df = load_gnb_ovsm_data()
    strokepl_df = load_strokepl_data()
    return pd.concat([gnb_ovsm_df, strokepl_df], ignore_index=True)

def analyze_pattern_eer(df, classifier_name):
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

def analyze_combo_eer(df, classifier_name):
    print("=" * 70)
    print(f"【{classifier_name}】2. 不同训练测试组合的EER统计分析")
    print("=" * 70)
    
    df_filtered = df[df['Classifier'] == classifier_name]
    
    combos = df_filtered['Combo'].unique()
    combo_data = [df_filtered[df_filtered['Combo'] == c]['EER'].values for c in combos]
    
    for i, c in enumerate(combos):
        print(f"{c} 样本数: {len(combo_data[i])}, 均值: {np.mean(combo_data[i]):.4f}, 标准差: {np.std(combo_data[i]):.4f}")
    print()
    
    print("正态性检验 (Shapiro-Wilk):")
    for i, c in enumerate(combos):
        stat, p = stats.shapiro(combo_data[i])
        print(f"  {c}: W={stat:.4f}, p={p:.4f} {'(正态分布)' if p > 0.05 else '(非正态分布)'}")
    print()
    
    print("方差齐性检验 (Levene):")
    stat, p = stats.levene(*combo_data)
    print(f"  W={stat:.4f}, p={p:.4f} {'(方差齐性)' if p > 0.05 else '(方差不齐)'}")
    print()
    
    print("组间差异检验 (Kruskal-Wallis H 检验):")
    h_stat, p_kw = stats.kruskal(*combo_data)
    print(f"  H({len(combo_data)-1}) = {h_stat:.4f}, p = {p_kw:.4f}")
    
    n_total = sum(len(data) for data in combo_data)
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
        dunn_result = sp.posthoc_dunn(combo_data, p_adjust='bonferroni')
        dunn_result.columns = combos
        dunn_result.index = combos
        print(f"  {dunn_result}")
        
        n_combos = len(combo_data)
        has_significant_diff = False
        for i in range(n_combos):
            for j in range(i + 1, n_combos):
                p_val = dunn_result.iloc[i, j]
                print(f"    {combos[i]} vs {combos[j]}: p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
                if p_val < 0.05:
                    has_significant_diff = True
        
        if has_significant_diff:
            print("  结论：不同训练测试组合的 EER 存在显著差异")
        else:
            print("  结论：不同训练测试组合的 EER 无显著差异")
    else:
        print("  结论：不同训练测试组合的 EER 无显著差异")
    print()

def analyze_posture_eer(df, classifier_name):
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
        print(f"  {posture_values[0]}: W={stat:.4f} {'(正态分布)' if p1 > 0.05 else '(非正态分布)'}")
        print(f"  {posture_values[1]}: W={stat:.4f} {'(正态分布)' if p2 > 0.05 else '(非正态分布)'}")
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
        print(f"  {classifiers[i]}: W={stat:.4f}, p={p:.4f} {'(正态分布)' if p > 0.05 else '(非正态分布)'}")
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
        
        has_significant_diff = False
        
        for i in range(len(classifiers)):
            for j in range(i + 1, len(classifiers)):
                p_val = dunn_result.iloc[i, j]
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
        analyze_combo_eer(df, clf)
        analyze_posture_eer(df, clf)
    
    analyze_classifier_eer(df)


# 总样本数: 5397
# 分类器分布:
# GNB         2271
# OVSVM       2271
# StrokePL     855
# Name: Classifier, dtype: int64

# ======================================================================
# 【GNB】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 714, 均值: 0.3634, 标准差: 0.1872
# three 样本数: 816, 均值: 0.3857, 标准差: 0.1791
# two 样本数: 741, 均值: 0.3667, 标准差: 0.1864

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.9105, p=0.0000 (非正态分布)
#   three: W=0.8924, p=0.0000 (非正态分布)
#   two: W=0.9096, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=1.6844, p=0.1858 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 7.5154, p = 0.0233
#   效应量 η² = 0.0033 (小效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#              four     three       two
# four   1.000000  0.039707  1.000000
# three  0.039707  1.000000  0.082813
# two    1.000000  0.082813  1.000000
#     four vs three: p = 0.0397 *
#     four vs two: p = 1.0000
#     three vs two: p = 0.0828

#   结论：不同 Pattern 的 EER 存在显著差异
#   EER 最低的 Pattern: four (均值=0.3634)

# ======================================================================
# 【GNB】2. 不同训练测试组合的EER统计分析
# ======================================================================
# 2-3 样本数: 757, 均值: 0.4291, 标准差: 0.1595
# 3-4 样本数: 757, 均值: 0.4312, 标准差: 0.1456
# 2&3-4 样本数: 757, 均值: 0.2573, 标准差: 0.1881

# 正态性检验 (Shapiro-Wilk):
#   2-3: W=0.8563, p=0.0000 (非正态分布)
#   3-4: W=0.8861, p=0.0000 (非正态分布)
#   2&3-4: W=0.9284, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=63.2218, p=0.0000 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 403.4817, p = 0.0000
#   效应量 η² = 0.1777 (大效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#                   2-3           3-4         2&3-4
# 2-3    1.000000e+00  1.000000e+00  1.072590e-68
# 3-4    1.000000e+00  1.000000e+00  7.109417e-66
# 2&3-4  1.072590e-68  7.109417e-66  1.000000e+00
#     2-3 vs 3-4: p = 1.0000
#     2-3 vs 2&3-4: p = 0.0000 *
#     3-4 vs 2&3-4: p = 0.0000 *
#   结论：不同训练测试组合的 EER 存在显著差异

# ======================================================================
# 【GNB】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# sit     1200
# walk    1071
# Name: Posture, dtype: int64

# sit 样本数: 1200, 均值: 0.4005, 标准差: 0.1727
# walk 样本数: 1071, 均值: 0.3411, 标准差: 0.1918

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.9215 (非正态分布)
#   walk: W=0.9215 (非正态分布)

# 方差齐性检验 (Levene):
#   W=26.6569, p=0.0000 (方差不齐)

# 组间差异检验:
#   Mann-Whitney U检验: U=757628, p=0.0000
#   结论：两种Posture的EER存在显著差异 (p < 0.05)
#   walk 的EER显著低于 sit

# ======================================================================
# 【OVSVM】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 714, 均值: 0.1823, 标准差: 0.2124
# three 样本数: 816, 均值: 0.1792, 标准差: 0.2081
# two 样本数: 741, 均值: 0.1961, 标准差: 0.2117

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.8163, p=0.0000 (非正态分布)
#   three: W=0.8192, p=0.0000 (非正态分布)
#   two: W=0.8392, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=0.1660, p=0.8471 (方差齐性)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 4.7104, p = 0.0949
#   效应量 η² = 0.0021 (小效应)

#   结论：不同 Pattern 的 EER 无显著差异

# ======================================================================
# 【OVSVM】2. 不同训练测试组合的EER统计分析
# ======================================================================
# 2-3 样本数: 757, 均值: 0.2114, 标准差: 0.2352
# 3-4 样本数: 757, 均值: 0.1799, 标准差: 0.2118
# 2&3-4 样本数: 757, 均值: 0.1657, 标准差: 0.1789

# 正态性检验 (Shapiro-Wilk):
#   2-3: W=0.8229, p=0.0000 (非正态分布)
#   3-4: W=0.8138, p=0.0000 (非正态分布)
#   2&3-4: W=0.8467, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=11.9503, p=0.0000 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 8.1180, p = 0.0173
#   效应量 η² = 0.0036 (小效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#               2-3       3-4     2&3-4
# 2-3    1.000000  0.018489  0.121325
# 3-4    0.018489  1.000000  1.000000
# 2&3-4  0.121325  1.000000  1.000000
#     2-3 vs 3-4: p = 0.0185 *
#     2-3 vs 2&3-4: p = 0.1213
#     3-4 vs 2&3-4: p = 1.0000
#   结论：不同训练测试组合的 EER 存在显著差异

# ======================================================================
# 【OVSVM】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# sit     1200
# walk    1071
# Name: Posture, dtype: int64

# sit 样本数: 1200, 均值: 0.1900, 标准差: 0.2220
# walk 样本数: 1071, 均值: 0.1808, 标准差: 0.1973

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.8454 (非正态分布)
#   walk: W=0.8454 (非正态分布)

# 方差齐性检验 (Levene):
#   W=1.0338, p=0.3094 (方差齐性)

# 组间差异检验:
#   Mann-Whitney U检验: U=645294, p=0.8599
#   结论：两种Posture的EER无显著差异 (p >= 0.05)

# ======================================================================
# 【StrokePL】1. 不同Pattern的EER统计分析
# ======================================================================
# four 样本数: 279, 均值: 0.0907, 标准差: 0.1384
# three 样本数: 288, 均值: 0.0988, 标准差: 0.1454
# two 样本数: 288, 均值: 0.1248, 标准差: 0.1769

# 正态性检验 (Shapiro-Wilk):
#   four: W=0.6947, p=0.0000 (非正态分布)
#   three: W=0.7169, p=0.0000 (非正态分布)
#   two: W=0.7407, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=3.9325, p=0.0200 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 4.7441, p = 0.0933
#   效应量 η² = 0.0056 (小效应)

#   结论：不同 Pattern 的 EER 无显著差异

# ======================================================================
# 【StrokePL】2. 不同训练测试组合的EER统计分析
# ======================================================================
# 2-3 样本数: 284, 均值: 0.0860, 标准差: 0.1220
# 3-4 样本数: 285, 均值: 0.1200, 标准差: 0.1734
# 2&3-4 样本数: 286, 均值: 0.1087, 标准差: 0.1637

# 正态性检验 (Shapiro-Wilk):
#   2-3: W=0.7347, p=0.0000 (非正态分布)
#   3-4: W=0.7288, p=0.0000 (非正态分布)
#   2&3-4: W=0.7029, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=3.5543, p=0.0290 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 2.6527, p = 0.2654
#   效应量 η² = 0.0031 (小效应)

#   结论：不同训练测试组合的 EER 无显著差异

# ======================================================================
# 【StrokePL】3. 不同Posture的EER统计分析
# ======================================================================
# Posture分布:
# walk    431
# sit     424
# Name: Posture, dtype: int64

# sit 样本数: 424, 均值: 0.1450, 标准差: 0.1804
# walk 样本数: 431, 均值: 0.0655, 标准差: 0.1128

# 正态性检验 (Shapiro-Wilk):
#   sit: W=0.6435 (非正态分布)
#   walk: W=0.6435 (非正态分布)

# 方差齐性检验 (Levene):
#   W=65.7674, p=0.0000 (方差不齐)

# 组间差异检验:
#   Mann-Whitney U检验: U=116226, p=0.0000
#   结论：两种Posture的EER存在显著差异 (p < 0.05)
#   walk 的EER显著低于 sit

# ======================================================================
# 【分类器对比】三种分类器的EER统计差异分析
# ======================================================================
# GNB 样本数: 2271, 均值: 0.3725, 标准差: 0.1843
# OVSVM 样本数: 2271, 均值: 0.1857, 标准差: 0.2108
# StrokePL 样本数: 855, 均值: 0.1049, 标准差: 0.1553

# 正态性检验 (Shapiro-Wilk):
#   GNB: W=0.9045, p=0.0000 (非正态分布)
#   OVSVM: W=0.8259, p=0.0000 (非正态分布)
#   StrokePL: W=0.7169, p=0.0000 (非正态分布)

# 方差齐性检验 (Levene):
#   W=42.4995, p=0.0000 (方差不齐)

# 组间差异检验 (Kruskal-Wallis H 检验):
#   H(2) = 1452.5004, p = 0.0000
#   效应量 η² = 0.2692 (大效应)

# 事后两两比较 (Dunn's test with Bonferroni 校正):
#                       GNB          OVSVM       StrokePL
# GNB        1.000000e+00  7.974718e-209  1.070335e-226
# OVSVM     7.974718e-209   1.000000e+00   2.801321e-20
# StrokePL  1.070335e-226   2.801321e-20   1.000000e+00
#     GNB vs OVSVM: p = 0.0000 *
#     GNB vs StrokePL: p = 0.0000 *
#     OVSVM vs StrokePL: p = 0.0000 *

#   结论：不同分类器的 EER 存在显著差异
#   EER 均值排序:
#     1. StrokePL: 0.1049
#     2. OVSVM: 0.1857
#     3. GNB: 0.3725