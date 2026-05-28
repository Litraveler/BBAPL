import pandas as pd
import os
import numpy as np
# ====================================================================================================
# Step: 2 
# ====================================================================================================
# 1. 定义文件目录和待处理的CSV文件
directory = r"E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas"
# directory = r"E:\论文撰写记录\投稿\StrokePL\Codes\datas"
csv_file = "touch_data.csv"

# 2. 拼接完整文件路径
file_path = os.path.join(directory, csv_file)

# 定义函数：对单组数据计算四分位并剔除异常值
def process_group_outliers(df_group, group_name):
    """
    对单个分组数据计算四分位、剔除异常值，并返回处理前后的统计信息
    :param df_group: 单个分组的DataFrame
    :param group_name: 分组名称 (Posture, pattern)
    :return: 处理后的数据、处理前统计、处理后统计、异常值信息
    """
    # 提取时长列（秒级）
    duration_col = "touch_duration_s"
    
    # ========== 处理前统计 ==========
    pre_stats = {
        "分组": group_name,
        "处理前样本数": len(df_group),
        "处理前均值(秒)": df_group[duration_col].mean(),
        "处理前最大值(秒)": df_group[duration_col].max(),
        "处理前最小值(秒)": df_group[duration_col].min(),
        "处理前Q1(秒)": df_group[duration_col].quantile(0.25),
        "处理前Q3(秒)": df_group[duration_col].quantile(0.75),
        "处理前IQR(秒)": df_group[duration_col].quantile(0.75) - df_group[duration_col].quantile(0.25)
    }
    
    # ========== 计算四分位并剔除异常值（IQR法） ==========
    Q1 = pre_stats["处理前Q1(秒)"]
    Q3 = pre_stats["处理前Q3(秒)"]
    IQR = pre_stats["处理前IQR(秒)"]
    upper_bound = Q3 + 1.5 * IQR
    
    # 过滤异常值（只保留上下限内的数据）
    filtered_group = df_group[df_group[duration_col] <= upper_bound]
    
    # ========== 异常值信息 ==========
    outlier_count = len(df_group) - len(filtered_group)
    outlier_info = {
        "分组": group_name,
        "异常值上限(秒)": upper_bound,
        "异常值数量": outlier_count,
        "异常值占比": f"{outlier_count/len(df_group)*100:.2f}%" if len(df_group) > 0 else "0.00%"
    }
    
    # ========== 处理后统计 ==========
    post_stats = {
        "分组": group_name,
        "处理后样本数": len(filtered_group),
        "处理后均值(秒)": filtered_group[duration_col].mean() if len(filtered_group) > 0 else 0,
        "处理后最大值(秒)": filtered_group[duration_col].max() if len(filtered_group) > 0 else 0,
        "处理后最小值(秒)": filtered_group[duration_col].min() if len(filtered_group) > 0 else 0,
        "处理后Q1(秒)": filtered_group[duration_col].quantile(0.25) if len(filtered_group) > 0 else 0,
        "处理后Q3(秒)": filtered_group[duration_col].quantile(0.75) if len(filtered_group) > 0 else 0,
        "处理后IQR(秒)": filtered_group[duration_col].quantile(0.75) - filtered_group[duration_col].quantile(0.25) if len(filtered_group) > 0 else 0
    }
    
    return filtered_group, pre_stats, post_stats, outlier_info

# 3. 主处理流程
try:
    # 读取CSV文件
    df = pd.read_csv(file_path, usecols=["Sample ID", "Posture", "pattern", "Time"])
    
    # 步骤1：计算每个样本的触摸时长
    sample_time_stats = df.groupby(["Sample ID", "Posture", "pattern"])["Time"].agg(["min", "max"]).reset_index()
    sample_time_stats["touch_duration_ns"] = sample_time_stats["max"] - sample_time_stats["min"]
    sample_time_stats["touch_duration_s"] = sample_time_stats["touch_duration_ns"] / 10**9
    
    # 初始化存储所有结果的列表
    all_pre_stats = []    # 所有分组处理前的统计
    all_post_stats = []   # 所有分组处理后的统计
    all_outlier_info = [] # 所有分组的异常值信息
    all_filtered_data = []# 所有分组处理后的数据
    
    # 步骤2：按 (Posture, pattern) 分组处理每个组的异常值
    grouped = sample_time_stats.groupby(["Posture", "pattern"])
    total_groups = len(grouped)
    print("=" * 100)
    print(f"开始处理 {total_groups} 个分组的异常值（按Posture+pattern分组）")
    print("=" * 100)
    
    for (posture, pattern), group_df in grouped:
        group_name = f"{posture}_{pattern}"
        # 处理当前分组的异常值
        filtered_df, pre_stats, post_stats, outlier_info = process_group_outliers(group_df, group_name)
        
        # 存储结果
        all_pre_stats.append(pre_stats)
        all_post_stats.append(post_stats)
        all_outlier_info.append(outlier_info)
        all_filtered_data.append(filtered_df)
        
        # 输出当前分组的对比结果（简洁版）
        print(f"\n【分组：{group_name}】")
        print(f"  处理前：样本数={pre_stats['处理前样本数']}，均值={pre_stats['处理前均值(秒)']:.9f}秒，最大值={pre_stats['处理前最大值(秒)']:.9f}秒")
        print(f"  异常值：数量={outlier_info['异常值数量']}，占比={outlier_info['异常值占比']}，上限={outlier_info['异常值上限(秒)']:.9f}秒")
        print(f"  处理后：样本数={post_stats['处理后样本数']}，均值={post_stats['处理后均值(秒)']:.9f}秒，最大值={post_stats['处理后最大值(秒)']:.9f}秒")
    
    # 步骤3：合并所有处理后的数据，生成全局统计
    combined_filtered = pd.concat(all_filtered_data, ignore_index=True)
    pre_combined_stats = {
        "全局处理前样本数": len(sample_time_stats),
        "全局处理前均值(秒)": sample_time_stats["touch_duration_s"].mean(),
        "全局处理前最大值(秒)": sample_time_stats["touch_duration_s"].max()
    }
    post_combined_stats = {
        "全局处理后样本数": len(combined_filtered),
        "全局处理后均值(秒)": combined_filtered["touch_duration_s"].mean() if len(combined_filtered) > 0 else 0,
        "全局处理后最大值(秒)": combined_filtered["touch_duration_s"].max() if len(combined_filtered) > 0 else 0
    }
    
    # 步骤4：输出全局对比结果
    print("\n" + "=" * 100)
    print("【全局统计对比】")
    print(f"  处理前：总样本数={pre_combined_stats['全局处理前样本数']}，均值={pre_combined_stats['全局处理前均值(秒)']:.9f}秒，最大值={pre_combined_stats['全局处理前最大值(秒)']:.9f}秒")
    print(f"  处理后：总样本数={post_combined_stats['全局处理后样本数']}，均值={post_combined_stats['全局处理后均值(秒)']:.9f}秒，最大值={post_combined_stats['全局处理后最大值(秒)']:.9f}秒")
    print("=" * 100)
    
    # 步骤5：保存详细结果到CSV（便于论文整理）
    # 转换为DataFrame
    pre_stats_df = pd.DataFrame(all_pre_stats)
    post_stats_df = pd.DataFrame(all_post_stats)
    outlier_info_df = pd.DataFrame(all_outlier_info)
    
    # 保存文件
    pre_stats_df.to_csv(os.path.join(directory, "分组处理前统计.csv"), index=False, encoding="utf-8")
    post_stats_df.to_csv(os.path.join(directory, "分组处理后统计.csv"), index=False, encoding="utf-8")
    outlier_info_df.to_csv(os.path.join(directory, "分组异常值信息.csv"), index=False, encoding="utf-8")
    combined_filtered.to_csv(os.path.join(directory, "处理后所有样本数据.csv"), index=False, encoding="utf-8")
    
    print(f"\n详细统计结果已保存至 {directory} 目录下：")
    print("  - 分组处理前统计.csv")
    print("  - 分组处理后统计.csv")
    print("  - 分组异常值信息.csv")
    print("  - 处理后所有样本数据.csv")

except FileNotFoundError:
    print(f"错误：未找到文件 {csv_file}，请检查文件路径是否正确。")
except KeyError as e:
    print(f"错误：文件 {csv_file} 中不存在 '{e.args[0]}' 列，请检查列名是否正确（区分大小写）。")
except TypeError:
    print(f"错误：文件 {csv_file} 的 'Time' 列包含非数值类型数据，无法计算时长。")
except Exception as e:
    print(f"处理文件 {csv_file} 时发生未知错误：{str(e)}")