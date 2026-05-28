import pandas as pd
import os
import numpy as np
# ====================================================================================================
# Step: 3 
# ====================================================================================================
# 1. 定义文件目录和待处理的CSV文件（修改为sensor_data.csv）
directory = r"E:\论文撰写记录\投稿\StrokePL\Codes\datas"
# directory = r"E:\论文撰写记录\投稿\StrokePL\Codes\datas"
csv_file = "sensor_data.csv"  # 文件名修改

# 2. 拼接完整文件路径
file_path = os.path.join(directory, csv_file)

# 定义函数：对单组传感器数据（按样本行数）计算四分位并剔除异常值
def process_sensor_outliers(df_group, group_name):
    """
    对单个传感器分组数据，按样本的行数计算四分位、剔除异常值
    :param df_group: 单个传感器类型的DataFrame（包含所有样本）
    :param group_name: 分组名称 (SensorType)
    :return: 处理后的数据、处理前统计、处理后统计、异常值信息
    """
    # 步骤1：统计每个样本的行数
    sample_row_count = df_group.groupby("Sample ID").size().reset_index(name="row_count")
    
    # ========== 处理前统计 ==========
    pre_stats = {
        "传感器类型": group_name,
        "处理前样本数": len(sample_row_count),
        "处理前平均行数": sample_row_count["row_count"].mean(),
        "处理前最大行数": sample_row_count["row_count"].max(),
        "处理前最小行数": sample_row_count["row_count"].min(),
        "处理前Q1(行数)": sample_row_count["row_count"].quantile(0.25),
        "处理前Q3(行数)": sample_row_count["row_count"].quantile(0.75),
        "处理前IQR(行数)": sample_row_count["row_count"].quantile(0.75) - sample_row_count["row_count"].quantile(0.25)
    }
    
    # ========== 计算四分位并剔除异常值（IQR法） ==========
    Q1 = pre_stats["处理前Q1(行数)"]
    Q3 = pre_stats["处理前Q3(行数)"]
    IQR = pre_stats["处理前IQR(行数)"]
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR  # 同时过滤下限异常值（行数过少）
    
    # 筛选正常样本（行数在上下限内）
    normal_samples = sample_row_count[
        (sample_row_count["row_count"] >= lower_bound) & 
        (sample_row_count["row_count"] <= upper_bound)
    ]["Sample ID"].tolist()
    
    # 过滤异常值：保留正常样本的所有行数据
    filtered_group = df_group[df_group["Sample ID"].isin(normal_samples)]
    
    # ========== 异常值信息 ==========
    outlier_sample_count = len(sample_row_count) - len(normal_samples)
    outlier_info = {
        "传感器类型": group_name,
        "行数下限": lower_bound,
        "行数上限": upper_bound,
        "异常样本数量": outlier_sample_count,
        "异常样本占比": f"{outlier_sample_count/len(sample_row_count)*100:.2f}%" if len(sample_row_count) > 0 else "0.00%"
    }
    
    # ========== 处理后统计 ==========
    post_sample_row_count = filtered_group.groupby("Sample ID").size().reset_index(name="row_count") if len(filtered_group) > 0 else pd.DataFrame(columns=["Sample ID", "row_count"])
    
    post_stats = {
        "传感器类型": group_name,
        "处理后样本数": len(post_sample_row_count),
        "处理后平均行数": post_sample_row_count["row_count"].mean() if len(post_sample_row_count) > 0 else 0,
        "处理后最大行数": post_sample_row_count["row_count"].max() if len(post_sample_row_count) > 0 else 0,
        "处理后最小行数": post_sample_row_count["row_count"].min() if len(post_sample_row_count) > 0 else 0,
        "处理后Q1(行数)": post_sample_row_count["row_count"].quantile(0.25) if len(post_sample_row_count) > 0 else 0,
        "处理后Q3(行数)": post_sample_row_count["row_count"].quantile(0.75) if len(post_sample_row_count) > 0 else 0,
        "处理后IQR(行数)": (post_sample_row_count["row_count"].quantile(0.75) - post_sample_row_count["row_count"].quantile(0.25)) if len(post_sample_row_count) > 0 else 0
    }
    
    return filtered_group, pre_stats, post_stats, outlier_info

# 3. 主处理流程
try:
    # 读取CSV文件（保留关键列：Sample ID、SensorType）
    df = pd.read_csv(file_path, usecols=["Sample ID", "SensorType"])
    
    # 校验SensorType是否包含指定三类值
    valid_sensor_types = {'Gravity', 'Accelerometer', 'Gyroscope'}
    df = df[df["SensorType"].isin(valid_sensor_types)]
    if len(df) == 0:
        raise ValueError("数据中无Gravity/Accelerometer/Gyroscope类型的传感器数据")
    
    # 初始化存储所有结果的列表
    all_pre_stats = []    # 所有传感器处理前的统计
    all_post_stats = []   # 所有传感器处理后的统计
    all_outlier_info = [] # 所有传感器的异常值信息
    all_filtered_data = []# 所有传感器处理后的数据
    
    # 步骤2：按 SensorType 分组处理每个传感器的异常值
    grouped = df.groupby("SensorType")
    total_groups = len(grouped)
    print("=" * 100)
    print(f"开始处理 {total_groups} 个传感器分组的异常值（按SensorType分组）")
    print("=" * 100)
    
    for sensor_type, group_df in grouped:
        group_name = sensor_type
        # 处理当前传感器的异常值（按样本行数）
        filtered_df, pre_stats, post_stats, outlier_info = process_sensor_outliers(group_df, group_name)
        
        # 存储结果
        all_pre_stats.append(pre_stats)
        all_post_stats.append(post_stats)
        all_outlier_info.append(outlier_info)
        all_filtered_data.append(filtered_df)
        
        # 输出当前传感器的对比结果（简洁版）
        print(f"\n【传感器类型：{group_name}】")
        print(f"  处理前：样本数={pre_stats['处理前样本数']}，平均行数={pre_stats['处理前平均行数']:.2f}，最大行数={pre_stats['处理前最大行数']}")
        print(f"  异常值：异常样本数={outlier_info['异常样本数量']}，占比={outlier_info['异常样本占比']}，行数范围=[{outlier_info['行数下限']:.2f}, {outlier_info['行数上限']:.2f}]")
        print(f"  处理后：样本数={post_stats['处理后样本数']}，平均行数={post_stats['处理后平均行数']:.2f}，最大行数={post_stats['处理后最大行数']}")
    
    # 步骤3：合并所有处理后的数据，生成全局统计
    combined_filtered = pd.concat(all_filtered_data, ignore_index=True)
    pre_combined_sample_count = len(df["Sample ID"].unique())
    pre_combined_avg_rows = df.groupby("Sample ID").size().mean()
    post_combined_sample_count = len(combined_filtered["Sample ID"].unique()) if len(combined_filtered) > 0 else 0
    post_combined_avg_rows = combined_filtered.groupby("Sample ID").size().mean() if len(combined_filtered) > 0 else 0
    
    # 步骤4：输出全局对比结果
    print("\n" + "=" * 100)
    print("【全局统计对比】")
    print(f"  处理前：总样本数={pre_combined_sample_count}，所有样本平均行数={pre_combined_avg_rows:.2f}")
    print(f"  处理后：总样本数={post_combined_sample_count}，所有样本平均行数={post_combined_avg_rows:.2f}")
    print("=" * 100)
    
    # 步骤5：保存详细结果到CSV（便于论文整理）
    # 转换为DataFrame
    pre_stats_df = pd.DataFrame(all_pre_stats)
    post_stats_df = pd.DataFrame(all_post_stats)
    outlier_info_df = pd.DataFrame(all_outlier_info)
    
    # 保存文件
    pre_stats_df.to_csv(os.path.join(directory, "传感器处理前统计.csv"), index=False, encoding="utf-8")
    post_stats_df.to_csv(os.path.join(directory, "传感器处理后统计.csv"), index=False, encoding="utf-8")
    outlier_info_df.to_csv(os.path.join(directory, "传感器异常值信息.csv"), index=False, encoding="utf-8")
    combined_filtered.to_csv(os.path.join(directory, "传感器处理后所有样本数据.csv"), index=False, encoding="utf-8")
    
    print(f"\n详细统计结果已保存至 {directory} 目录下：")
    print("  - 传感器处理前统计.csv")
    print("  - 传感器处理后统计.csv")
    print("  - 传感器异常值信息.csv")
    print("  - 传感器处理后所有样本数据.csv")

except FileNotFoundError:
    print(f"错误：未找到文件 {csv_file}，请检查文件路径是否正确。")
except KeyError as e:
    print(f"错误：文件 {csv_file} 中不存在 '{e.args[0]}' 列，请检查列名是否正确（区分大小写）。")
except ValueError as e:
    print(f"数据校验错误：{str(e)}")
except Exception as e:
    print(f"处理文件 {csv_file} 时发生未知错误：{str(e)}")

# datas
# ====================================================================================================
# 开始处理 3 个传感器分组的异常值（按SensorType分组）
# ====================================================================================================

# 【传感器类型：Accelerometer】
#   处理前：样本数=32194，平均行数=9.55，最大行数=399
#   异常值：异常样本数=2389，占比=7.42%，行数范围=[3.00, 11.00]
#   处理后：样本数=29805，平均行数=6.82，最大行数=11

# 【传感器类型：Gravity】
#   处理前：样本数=32194，平均行数=72.63，最大行数=596
#   异常值：异常样本数=1640，占比=5.09%，行数范围=[21.00, 117.00]
#   处理后：样本数=30554，平均行数=68.46，最大行数=117

# 【传感器类型：Gyroscope】
#   处理前：样本数=32194，平均行数=9.55，最大行数=399
#   异常值：异常样本数=2389，占比=7.42%，行数范围=[3.00, 11.00]
#   处理后：样本数=29805，平均行数=6.83，最大行数=11

# ====================================================================================================
# 【全局统计对比】
#   处理前：总样本数=32194，所有样本平均行数=91.73
#   处理后：总样本数=30593，所有样本平均行数=81.67
# ====================================================================================================

# Tdatas
# ====================================================================================================

# 【传感器类型：Accelerometer】
#   处理前：样本数=25794，平均行数=7.07，最大行数=226
#   异常值：异常样本数=835，占比=3.24%，行数范围=[2.00, 10.00]
#   处理后：样本数=24959，平均行数=6.27，最大行数=10

# 【传感器类型：Gravity】
#   处理前：样本数=25794，平均行数=64.47，最大行数=421
#   异常值：异常样本数=777，占比=3.01%，行数范围=[22.00, 102.00]
#   处理后：样本数=25017，平均行数=62.56，最大行数=102

# 【传感器类型：Gyroscope】
#   处理前：样本数=25794，平均行数=7.07，最大行数=226
#   异常值：异常样本数=838，占比=3.25%，行数范围=[2.00, 10.00]
#   处理后：样本数=24956，平均行数=6.27，最大行数=10

# ====================================================================================================
# 【全局统计对比】
#   处理前：总样本数=25794，所有样本平均行数=78.61
#   处理后：总样本数=25124，所有样本平均行数=74.74