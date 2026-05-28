import pandas as pd
import os
# ====================================================================================================
# Step: 5 
# ====================================================================================================
# datas
# === 所有样本时间间隔的统计结果 ===
# 均值(秒): 0.8543
# 最大值(秒): 1.8960
# 最小值(秒): 0.2485
# 中位数(秒): 0.7952
# 样本总数: 29698.0000
# =================================================================
# Tdatas
# === 所有样本时间间隔的统计结果 ===
# 均值(秒): 0.7446
# 最大值(秒): 1.6882
# 最小值(秒): 0.2153
# 中位数(秒): 0.6974
# 样本总数: 24471.0000

# 定义文件路径
file_path = r"E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\cleaned_touch_data.csv"

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件不存在：{file_path}")

# 读取CSV文件
df = pd.read_csv(file_path)

# 验证必要的列是否存在
required_columns = ["Sample ID", "Time"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"数据中缺少必要的列：{missing_cols}")

# 1. 按Sample ID分组，计算每个样本的时间间隔（最大时间 - 最小时间）
sample_time_stats = df.groupby("Sample ID")["Time"].agg(
    touch_duration_ns=lambda x: x.max() - x.min()
).reset_index()

# 2. 将纳秒转换为秒
sample_time_stats["touch_duration_s"] = sample_time_stats["touch_duration_ns"] / 10**9

# 3. 计算所有样本时间间隔的统计指标（秒为单位）
duration_stats = {
    "均值(秒)": sample_time_stats["touch_duration_s"].mean(),
    "最大值(秒)": sample_time_stats["touch_duration_s"].max(),
    "最小值(秒)": sample_time_stats["touch_duration_s"].min(),
    "中位数(秒)": sample_time_stats["touch_duration_s"].median(),  # 额外补充中位数，更全面
    "样本总数": len(sample_time_stats)
}

# 打印结果
print("=== 每个样本的时间间隔（秒） ===")
print(sample_time_stats[["Sample ID", "touch_duration_s"]].head(10))  # 展示前10个样本
print("\n=== 所有样本时间间隔的统计结果 ===")
for key, value in duration_stats.items():
    print(f"{key}: {value:.4f}")

# 可选：将结果保存为CSV文件
output_path = r"E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\sample_time_duration.csv"
sample_time_stats.to_csv(output_path, index=False)
print(f"\n每个样本的时间间隔已保存至：{output_path}")