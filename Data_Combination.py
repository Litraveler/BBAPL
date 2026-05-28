import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# ====================================================================================================
# Step: 7 
# ====================================================================================================
# ===================== 配置项 =====================
# 文件路径
DATA_DIR = Path(r"E:\论文撰写记录\投稿\StrokePL\Codes\datas")
SENSOR_FILE = DATA_DIR / "normalized_sensor_data.csv"
TOUCH_FILE = DATA_DIR / "normalized_touch_data.csv"
OUTPUT_FILE = DATA_DIR / "merged_touch_with_sensor.csv"

# 需要匹配的传感器类型
SENSOR_TYPES = ['Gravity', 'Gyroscope', 'Accelerometer']
# ==================================================

def load_and_preprocess_data():
    """加载并预处理数据：类型转换、空值处理"""
    print("正在读取数据文件...")
    # 读取数据
    sensor_df = pd.read_csv(SENSOR_FILE)
    touch_df = pd.read_csv(TOUCH_FILE)

    return sensor_df, touch_df

def build_sensor_index(sensor_df):
    """构建传感器数据索引：按Sample ID+SensorType分组，提高匹配效率"""
    print("正在构建传感器数据索引...")
    sensor_index = {}
    # 按样本ID和传感器类型双层分组
    for sample_id in sensor_df['Sample ID'].unique():
        sample_data = sensor_df[sensor_df['Sample ID'] == sample_id]
        sensor_index[sample_id] = {}
        for sensor_type in SENSOR_TYPES:
            type_data = sample_data[sample_data['SensorType'] == sensor_type].sort_values('Time')
            sensor_index[sample_id][sensor_type] = type_data
    return sensor_index

def find_nearest_touch_row(sensor_time, touch_data):
    """
    核心函数：为单条传感器数据找最近的触摸数据行
    返回：匹配到的触摸数据的全局索引（不是相对行号）
    """
    if touch_data.empty:
        return None
    
    # 计算时间差绝对值，找最小值对应的触摸行
    time_diff = np.abs(touch_data['Time'].values - sensor_time)
    min_pos = np.argmin(time_diff)  # 子集内的位置
    
    # 返回该位置对应的全局索引（关键修正）
    return touch_data.index[min_pos]

def merge_data(touch_df, sensor_index):
    """核心合并逻辑：为每一行触摸数据匹配各类传感器数据"""
    print("开始合并数据（触摸数据为基准，添加传感器数据）...")
    # 复制触摸数据作为结果基础
    result_df = touch_df.copy()
    
    # 初始化所有传感器相关列（先填充NaN，比0更合理）
    for sensor_type in SENSOR_TYPES:
        result_df[f"{sensor_type}Is"] = 0
        result_df[f"{sensor_type}X"] = 0
        result_df[f"{sensor_type}Y"] = 0
        result_df[f"{sensor_type}Z"] = 0
    
    # 按样本ID分组处理，保证同样本内匹配
    total_samples = len(touch_df['Sample ID'].unique())
    processed = 0
    
    for sample_id in touch_df['Sample ID'].unique():
        processed += 1
        if processed % 100 == 0:
            print(f"进度：{processed}/{total_samples} 个样本已处理")
        
        # 获取当前样本的触摸数据（带全局索引）和传感器数据
        touch_sample = touch_df[touch_df['Sample ID'] == sample_id]
        sensor_sample = sensor_index.get(sample_id, {})
        
        if touch_sample.empty or not sensor_sample:
            continue
        
        # 遍历每种传感器类型
        for sensor_type in SENSOR_TYPES:
            # 获取当前样本的该类型传感器数据
            sensor_type_data = sensor_sample.get(sensor_type, pd.DataFrame())
            if sensor_type_data.empty:
                continue
            
            # 遍历每条传感器数据，匹配最近的触摸数据
            for _, sensor_row in sensor_type_data.iterrows():
                sensor_time = sensor_row['Time']
                
                # 找到匹配的触摸数据全局索引（关键修正）
                touch_global_idx = find_nearest_touch_row(sensor_time, touch_sample)
                
                if touch_global_idx is not None:
                    # 使用全局索引赋值，确保位置正确
                    result_df.loc[touch_global_idx, f"{sensor_type}Is"] = 1
                    result_df.loc[touch_global_idx, f"{sensor_type}X"] = sensor_row['X']
                    result_df.loc[touch_global_idx, f"{sensor_type}Y"] = sensor_row['Y']
                    result_df.loc[touch_global_idx, f"{sensor_type}Z"] = sensor_row['Z']
    
    return result_df

def main():
    """主函数：执行完整流程"""
    # 1. 加载预处理数据
    sensor_df, touch_df = load_and_preprocess_data()
    
    # 2. 构建传感器数据索引
    sensor_index = build_sensor_index(sensor_df)
    
    # 3. 合并数据
    merged_df = merge_data(touch_df, sensor_index)
    
    # 4. 保存结果
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 输出统计信息
    print("\n=== 合并完成 ===")
    print(f"结果文件路径：{OUTPUT_FILE}")
    print(f"原始触摸数据行数：{len(touch_df)}")
    print(f"合并后数据行数：{len(merged_df)}")
    print(f"新增传感器列数：{len(merged_df.columns) - len(touch_df.columns)}")
    
    # 统计各传感器匹配情况
    for sensor_type in SENSOR_TYPES:
        matched_count = merged_df[f"{sensor_type}Is"].sum()
        total_count = len(merged_df)
        print(f"{sensor_type} 匹配率：{matched_count}/{total_count} ({matched_count/total_count:.2%})")

if __name__ == "__main__":
    main()
# datas
# 原始触摸数据行数：1687809
# 合并后数据行数：1687809
# 新增传感器列数：12
# Gravity 匹配率：1541412/1687809 (91.33%)
# Gyroscope 匹配率：209562/1687809 (12.42%)
# Accelerometer 匹配率：209604/1687809 (12.42%)

# Tdatas
# 结果文件路径：E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\merged_touch_with_sensor.csv
# 原始触摸数据行数：1129237
# 合并后数据行数：1129237
# 新增传感器列数：12
# Gravity 匹配率：1105905/1129237 (97.93%)
# Gyroscope 匹配率：146506/1129237 (12.97%)
# Accelerometer 匹配率：146476/1129237 (12.97%)