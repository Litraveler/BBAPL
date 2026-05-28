import pandas as pd
import os
# ====================================================================================================
# Step: 4 
# ====================================================================================================
# Tdatas 处理结果
# 第一步：读取有效Sample ID列表...
# 从'处理后所有样本数据.csv'读取有效Sample ID数量：24872
# 从'传感器处理后所有样本数据.csv'读取有效Sample ID数量：25124
# 合并后最终有效Sample ID数量（并集）：24471

# 第二步：处理sensor_data.csv...
# sensor_data.csv 原始行数：2027710
# 过滤后行数：1858506
# 清洗后的sensor数据已保存至：E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\cleaned_sensor_data.csv

# 第三步：处理touch_data.csv...
# 已删除touch_data.csv中的'ACTION_TYPE'列
# touch_data.csv 原始行数：1251432
# 过滤后行数：1129237
# 清洗后的touch数据已保存至：E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\cleaned_touch_data.csv

# ============================================================
# 数据清洗完成！最终统计：
# - 处理后所有样本数据.csv 有效ID数：24872
# - 传感器处理后所有样本数据.csv 有效ID数：25124
# - 最终有效Sample ID数量（并集）：24471
# - 清洗后sensor数据行数：1858506
# - 清洗后touch数据行数：1129237
# - 输出文件路径：
#   - E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\cleaned_sensor_data.csv
#   - E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\cleaned_touch_data.csv
# ====================================================================================================
# datas 处理结果
# 第一步：读取有效Sample ID列表...
# 从'处理后所有样本数据.csv'读取有效Sample ID数量：30918
# 从'传感器处理后所有样本数据.csv'读取有效Sample ID数量：30593
# 合并后最终有效Sample ID数量（并集）：29698

# 第二步：处理sensor_data.csv...
# sensor_data.csv 原始行数：2953090
# 过滤后行数：2542567
# 清洗后的sensor数据已保存至：E:\论文撰写记录\投稿\StrokePL\Codes\datas\cleaned_sensor_data.csv

# 第三步：处理touch_data.csv...
# 已删除touch_data.csv中的'ACTION_TYPE'列
# touch_data.csv 原始行数：1839192
# 过滤后行数：1553515
# 清洗后的touch数据已保存至：E:\论文撰写记录\投稿\StrokePL\Codes\datas\cleaned_touch_data.csv

# ============================================================
# 数据清洗完成！最终统计：
# - 处理后所有样本数据.csv 有效ID数：30918
# - 传感器处理后所有样本数据.csv 有效ID数：30593
# - 最终有效Sample ID数量（并集）：29698
# - 清洗后sensor数据行数：2542567
# - 清洗后touch数据行数：1553515
# ====================================================================================================

# 1. 定义文件路径
directory = r"E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas"
# 定义各文件路径
valid_sample_file = os.path.join(directory, "处理后所有样本数据.csv")
sensor_valid_sample_file = os.path.join(directory, "传感器处理后所有样本数据.csv")  
sensor_file = os.path.join(directory, "sensor_data.csv")
touch_file = os.path.join(directory, "touch_data.csv")
# 定义输出文件路径
cleaned_sensor_file = os.path.join(directory, "cleaned_sensor_data.csv")
cleaned_touch_file = os.path.join(directory, "cleaned_touch_data.csv")

try:
    # 2. 读取有效Sample ID列表（合并两个文件，取并集）
    print("第一步：读取有效Sample ID列表...")
    # 读取第一个有效样本文件
    valid_samples_df = pd.read_csv(valid_sample_file, usecols=["Sample ID"])
    valid_sample_ids_1 = set(valid_samples_df["Sample ID"].unique())
    print(f"从'处理后所有样本数据.csv'读取有效Sample ID数量：{len(valid_sample_ids_1)}")
    
    # 读取第二个传感器有效样本文件
    sensor_valid_samples_df = pd.read_csv(sensor_valid_sample_file, usecols=["Sample ID"])
    valid_sample_ids_2 = set(sensor_valid_samples_df["Sample ID"].unique())
    print(f"从'传感器处理后所有样本数据.csv'读取有效Sample ID数量：{len(valid_sample_ids_2)}")
    
    # 合并两个集合（取并集，只要在任意一个文件中存在即有效）
    valid_sample_ids = valid_sample_ids_1.intersection(valid_sample_ids_2)
    print(f"合并后最终有效Sample ID数量（并集）：{len(valid_sample_ids)}")
    
    # 3. 处理sensor_data.csv：仅保留有效Sample ID的行
    print("\n第二步：处理sensor_data.csv...")
    sensor_df = pd.read_csv(sensor_file)
    
    # 检查是否存在Sample ID列
    if "Sample ID" not in sensor_df.columns:
        raise KeyError("sensor_data.csv 中未找到 'Sample ID' 列，请检查列名")
    
    # 过滤有效样本
    cleaned_sensor_df = sensor_df[sensor_df["Sample ID"].isin(valid_sample_ids)]
    print(f"sensor_data.csv 原始行数：{len(sensor_df)}")
    print(f"过滤后行数：{len(cleaned_sensor_df)}")
    
    # 保存清洗后的sensor数据
    cleaned_sensor_df.to_csv(cleaned_sensor_file, index=False, encoding="utf-8")
    print(f"清洗后的sensor数据已保存至：{cleaned_sensor_file}")
    
    # 4. 处理touch_data.csv：保留有效Sample ID + 删除ACTION_TYPE列
    print("\n第三步：处理touch_data.csv...")
    touch_df = pd.read_csv(touch_file)
    
    # 检查是否存在Sample ID列
    if "Sample ID" not in touch_df.columns:
        raise KeyError("touch_data.csv 中未找到 'Sample ID' 列，请检查列名")
    
    # 步骤1：过滤有效样本
    filtered_touch_df = touch_df[touch_df["Sample ID"].isin(valid_sample_ids)]
    # 步骤2：删除ACTION_TYPE列（如果存在）
    if "ACTION_TYPE" in filtered_touch_df.columns:
        cleaned_touch_df = filtered_touch_df.drop(columns=["ACTION_TYPE"])
        print("已删除touch_data.csv中的'ACTION_TYPE'列")
    else:
        cleaned_touch_df = filtered_touch_df
        print("touch_data.csv中未找到'ACTION_TYPE'列，无需删除")
    
    print(f"touch_data.csv 原始行数：{len(touch_df)}")
    print(f"过滤后行数：{len(cleaned_touch_df)}")
    
    # 保存清洗后的touch数据
    cleaned_touch_df.to_csv(cleaned_touch_file, index=False, encoding="utf-8")
    print(f"清洗后的touch数据已保存至：{cleaned_touch_file}")
    
    # 5. 输出最终统计信息
    print("\n" + "="*60)
    print("数据清洗完成！最终统计：")
    print(f"- 处理后所有样本数据.csv 有效ID数：{len(valid_sample_ids_1)}")
    print(f"- 传感器处理后所有样本数据.csv 有效ID数：{len(valid_sample_ids_2)}")
    print(f"- 最终有效Sample ID数量（并集）：{len(valid_sample_ids)}")
    print(f"- 清洗后sensor数据行数：{len(cleaned_sensor_df)}")
    print(f"- 清洗后touch数据行数：{len(cleaned_touch_df)}")
    print(f"- 输出文件路径：")
    print(f"  - {cleaned_sensor_file}")
    print(f"  - {cleaned_touch_file}")

except FileNotFoundError as e:
    print(f"错误：未找到文件 - {e.filename}，请检查文件是否存在")
except KeyError as e:
    print(f"错误：{str(e)}")
except Exception as e:
    print(f"处理过程中发生未知错误：{str(e)}")