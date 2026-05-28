import pandas as pd
import os
# ====================================================================================================
# Step: 8 
# ====================================================================================================
# 定义文件路径
file_path = r'E:\论文撰写记录\投稿\StrokePL\Codes\Tdatas\merged_touch_with_sensor.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：文件 {file_path} 不存在，请检查路径是否正确！")
else:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查 'Sample ID' 列是否存在
        if 'Sample ID' not in df.columns:
            print("错误：文件中不存在 'Sample ID' 列，请检查列名是否正确！")
        else:
            # 统计每个Sample ID的行数（样本长度）
            sample_lengths = df['Sample ID'].value_counts()
            
            # 计算最大值、最小值、均值
            max_length = sample_lengths.max()
            min_length = sample_lengths.min()
            mean_length = sample_lengths.mean()
            
            # 输出结果
            print("样本长度统计结果：")
            print(f"最大值：{max_length} 行")
            print(f"最小值：{min_length} 行")
            print(f"均值：{mean_length:.2f} 行")
            
            # 可选：查看每个样本的具体长度（前10个）
            print("\n前10个样本的长度：")
            print(sample_lengths.head(10))
            
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
# datas
# 最大值：144 行
# 最小值：17 行
# 均值：52.31 行
# Tdatas
# 样本长度统计结果：
# 样本长度统计结果：
# 最大值：107 行
# 最小值：13 行
# 均值：46.15 行