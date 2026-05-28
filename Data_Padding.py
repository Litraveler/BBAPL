import pandas as pd
import numpy as np
# ====================================================================================================
# Step: 9 
# ====================================================================================================
file_paths = {
    "merged_touch_with_sensor.csv": "Tdatas/merged_touch_with_sensor.csv"
}

# 定义目标长度
target_lengths = {
    "merged_touch_with_sensor.csv": 150,
}

# 定义需要扩充的列
columns_to_pad = ['Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size', 'GravityIs', 'GravityX', 'GravityY', 'GravityZ', 'GyroscopeIs', 'GyroscopeX', 'GyroscopeY', 'GyroscopeZ', 'AccelerometerIs','AccelerometerX','AccelerometerY', 'AccelerometerZ']

# 定义处理函数
def process_files(file, target_length):
    # 读取文件
    df = pd.read_csv(file)

    # # ========== 核心修改1：添加TimePeriod列并全部置为1 ==========
    # df['TimePeriod'] = 1

    sample_ids = df['Sample ID'].unique()

    # 初始化结果列表
    result_data = []

    # 遍历每个样本
    for sample_id in sample_ids:
        group = df[df['Sample ID'] == sample_id]

        # 扩充touch数据
        sample_length = len(group)
        if sample_length < target_length:
            touch_pad_data = {col: [0] * (target_length - sample_length) for col in columns_to_pad}
            touch_pad_data['Sample ID'] = [sample_id] * (target_length - sample_length)
            touch_pad_data['pattern'] = [group['pattern'].iloc[0]] * (target_length - sample_length)
            touch_pad_data['UUID'] = [group['UUID'].iloc[0]] * (target_length - sample_length)
            touch_pad_data['Posture'] = [group['Posture'].iloc[0]] * (target_length - sample_length)
            touch_pad_data['TimePeriod'] = [group['TimePeriod'].iloc[0]] * (target_length - sample_length)
            pad_df = pd.DataFrame(touch_pad_data)
            touch_group = pd.concat([group.assign(is_original=1), pad_df], ignore_index=True)

        # 保存扩充后的样本
        result_data.append(touch_group)

    # 合并所有样本
    touch_result_df = pd.concat(result_data, ignore_index=True)

    # 保存到新的文件
    touch_output_file = f"Tdatas/padding_{file.split('/')[-1]}"
    touch_result_df.to_csv(touch_output_file, index=False)
    print(f"处理完成，结果已保存到 {touch_output_file}")


# 处理sit和walk数据
process_files(file_paths["merged_touch_with_sensor.csv"], target_lengths["merged_touch_with_sensor.csv"])