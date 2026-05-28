import pandas as pd
import os
import glob
import uuid
# ====================================================================================================
# Step: 1 
# ====================================================================================================
# 配置路径（修改为新数据集路径）
data_dir = r"E:\代码库 数据库\混合数据集\4个时间段数据集"
os.makedirs(data_dir, exist_ok=True)

postures = ["sit", "walk"]

def find_file_name(user_path, pattern):
    """查找匹配的文件路径"""
    file_pattern = os.path.join(user_path, pattern)
    matching_files = glob.glob(file_pattern)
    
    # 如果没有找到匹配的文件，尝试使用用户文件夹名称作为前缀进行匹配
    if not matching_files:
        user_folder = os.path.basename(user_path)
        alternative_pattern = os.path.join(user_path, f"{user_folder}_*{pattern.strip('*')}")
        matching_files = glob.glob(alternative_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"未找到匹配的文件: {file_pattern}")
    
    return matching_files[0]

# 处理每个用户：获取顶级用户文件夹（19个用户）
user_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, f))]

save_file_path = "Tdatas"
os.makedirs(save_file_path, exist_ok=True)
output_sensor_file_name = f"sensor_data.csv"
output_touch_file_name = f"touch_data.csv"
output_sensor_path = os.path.join(save_file_path, output_sensor_file_name)
output_touch_path = os.path.join(save_file_path, output_touch_file_name)

# 为不同姿势分别初始化DataFrame
filtered_sensor_data = pd.DataFrame()

filtered_touch_data = pd.DataFrame()


# 为每个用户生成唯一UUID（一个用户对应一个UUID）
user_id_map = {}
for user_folder in user_folders:
    while True:
        user_id = str(uuid.uuid4())
        if user_id not in user_id_map.values():
            user_id_map[user_folder] = user_id
            break
def extract_type_label(action_type):
    """从ACTION_TYPE中提取Type和Label"""
    if action_type.startswith('Down_') or action_type.startswith('Up_'):
        parts = action_type.split('_')
        if len(parts) >= 3:
            type_val = parts[1]
            label_val = parts[2]
            return f"{type_val}_{label_val}"
    return None

def split_touch_samples(touch_data):
    """将触摸数据分割为完整的样本（Down-Move-Up）"""
    samples = []
    current_sample = None
    current_pattern = None
    
    for idx, row in touch_data.iterrows():
        action_type = row['ACTION_TYPE']
        # 跳过无效的action
        if not isinstance(action_type, str) or action_type == '':
            continue
        
        # 检测Down事件，开始新样本
        if action_type.startswith('Down_'):
            # 如果已有未完成的样本，丢弃
            if current_sample is not None:
                print(f"发现不完整的触摸样本，已丢弃")
            
            current_pattern = extract_type_label(action_type)
            current_sample = {
                'start_time': row['Time'],
                'end_time': None,
                'pattern': current_pattern,
                'data': [row]
            }
        
        # 检测Move事件，添加到当前样本
        elif action_type.startswith('Move_') and current_sample is not None:
            current_sample['data'].append(row)
        
        # 检测Up事件，结束当前样本
        elif action_type.startswith('Up_') and current_sample is not None:
            # 验证Up事件的Type_Label是否与Down事件匹配
            if action_type.endswith("true"):
                current_sample['end_time'] = row['Time']
                current_sample['data'].append(row)
                samples.append(current_sample)
                current_sample = None
                current_pattern = None
            else:
                print(f"Up事件与Down事件的Type_Label不匹配，丢弃样本")
                current_sample = None
                current_pattern = None
    
    return samples

for user_folder in user_folders:
    user_id = user_id_map[user_folder]
    print(f"正在处理用户: {user_folder}，UUID: {user_id}")
    
    # 获取用户文件夹下的所有时间段文件夹，并按时间排序（假设文件夹名按日期从小到大排序）
    time_folders = [f for f in os.listdir(user_folder) 
                   if os.path.isdir(os.path.join(user_folder, f))]
    # 按文件夹名称排序（确保时间从早到晚）
    time_folders_sorted = sorted(time_folders)
    print(time_folders_sorted)
    
    # 遍历每个时间段文件夹（1-4）
    for time_period, time_folder in enumerate(time_folders_sorted, start=1):
        time_folder_path = os.path.join(user_folder, time_folder)
        print(f"  处理时间段 {time_period}：{time_folder_path}")
        
        # 检查该时间段文件夹下是否有CSV文件
        if not any(file.endswith('.csv') for file in os.listdir(time_folder_path)):
            print(f"  时间段 {time_period} 无CSV文件，跳过")
            continue
        
        for posture in postures:
            # 查找触摸数据和传感器数据文件
            try:
                touch_file_path = find_file_name(time_folder_path, f'*_pattern_lock_{posture}_touchData*')
                sensor_file_path = find_file_name(time_folder_path, f'*_pattern_lock_{posture}_sensorData*')
                
                # 读取数据
                touch_data = pd.read_csv(touch_file_path)
                sensor_data = pd.read_csv(sensor_file_path)
                
                # 检查必要的列是否存在
                required_touch_cols = ['ACTION_TYPE', 'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size']
                required_sensor_cols = ['Time', 'SensorType', 'X', 'Y', 'Z']
                
                if not all(col in touch_data.columns for col in required_touch_cols):
                    print(f"触摸数据缺少必要列，跳过")
                    continue
                
                if not all(col in sensor_data.columns for col in required_sensor_cols):
                    print(f"传感器数据缺少必要列，跳过")
                    continue
                
            except FileNotFoundError as e:
                print(f"用户 {time_folder_path} {posture} 姿势的文件未找到: {e}")
                continue
            except Exception as e:
                print(f"读取用户 {time_folder_path} {posture} 姿势的数据时出错: {e}")
                continue
            
            # 分割触摸样本
            touch_samples = split_touch_samples(touch_data)
            print(f"找到 {len(touch_samples)} 个完整的{posture}姿势触摸样本")
            
            # 处理每个触摸样本
            for sample in touch_samples:
                # 生成唯一样本ID
                sample_id = str(uuid.uuid4())
                pattern = sample['pattern']
                start_time = sample['start_time']
                end_time = sample['end_time']
                
                # 处理触摸数据
                sample_touch_df = pd.DataFrame(sample['data'])
                sample_touch_df['Posture'] = posture
                sample_touch_df['pattern'] = pattern
                sample_touch_df['Sample ID'] = sample_id
                sample_touch_df['UUID'] = user_id
                sample_touch_df["TimePeriod"] = time_period
                
                # 合并到总触摸数据
                filtered_touch_data = pd.concat([filtered_touch_data, sample_touch_df], ignore_index=True)
                
                # 处理传感器数据 - 扩展时间范围（前后各250ms，单位根据数据调整）
                # 注意：这里的时间单位需要根据你的实际数据调整，250000000是纳秒，如数据单位是毫秒则改为250
                sensor_start = start_time - 250000000
                sensor_end = end_time + 250000000
                
                # 筛选传感器数据
                valid_sensor_types = ['Gravity', 'Gyroscope', 'Accelerometer']
                sensor_mask = (sensor_data['Time'] >= sensor_start) & (sensor_data['Time'] <= sensor_end)
                period_sensor_data = sensor_data[sensor_mask].copy()
                
                # 筛选有效传感器类型
                period_sensor_data = period_sensor_data[period_sensor_data['SensorType'].isin(valid_sensor_types)]
                
                # 添加元数据
                period_sensor_data['posture'] = posture
                period_sensor_data['pattern'] = pattern
                period_sensor_data['Sample ID'] = sample_id
                period_sensor_data['UUID'] = user_id
                period_sensor_data["TimePeriod"] = time_period
                
                # 合并到总传感器数据
                filtered_sensor_data = pd.concat([filtered_sensor_data, period_sensor_data], ignore_index=True)

filtered_sensor_data.to_csv(output_sensor_path, index=False)
filtered_touch_data.to_csv(output_touch_path, index=False)

