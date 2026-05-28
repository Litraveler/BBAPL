import os
import uuid
import numpy as np
import pandas as pd
import random

# 1. 定义路径和参数
data_files = [
    r"E:\【论文撰写】\投稿\StrokePL\Codes\datas\padding_data.csv",
    r"E:\【论文撰写】\投稿\StrokePL\Codes\Tdatas\padding_data.csv"
]
MIN_SAMPLE_NUM = 5  # 最小样本数阈值
RANDOM_SEED = 32    # 固定随机种子保证可复现
# 明确指定需要排除的列（仅用于训练集的npy生成）
EXCLUDE_COLS = ["Posture", "pattern", "Sample ID", "UUID", "TimePeriod", "GravityIs", "GyroscopeIs", "AccelerometerIs", "is_original"
                ,'GravityX', 'GravityY', 'GravityZ',
                'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
                'AccelerometerX', 'AccelerometerY', 'AccelerometerZ'
                ]
# 测试集CSV保存路径
TEST_CSV_PATH = "test_data.csv"

# 2. 读取单个文件并返回数据和该文件的唯一UUID
def load_single_file(file_path):
    """读取单个CSV文件，返回数据框和该文件的唯一UUID列表"""
    if not os.path.exists(file_path):
        raise ValueError(f"文件不存在：{file_path}")
    df = pd.read_csv(file_path)
    
    # 新增：处理is_original列空值，将空值填充为0
    if "is_original" in df.columns:
        df["is_original"] = df["is_original"].fillna(0)
        df["is_original"] = pd.to_numeric(df["is_original"], errors="coerce").fillna(0)
        print(f"已处理 {file_path} 中is_original列空值，填充为0的数量：{df['is_original'].isna().sum()}")
    else:
        print(f"警告：{file_path} 中未找到is_original列，跳过空值处理")
    
    # 检查必要列
    required_cols = ["UUID", "Sample ID", "Posture", "pattern"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"文件 {file_path} 缺少必要列：{missing_cols}")
    
    file_uuids = df["UUID"].unique()
    print(f"读取文件 {file_path}：数据行数={len(df)}，独立用户数={len(file_uuids)}")
    return df, file_uuids

# 3. 对第二个文件的UUID按1:1划分
def split_uuids(uuids, seed=42):
    """
    对UUID列表按9:1划分（训练:测试）
    :param uuids: UUID列表
    :param seed: 随机种子
    :return: (train_uuids, test_uuids)
    """
    random.seed(seed)
    shuffled_uuids = uuids.copy()
    random.shuffle(shuffled_uuids)
    
    total = len(shuffled_uuids)
    train_num = total * 7 // 10  # 70%
    test_num = total - train_num  # 20%
    
    train = shuffled_uuids[:train_num]
    test = shuffled_uuids[train_num:]
    
    return train, test

# 4. 主流程：逐个文件处理，收集划分结果
all_dfs = []
global_train_uuids = []
global_test_uuids = []
cross_session_train_uuids = []   # 新增：记录来自data_files[2]的训练用户

# 处理第一个文件（全部归为训练集）
first_file_path = data_files[0]
try:
    df1, file1_uuids = load_single_file(first_file_path)
    all_dfs.append(df1)
    global_train_uuids.extend(file1_uuids)
    print(f"\n文件 {first_file_path} 划分结果：")
    print(f"  全部用户({len(file1_uuids)}个)归为训练集")
except Exception as e:
    print(f"处理文件 {first_file_path} 失败：{str(e)}")

third_file_path = data_files[1]
try:
    df3, file3_uuids = load_single_file(third_file_path)
    all_dfs.append(df3)
    
    # 第二个文件按1:1划分
    train_u2, test_u2 = split_uuids(file3_uuids, RANDOM_SEED)
    
    # 记录来自第三个文件的训练用户（用于后续标记跨会话）
    cross_session_train_uuids = train_u2.copy()
    
    # 收集到全局列表
    global_train_uuids.extend(train_u2)
    global_test_uuids.extend(test_u2)
    
    print(f"\n文件 {third_file_path} 划分结果：")
    print(f"  训练用户：{len(train_u2)} | 测试用户：{len(test_u2)}")
    print(f"  比例：{len(train_u2)}:{len(test_u2)}")
except Exception as e:
    print(f"处理文件 {third_file_path} 失败：{str(e)}")

# 合并所有原始数据
merged_df = pd.concat(all_dfs, ignore_index=True)
if "is_original" in merged_df.columns:
    merged_df["is_original"] = merged_df["is_original"].fillna(0)
    merged_df["is_original"] = pd.to_numeric(merged_df["is_original"], errors="coerce").fillna(0)
    print(f"\n合并后数据中is_original列空值数量：{merged_df['is_original'].isna().sum()}")

# 去重（防止不同文件有重复UUID）
global_train_uuids = list(set(global_train_uuids))
global_test_uuids = list(set(global_test_uuids))

# 校验：确保训练/测试用户无交集
assert len(set(global_train_uuids) & set(global_test_uuids)) == 0, "训练/测试用户重复"

print(f"\n【全局划分汇总】")
print(f"总训练用户数：{len(global_train_uuids)}")
print(f"总测试用户数：{len(global_test_uuids)}")
print(f"全局比例：训练:{len(global_train_uuids)} | 测试:{len(global_test_uuids)}")

# 5. 处理训练集（生成npy），增加跨会话标记
def process_dataset(user_uuids, df, min_sample_num, exclude_cols, cross_session_set=None):
    """
    处理指定用户集的数据，返回符合条件的三层结构数据（用于训练集npy生成）
    结构：{posture: {pattern: {is_cross: {group_id: {sample_id: sample_array}}}}}
    :param user_uuids: 目标用户UUID列表
    :param df: 完整数据集
    :param min_sample_num: 最小样本数阈值
    :param exclude_cols: 需要排除的列列表
    :param cross_session_set: 跨会话用户集合（即来自data_files[2]的训练用户）
    :return: 字典格式
    """
    # 筛选目标用户数据
    user_df = df[df["UUID"].isin(user_uuids)].copy()
    if len(user_df) == 0:
        print("警告：该数据集无匹配的用户数据")
        return {}
    
    # 确定特征列（排除指定列后的所有列）
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    if len(feature_cols) == 0:
        raise ValueError("特征列为空！请检查排除列列表是否覆盖了所有列")
    print(f"\n特征列列表（共{len(feature_cols)}列）：{feature_cols}")
    
    # 按Posture、pattern分组处理
    result = {}
    for (posture, pattern), group_df in user_df.groupby(["Posture", "pattern"]):
        # 统计每个用户的样本数（按Sample ID去重）
        user_sample_count = group_df.groupby("UUID")["Sample ID"].nunique()
        # 筛选样本数≥5的用户
        qualified_uuids = user_sample_count[user_sample_count >= min_sample_num].index
        
        if len(qualified_uuids) == 0:
            continue
        
        # 初始化posture和pattern层
        if posture not in result:
            result[posture] = {}
        if pattern not in result[posture]:
            result[posture][pattern] = {}
        
        # 处理每个符合条件的用户
        for user_uuid in qualified_uuids:
            # 判断是否跨会话（1表示是，0表示否）
            is_cross = 1 if (cross_session_set is not None and user_uuid in cross_session_set) else 0
            
            # 初始化is_cross层
            if is_cross not in result[posture][pattern]:
                result[posture][pattern][is_cross] = {}
            
            # 获取该用户的数据子集
            user_group_df = group_df[group_df["UUID"] == user_uuid]
            sample_dict = {}
            for sample_id, sample_df in user_group_df.groupby("Sample ID"):
                sample_array = sample_df[feature_cols].values
                if sample_array.size == 0:
                    print(f"警告：用户{user_uuid}样本{sample_id}的特征数组为空，跳过")
                    continue
                sample_dict[sample_id] = sample_array
            
            # 生成唯一组ID并保存
            if sample_dict:
                group_id = str(uuid.uuid4())
                result[posture][pattern][is_cross][group_id] = sample_dict
    return result

# 处理训练集，传入跨会话用户集合
cross_session_set = set(cross_session_train_uuids)
train_data = process_dataset(global_train_uuids, merged_df, MIN_SAMPLE_NUM, EXCLUDE_COLS, cross_session_set)

print(f"\n【训练集筛选结果】")
print(f"训练集有效组数：{len(train_data)}")

# 6. 处理测试集（保存为CSV，保留所有列）—— 与原代码相同
def save_test_data_to_csv(test_uuids, df, save_path):
    test_df = df[df["UUID"].isin(test_uuids)].copy()
    if len(test_df) == 0:
        print("警告：测试集无匹配数据！")
        return
    test_df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"\n【测试集保存结果】")
    print(f"测试数据总行数：{len(test_df)}")
    print(f"测试数据列数：{len(test_df.columns)}（保留所有列）")
    print(f"测试数据已保存至：{os.path.abspath(save_path)}")
    loaded_test_df = pd.read_csv(save_path)
    if len(loaded_test_df) == len(test_df):
        print(f"✅ 测试CSV文件验证成功，数据量匹配")
    else:
        print(f"❌ 测试CSV文件验证失败，原数据量{len(test_df)}，读取量{len(loaded_test_df)}")

save_test_data_to_csv(global_test_uuids, merged_df, TEST_CSV_PATH)

# 7. 保存训练集的npy文件
def save_and_verify(data, file_name):
    np.save(file_name, data)
    loaded_data = np.load(file_name, allow_pickle=True).item()
    if len(loaded_data) == len(data):
        print(f"✅ {file_name} 保存成功，数据量匹配")
    else:
        print(f"❌ {file_name} 保存异常，原数据量{len(data)}，读取量{len(loaded_data)}")
    return loaded_data

print("\n【训练集npy保存】")
save_and_verify(train_data, "xt.npy")