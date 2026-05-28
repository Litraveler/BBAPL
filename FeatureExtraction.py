import os
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. =====================
# 熵值评估：特征提取

# ===================== 1. 配置路径和参数 =====================
WEIGHTS_DIR = r"E:\【论文撰写】\投稿\StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices\Codes\modelDis_Split"
SEQUENCE_LENGTH = 150
INPUT_FEATURES = 17

# 数据文件路径（从 Data_Split.py 获取）
DATA_FILES = [
    r"E:\【论文撰写】\投稿\StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices\Codes\datas\padding_data.csv",
    r"E:\【论文撰写】\投稿\StrokePL：Enhancing Pattern Lock Authentication with Behavioral Biometrics for Mobile Devices\Codes\Tdatas\padding_data.csv"
]

# 输出文件路径
OUTPUT_CSV = "extracted_features.csv"

# 特征列定义
FEATURE_COLS = [
    'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size',
    'GravityX', 'GravityY', 'GravityZ',
    'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
    'AccelerometerX', 'AccelerometerY', 'AccelerometerZ'
]

# ===================== 2. 加载训练好的模型 =====================
def load_trained_model():
    """加载训练好的模型结构并载入权重"""
    from Models import SplitArchitecture
    descriptor = {}
    descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH
    descriptor["INPUT_FEATURES"] = INPUT_FEATURES
    descriptor = SplitArchitecture.get_model(descriptor)
    model = descriptor["model"]
    path = "./modelDis_Split/final_model_weights"
    model.load_weights(path)
    return model

# ===================== 3. 读取和预处理数据 =====================
def load_and_preprocess_data(file_paths):
    """读取多个数据文件并合并"""
    all_dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"警告：文件不存在，跳过: {file_path}")
            continue
        df = pd.read_csv(file_path)
        print(f"读取文件: {file_path}, 数据形状: {df.shape}")
        all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("未找到任何数据文件")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"合并后数据形状: {combined_df.shape}")
    
    # 检查必要列
    required_cols = ['UUID', 'Sample ID', 'Posture', 'pattern'] + FEATURE_COLS
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要列: {missing_cols}")
    
    return combined_df

# ===================== 4. 提取单个样本的特征向量 =====================
def get_sample_embedding(model, sample_data):
    """使用模型提取单个样本的特征向量"""
    sample_input = np.expand_dims(sample_data, axis=0)
    embedding = model.predict(sample_input, verbose=0)[0]
    return np.array(embedding)

# ===================== 5. 主流程：提取所有样本的特征向量 =====================
def extract_all_features(model, df):
    """提取所有样本的特征向量"""
    results = []
    
    # 按用户、样本分组处理
    grouped = df.groupby(['UUID', 'Sample ID', 'Posture', 'pattern'])
    
    total_groups = len(grouped)
    processed_count = 0
    
    for (uuid, sample_id, posture, pattern), group_df in grouped:
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"正在处理样本 {processed_count}/{total_groups}")
        
        # 按Time排序
        group_df = group_df.sort_values('Time').reset_index(drop=True)
        
        # 提取特征数据
        seq_data = group_df[FEATURE_COLS].values
        
        # 截断或填充到固定长度
        if len(seq_data) < SEQUENCE_LENGTH:
            pad_len = SEQUENCE_LENGTH - len(seq_data)
            seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
        else:
            seq_data = seq_data[:SEQUENCE_LENGTH]
        
        # 提取特征向量
        embedding = get_sample_embedding(model, seq_data)
        
        # 保存结果
        results.append({
            '用户ID': uuid,
            '样本ID': sample_id,
            'Posture': posture,
            'Pattern': pattern,
            '特征向量': embedding
        })
    
    return results

# ===================== 6. 保存特征向量到CSV =====================
def save_features_to_csv(results, output_path):
    """将特征向量保存到CSV文件"""
    # 将特征向量转换为字符串以便保存
    for res in results:
        res['特征向量'] = ','.join(map(str, res['特征向量']))
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 调整列顺序
    df = df[['用户ID', '样本ID', 'Posture', 'Pattern', '特征向量']]
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n特征向量已保存至: {output_path}")
    print(f"共提取 {len(df)} 个样本的特征向量")

# ===================== 7. 主函数 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("特征提取程序启动")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        print("\n1. 正在读取数据文件...")
        df = load_and_preprocess_data(DATA_FILES)
        
        # 2. 加载模型
        print("\n2. 正在加载训练好的模型...")
        model = load_trained_model()
        print("模型加载成功")
        
        # 3. 提取特征向量
        print("\n3. 正在提取特征向量...")
        features = extract_all_features(model, df)
        
        # 4. 保存结果
        print("\n4. 正在保存特征向量...")
        save_features_to_csv(features, OUTPUT_CSV)
        
        print("\n" + "=" * 60)
        print("特征提取完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()