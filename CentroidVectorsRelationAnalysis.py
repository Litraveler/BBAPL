import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 配置路径和参数 =====================
# 输入文件路径
INPUT_FEATURES_FILE = "extracted_features.csv"
INPUT_CENTROIDS_FILE = "centroids_features.csv"

# 输出结果文件前缀
OUTPUT_PREFIX = "centroid_relation_analysis"

# 角距离阈值列表（0.01到0.1，共10个值）
THRESHOLDS = np.linspace(0.01, 0.1, 10).tolist()

# ===================== 2. 角距离计算 =====================
def angular_distance_2tensors(feature1, feature2):
    """计算两个特征张量之间的角距离"""
    feature1 = tf.math.l2_normalize(feature1, axis=1)
    feature2 = tf.math.l2_normalize(feature2, axis=1)
    angular_distances = 1 - tf.matmul(feature1, feature2, transpose_b=True)
    return angular_distances

# ===================== 3. 读取数据 =====================
def load_features(file_path):
    """读取特征向量CSV文件"""
    if not os.path.exists(file_path):
        raise ValueError(f"文件不存在: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"读取文件: {file_path}, 数据形状: {df.shape}")
    
    # 将特征向量字符串转换为numpy数组
    if '特征向量' in df.columns:
        df['特征向量'] = df['特征向量'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    elif '中心向量' in df.columns:
        df['中心向量'] = df['中心向量'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    
    return df

# ===================== 4. 计算匹配概率 =====================
def compute_matching_probability(user_centroids, user_samples, thr):
    """
    计算每个用户的样本与中心向量的匹配概率
    :param user_centroids: 包含各用户中心向量的DataFrame
    :param user_samples: 包含所有用户样本的DataFrame
    :param thr: 角距离阈值
    :return: 各用户的匹配概率（Series）
    """
    # 构建中心向量字典
    centroids_dict = {}
    for _, row in user_centroids.iterrows():
        key = (row['用户ID'], row['Posture'], row['Pattern'])
        centroids_dict[key] = row['中心向量']
    
    probabilities = {}
    
    # 按用户ID、Posture、Pattern分组处理
    grouped = user_samples.groupby(['用户ID', 'Posture', 'Pattern'])
    
    for (user_id, posture, pattern), group_df in grouped:
        # 获取对应的中心向量
        key = (user_id, posture, pattern)
        if key not in centroids_dict:
            probabilities[f"{user_id}_{posture}_{pattern}"] = 0.0
            continue
        
        centroid = centroids_dict[key].reshape(1, -1)
        
        # 获取该用户该分组下的所有样本特征向量
        samples = np.array(group_df['特征向量'].tolist())
        
        # 转换为Tensor并计算角距离
        samples_tensor = tf.convert_to_tensor(samples, dtype=tf.float32)
        centroid_tensor = tf.convert_to_tensor(centroid, dtype=tf.float32)
        
        distances = angular_distance_2tensors(samples_tensor, centroid_tensor)
        distances_np = distances.numpy().flatten()
        
        # 统计匹配比例
        count = np.sum(distances_np < thr)
        total = len(distances_np)
        probabilities[f"{user_id}_{posture}_{pattern}"] = count / total if total > 0 else 0.0
    
    return pd.Series(probabilities)

# ===================== 5. 主分析函数 =====================
def analyze_centroid_relation():
    """主分析流程（仅整体分析）"""
    print("=" * 70)
    print("中心向量与样本关系分析（整体）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 正在加载数据...")
    samples_df = load_features(INPUT_FEATURES_FILE)
    centroids_df = load_features(INPUT_CENTROIDS_FILE)
    
    print(f"\n2. 样本总数: {len(samples_df)}, 中心向量总数: {len(centroids_df)}")
    
    # 2. 整体分析（不分组）
    print("\n3. 整体分析（所有分组合并）...")
    for thr in THRESHOLDS:
        matching_probs = compute_matching_probability(centroids_df, samples_df, thr)
        
        print(f"\n  阈值 thr={thr:.3f}:")
        print(f"    匹配概率统计:")
        print(f"      均值: {matching_probs.mean():.4f}")
        print(f"      标准差: {matching_probs.std():.4f}")
        print(f"      最小值: {matching_probs.min():.4f}")
        print(f"      最大值: {matching_probs.max():.4f}")
        print(f"      中位数: {matching_probs.median():.4f}")
        
        # 保存整体结果
        output_file = f"{OUTPUT_PREFIX}_all_thr{thr:.3f}.csv"
        matching_probs.to_csv(output_file, header=['匹配概率'], encoding='utf-8-sig')
        print(f"    结果已保存至: {output_file}")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

# ===================== 6. 主函数 =====================
if __name__ == "__main__":
    try:
        analyze_centroid_relation()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()