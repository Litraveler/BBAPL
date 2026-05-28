import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# ===================== 2. =====================
# 熵值评估：中心向量提取
# ===================== 1. 配置路径和参数 =====================
# 输入特征文件路径（来自 FeatureExtraction.py 的输出）
INPUT_FEATURES_FILE = "extracted_features.csv"

# 输出中心向量文件路径
OUTPUT_CENTROIDS_FILE = "centroids_features.csv"

# ===================== 2. 读取提取的特征数据 =====================
def load_extracted_features(file_path):
    """读取之前提取的特征向量CSV文件"""
    if not os.path.exists(file_path):
        raise ValueError(f"特征文件不存在: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"读取特征文件: {file_path}, 数据形状: {df.shape}")
    
    # 检查必要列
    required_cols = ['用户ID', '样本ID', 'Posture', 'Pattern', '特征向量']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要列: {missing_cols}")
    
    # 将特征向量字符串转换为numpy数组
    df['特征向量'] = df['特征向量'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    
    return df

# ===================== 3. 提取用户中心向量 =====================
def extract_user_centroids(df):
    """
    根据posture和pattern分组，提取每个用户的中心向量（平均值）
    """
    results = []
    
    # 按Posture和Pattern分组
    posture_pattern_groups = df.groupby(['Posture', 'Pattern'])
    
    for (posture, pattern), group_df in posture_pattern_groups:
        print(f"\n处理分组: Posture={posture}, Pattern={pattern}")
        
        # 在每组内按用户ID分组
        user_groups = group_df.groupby('用户ID')
        
        for user_id, user_df in user_groups:
            # 获取该用户的所有特征向量
            embeddings = np.array(user_df['特征向量'].tolist())
            
            # 计算中心向量（平均值）
            centroid = np.mean(embeddings, axis=0)
            
            results.append({
                '用户ID': user_id,
                'Posture': posture,
                'Pattern': pattern,
                '中心向量': centroid
            })
        
        print(f"  该分组共 {len(user_groups)} 个用户")
    
    return results

# ===================== 4. 保存中心向量到CSV =====================
def save_centroids_to_csv(results, output_path):
    """将中心向量保存到CSV文件"""
    # 将中心向量转换为字符串以便保存
    for res in results:
        res['中心向量'] = ','.join(map(str, res['中心向量']))
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 调整列顺序
    df = df[['用户ID', 'Posture', 'Pattern', '中心向量']]
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n中心向量已保存至: {output_path}")
    print(f"共提取 {len(df)} 个用户的中心向量")

# ===================== 5. 主函数 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("中心向量提取程序启动")
    print("=" * 60)
    
    try:
        # 1. 加载特征数据
        print("\n1. 正在读取特征文件...")
        features_df = load_extracted_features(INPUT_FEATURES_FILE)
        
        # 2. 提取中心向量
        print("\n2. 正在提取用户中心向量...")
        centroids = extract_user_centroids(features_df)
        
        # 3. 保存结果
        print("\n3. 正在保存中心向量...")
        save_centroids_to_csv(centroids, OUTPUT_CENTROIDS_FILE)
        
        print("\n" + "=" * 60)
        print("中心向量提取完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()