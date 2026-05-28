import os
import pandas as pd
from pathlib import Path
import sys
import csv  # 添加csv模块处理引号

#！！！！！！！！！！！！！！！！！！！根据用户样本计算中心点！！！！！！！！！！！！！！！！！！！
# 添加父目录到系统路径
current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir))

import re
import numpy as np

def parse_feature_vector(s):
    """安全解析特征向量字符串为 NumPy 数组"""
    if not isinstance(s, str):
        print("输入不是字符串")
        return None

    # 去掉首尾空白和可能的引号
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # 用正则提取所有浮点数（支持科学计数法）
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if not numbers:
        print("未提取到任何数字")
        return None

    try:
        vec = np.array(numbers, dtype=np.float32)
        return vec
    except ValueError as e:
        print("转换浮点数失败:", e)
        return None

def main():
    # 设置特征文件目录
    saved_features_dir = "../E7_results"
    # 确保目录存在
    if not os.path.exists(saved_features_dir):
        print(f"错误：目录 '{saved_features_dir}' 不存在")
        return
    # 获取所有特征文件
    feature_files = [f for f in os.listdir(saved_features_dir)]
    if not feature_files:
        print(f"警告：在 '{saved_features_dir}' 中没有找到特征文件")
        return
    # 处理每个特征文件
    for i, feature_file in enumerate(feature_files):
        file_path = os.path.join(saved_features_dir, feature_file)
        # 读取CSV文件 - 添加引号处理
        try:
            df = pd.read_csv(file_path, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue

        # 确保数据类型正确
        df['Posture'] = df['Posture'].astype(str)
        df['PIN'] = df['PIN'].astype(str)
        df['User'] = df['User'].astype(str)
        # 按Posture和PIN分组
        for posture, posture_group in df.groupby('Posture'):
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
            self_distance = []  # 存储当前用户自身样本到质心的距离
            other_distance = []  # 存储其他用户样本到当前用户质心的距离
            for pin, pin_group in posture_group.groupby('PIN'):
                # 先收集所有用户的数据
                all_users_data = {}
                for uuid, user_group in pin_group.groupby('User'):
                    # 收集特征向量
                    feature_vectors = []
                    for _, row in user_group.iterrows():
                        vec = parse_feature_vector(row['feature_vector'])
                        if vec is not None:
                            feature_vectors.append(vec)

                    if feature_vectors:
                        all_users_data[uuid] = np.array(feature_vectors)

                # 对每个用户计算质心并计算距离
                for uuid, user_vectors in all_users_data.items():
                    # 计算当前用户的质心
                    centroid = user_vectors.mean(axis=0)
                    centroid_unit = centroid / np.linalg.norm(centroid)  # 单位化

                    # 计算当前用户自身样本到质心的距离
                    projections = user_vectors @ centroid_unit
                    orthogonal_components = user_vectors - projections[:, np.newaxis] * centroid_unit
                    perpendicular_distances = np.linalg.norm(orthogonal_components, axis=1)
                    self_distance.extend(perpendicular_distances)

                    # 计算其他用户样本到当前用户质心的距离
                    for other_uuid, other_vectors in all_users_data.items():
                        if other_uuid == uuid:
                            continue  # 跳过当前用户自身

                        # 计算其他用户向量在当前用户质心单位向量上的投影
                        other_projections = other_vectors @ centroid_unit
                        other_orthogonal = other_vectors - other_projections[:, np.newaxis] * centroid_unit
                        other_perpendicular = np.linalg.norm(other_orthogonal, axis=1)
                        other_distance.extend(other_perpendicular)
            print(f"\n姿势: {posture}")
            print("自身样本到质心的距离统计:")
            for threshold in thresholds:
                count = sum(1 for d in self_distance if d < threshold)
                print(f"阈值 {threshold}: {count / len(self_distance):.4f} ({count}/{len(self_distance)})")

            print("\n其他用户样本到质心的距离统计:")
            for threshold in thresholds:
                count = sum(1 for d in other_distance if d < threshold)
                print(f"阈值 {threshold}: {count / len(other_distance):.4f} ({count}/{len(other_distance)})")

if __name__ == "__main__":
    main()