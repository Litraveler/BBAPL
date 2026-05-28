import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from compute_guessing_lists import compute_coo_arr, compute_guessing_list
import guessing_metrics as em
from spherical_codes import compute_far_ifo_dimensions, convert_to_angle, upper_bound

# ===================== 配置参数 =====================
# 输入中心向量文件路径
CENTROIDS_FILE = "../centroids_features.csv"

# 参数设置
rounds = ["0"]
thresholds = 0.05

# 输出数据结构
export = {
    "round": [], 
    "posture": [], 
    "pattern": [], 
    "num_users": [],
    "beta_entropy_3": [], 
    "beta_entropy_6": [], 
    "beta_entropy_10": [],
    "alfa_guess_work_entropy_5": [], 
    "alfa_guess_work_entropy_2": [], 
    "upper_bound": []
}

# ===================== 读取中心向量数据 =====================
def load_centroids_from_csv(file_path):
    """从CSV文件读取中心向量数据"""
    if not os.path.exists(file_path):
        raise ValueError(f"中心向量文件不存在: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"读取中心向量文件: {file_path}, 数据形状: {df.shape}")
    
    # 解析中心向量（将字符串转换为numpy数组）
    df['中心向量'] = df['中心向量'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    
    return df

# ===================== 主分析函数 =====================
def main():
    # 加载中心向量数据
    centroids_df = load_centroids_from_csv(CENTROIDS_FILE)
    
    # 获取所有唯一的Posture和Pattern组合
    unique_combinations = centroids_df[['Posture', 'Pattern']].drop_duplicates().values
    
    for round in rounds:
        for posture, pattern in unique_combinations:
            print(f"\n=== 处理: round={round}, posture={posture}, pattern={pattern} ===")
            
            # 筛选该Posture和Pattern的中心向量
            group_df = centroids_df[(centroids_df['Posture'] == posture) & (centroids_df['Pattern'] == pattern)]
            
            if len(group_df) == 0:
                print(f"  该分组无数据，跳过")
                continue
            
            # 提取中心向量和用户标签
            all_centroids = np.array(group_df['中心向量'].tolist())
            all_centroid_labels = group_df['用户ID'].values
            
            # 使用LabelEncoder转换用户标签
            all_centroid_labels = LabelEncoder().fit_transform(all_centroid_labels)
            num_enrolled_users = len(np.unique(all_centroid_labels))
            print(f"  注册用户数: {num_enrolled_users}")
            print(f"  中心向量总数: {all_centroids.shape[0]}")
            
            if num_enrolled_users < 2:
                print(f"  用户数不足，跳过")
                continue
            
            # 分割训练集和测试集 (2/3 训练, 1/3 测试)
            n = all_centroids.shape[0]
            split_idx = int(n * 2 / 3)
            
            if split_idx < 1 or (n - split_idx) < 1:
                print(f"  样本数不足，无法分割，跳过")
                continue
            
            centroids = all_centroids[:split_idx]
            query_templates = all_centroids[split_idx:]
            
            print(f"  训练中心向量数: {centroids.shape[0]}")
            print(f"  测试模板数: {query_templates.shape[0]}")
            
            # 计算COO数组和CSR数组
            thr = thresholds
            coo_arr = compute_coo_arr(centroids, query_templates, thr)
            csr_arr = coo_arr.tocsr()
            
            # 计算猜测列表
            dictionary = compute_guessing_list(csr_arr)
            dictionary = [x / num_enrolled_users for x in dictionary]
            
            # 计算各种熵指标
            print('\n  beta success rate 3, 6: ')
            beta_success_rate_3 = em.beta_success_rate(3, dictionary)
            print(f"    beta_success_rate_3: {beta_success_rate_3}")
            beta_success_rate_6 = em.beta_success_rate(6, dictionary)
            print(f"    beta_success_rate_6: {beta_success_rate_6}")
            
            print('\n  beta entropy 3, 6, 10:')
            beta_entropy_3 = em.beta_entropy(3, dictionary)
            print(f"    beta_entropy_3: {beta_entropy_3}")
            beta_entropy_6 = em.beta_entropy(6, dictionary)
            print(f"    beta_entropy_6: {beta_entropy_6}")
            beta_entropy_10 = em.beta_entropy(10, dictionary)
            print(f"    beta_entropy_10: {beta_entropy_10}")
            
            print('\n  alfa guess work 50% and 20%:')
            alfa_guess_work_entropy_5 = em.alfa_guess_work(0.5, dictionary)
            print(f"    alfa_guess_work_50%: {alfa_guess_work_entropy_5}")
            alfa_guess_work_entropy_2 = em.alfa_guess_work(0.2, dictionary)
            print(f"    alfa_guess_work_20%: {alfa_guess_work_entropy_2}")
            
            # 计算上界（如果需要）
            upper_bounds = upper_bound(centroids, query_templates, thr) if 'upper_bound' in dir() else 0
            
            # 保存结果到导出字典
            export["round"].append(round)
            export["posture"].append(posture)
            export["pattern"].append(pattern)
            export["num_users"].append(num_enrolled_users)
            export["beta_entropy_3"].append(beta_entropy_3)
            export["beta_entropy_6"].append(beta_entropy_6)
            export["beta_entropy_10"].append(beta_entropy_10)
            export["alfa_guess_work_entropy_2"].append(alfa_guess_work_entropy_2)
            export["alfa_guess_work_entropy_5"].append(alfa_guess_work_entropy_5)
            export["upper_bound"].append(upper_bounds)
    
    # 保存结果到CSV
    output_dir = 'E7_results_entropy'
    os.makedirs(output_dir, exist_ok=True)
    results = os.path.join(output_dir, f'entropy.csv')
    df = pd.DataFrame(export)
    df.to_csv(results, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {results}")
    print(f"共 {len(df)} 条记录")

if __name__ == "__main__":
    main()