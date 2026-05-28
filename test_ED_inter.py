import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# ===================== 1. 配置路径和参数 =====================
WEIGHTS_DIR = r"E:\【论文撰写】\投稿\StrokePL\Codes\modelDis"
TEST_DATA_PATH = r"E:\【论文撰写】\投稿\StrokePL\Codes\TrainTestData\test_data.csv"
SEQUENCE_LENGTH = 150
INPUT_FEATURES = 17

# ===================== 【核心修改】固定单个 TimePeriod =====================
# 你可以手动修改这个值：4 / 3 / 2 / 1 等
FIXED_TIME_PERIOD = 1

# TSNE配置
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_FIG_SIZE = (16, 12)

# 聚类配置
CLUSTER_NUM = 3
CLUSTER_RANDOM_STATE = 42

# ===================== 2. 加载并预处理测试数据 =====================
def load_and_preprocess_data(file_path):
    """
    【已修改】加载数据，仅使用 FIXED_TIME_PERIOD 时段
    模板：该时段内 第一个样本
    正测试样本：该时段内 除第一个外的所有剩余样本
    """
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    if "TimePeriod" not in df.columns:
        raise ValueError("数据缺失TimePeriod列，请检查test_data.csv")
    
    # 定义特征列
    feature_cols = [
        'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size'
        ,'GravityX', 'GravityY', 'GravityZ',
        'GyroscopeX', 'GyroscopeY', 'GyroscopeZ',
        'AccelerometerX', 'AccelerometerY', 'AccelerometerZ'
    ]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失特征列: {missing_cols}")
    
    # 按Posture + pattern分层级分组
    groups = {}
    for posture_name, posture_df in df.groupby('Posture'):
        groups[posture_name] = {}
        for pattern_name, pattern_df in posture_df.groupby('pattern'):
            user_samples = {}
            # 遍历每个用户
            for uuid, user_df in pattern_df.groupby('UUID'):
                # 只保留固定的 TimePeriod
                target_df = user_df[user_df['TimePeriod'] == FIXED_TIME_PERIOD].copy()
                if len(target_df) < 2:
                    # 至少需要1个模板 + 1个测试样本
                    print(f"用户{uuid}在TimePeriod={FIXED_TIME_PERIOD}样本不足，跳过")
                    continue
                
                # 按Sample ID分组并排序
                sample_list = []
                for sample_id, sample_df in target_df.groupby('Sample ID'):
                    sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                    seq_data = sample_df[feature_cols].values
                    # 统一长度
                    if len(seq_data) < SEQUENCE_LENGTH:
                        pad_len = SEQUENCE_LENGTH - len(seq_data)
                        seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
                    else:
                        seq_data = seq_data[:SEQUENCE_LENGTH]
                    sample_list.append(seq_data)
                
                # 【核心逻辑】第一个样本=模板，其余=正测试样本
                template_sample = sample_list[0]
                test_pos_samples = sample_list[1:]
                
                user_samples[uuid] = {
                    "template_sample": template_sample,       # 单个模板样本
                    "test_positive_samples": test_pos_samples  # 同一用户正样本
                }
            
            if user_samples:
                groups[posture_name][pattern_name] = user_samples
    
    return groups, feature_cols

def load_trained_model():
    """从 ./model/ 加载训练好的模型"""
    from Models import SplitArchitecture
    descriptor = {}
    descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH
    descriptor["INPUT_FEATURES"] = INPUT_FEATURES
    descriptor = SplitArchitecture.get_model(descriptor)
    m = descriptor["model"]
    path = "./modelDis_Split/final_model_weights"
    m.load_weights(path)
    return m

# ===================== 3. 核心函数：特征向量提取和欧几里得距离计算 =====================
def get_sample_embedding(model, sample_data):
    """提取单个样本特征向量"""
    sample_input = np.expand_dims(sample_data, axis=0)
    embedding = model.predict(sample_input, verbose=0)[0]
    return np.array(embedding)

def get_template_centers(model, template_sample):
    """
    【已修改】单个模板样本，直接作为聚类中心
    兼容原有 KMeans 逻辑
    """
    emb = get_sample_embedding(model, template_sample)
    # 单个样本扩展为3个中心（保持聚类逻辑一致）
    return np.repeat([emb], CLUSTER_NUM, axis=0)

def calculate_min_euclidean_distance(test_emb, template_centers):
    """计算最小欧氏距离"""
    test_emb = test_emb.reshape(1, -1)
    distances = euclidean_distances(test_emb, template_centers)[0]
    return np.min(distances)

def calculate_eer(distances, labels):
    """计算EER"""
    if not distances or len(distances) != len(labels):
        return 1.0, 0.0
    
    thresholds = np.linspace(0, 1, 100)
    min_eer = 1.0
    best_threshold = 0.0
    
    for th in thresholds:
        preds = [1 if d < th else 0 for d in distances]
        fp = fn = tn = tp = 0
        
        for label, pred in zip(labels, preds):
            if label == 1:
                tp += 1 if pred == 1 else 0
                fn += 1 if pred == 0 else 0
            else:
                fp += 1 if pred == 1 else 0
                tn += 1 if pred == 0 else 0
        
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        current_eer_diff = abs(FAR - FRR)
        
        if current_eer_diff < min_eer:
            min_eer = (FAR + FRR) / 2
            best_threshold = th
    
    return min_eer, best_threshold

# ===================== 4. 按用户计算EER =====================
def calculate_session_level_eer(model, posture_pattern_data):
    """【已修改】基于固定TimePeriod计算EER"""
    session_eer_list = []
    all_users = list(posture_pattern_data.keys())
    
    # 提取所有用户特征
    user_emb_info = {}
    for user_id in all_users:
        data = posture_pattern_data[user_id]
        # 模板中心
        template_centers = get_template_centers(model, data["template_sample"])
        # 正样本特征
        pos_embs = [get_sample_embedding(model, s) for s in data["test_positive_samples"]]
        # 所有样本（用于负样本）
        all_embs = [get_sample_embedding(model, data["template_sample"])] + pos_embs
        
        user_emb_info[user_id] = {
            "template_centers": template_centers,
            "pos_embs": pos_embs,
            "all_embs": all_embs
        }
    
    # 计算每个用户 EER
    for target_user in all_users:
        info = user_emb_info[target_user]
        centers = info["template_centers"]
        pos_embs = info["pos_embs"]
        
        if not pos_embs:
            continue
        
        # 正样本距离
        pos_dist = [(calculate_min_euclidean_distance(emb, centers), 1) for emb in pos_embs]
        
        # 负样本距离（其他用户所有样本）
        neg_dist = []
        for other in all_users:
            if other == target_user:
                continue
            for emb in user_emb_info[other]["all_embs"]:
                neg_dist.append((calculate_min_euclidean_distance(emb, centers), 0))
        
        # 正负样本均衡
        try:
            neg_dist = random.sample(neg_dist, len(pos_dist))
        except ValueError:
            neg_dist = neg_dist[:len(pos_dist)]
        
        # 计算 EER
        all_pairs = pos_dist + neg_dist
        distances = [d for d, _ in all_pairs]
        labels = [l for _, l in all_pairs]
        eer, _ = calculate_eer(distances, labels)
        
        session_eer_list.append({
            "用户ID": target_user,
            "TimePeriod": FIXED_TIME_PERIOD,
            "EER": round(eer, 4)
        })
    
    return session_eer_list

# ===================== 5. 提取特征 & TSNE 绘图 =====================
def extract_user_embeddings(model, posture_pattern_data):
    user_embeddings = {}
    all_embeddings = []
    all_user_labels = []
    
    for user_id, data in posture_pattern_data.items():
        embs = []
        # 模板
        emb = get_sample_embedding(model, data["template_sample"])
        embs.append(emb)
        all_embeddings.append(emb)
        all_user_labels.append(user_id)
        # 测试正样本
        for s in data["test_positive_samples"]:
            emb = get_sample_embedding(model, s)
            embs.append(emb)
            all_embeddings.append(emb)
            all_user_labels.append(user_id)
        user_embeddings[user_id] = np.array(embs)
    
    return user_embeddings, np.array(all_embeddings), np.array(all_user_labels)

def plot_tsne_for_walk_twoone(model, groups):
    if "walk" not in groups or "two_one" not in groups["walk"]:
        raise ValueError("未找到walk_two-one分组数据")
    
    target_data = groups["walk"]["two_one"]
    _, embeddings, user_labels = extract_user_embeddings(model, target_data)
    
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=TSNE_FIG_SIZE)
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=range(len(user_labels)), cmap='tab10', s=150, alpha=0.9)
    plt.ylabel('t-SNE StrokePL', fontsize=32)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("tsne_walk_twoone.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("TSNE 绘图完成")
    return tsne_results, user_labels

# ===================== 主流程 =====================
if __name__ == "__main__":
    print(f"当前使用固定 TimePeriod = {FIXED_TIME_PERIOD}")
    groups, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
    model = load_trained_model()
    
    all_session_results = []
    for posture_name, pattern_dict in groups.items():
        for pattern_name, user_data in pattern_dict.items():
            group_name = f"{posture_name}_{pattern_name}"
            print(f"\n===== 处理分组: {group_name} =====")
            
            try:
                session_eer_list = calculate_session_level_eer(model, user_data)
                for res in session_eer_list:
                    res["Posture"] = posture_name
                    res["Pattern"] = pattern_name
                all_session_results.extend(session_eer_list)
                
                for res in session_eer_list:
                    print(f"用户{res['用户ID']} EER: {res['EER']}")
                
                # 统计
                df = pd.DataFrame(session_eer_list)
                mean_eer = df['EER'].mean().round(4)
                std_eer = df['EER'].std().round(4)
                print(f"分组平均EER: {mean_eer} ± {std_eer}")
                
            except Exception as e:
                print(f"处理失败: {e}")
                continue
    
    # 绘图
    try:
        plot_tsne_for_walk_twoone(model, groups)
    except:
        print("TSNE 绘图失败")
    
    # 保存结果
    final_df = pd.DataFrame(all_session_results)
    final_df = final_df[["Posture", "Pattern", "用户ID", "TimePeriod", "EER"]]
    final_df.to_csv(f"single_timeperiod_eer_results_FIX_{FIXED_TIME_PERIOD}.csv", index=False, encoding="utf-8-sig")
    print(f"\n全部结果已保存，共 {len(final_df)} 条记录")