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
WEIGHTS_DIR = r"E:\【论文撰写】\投稿\StrokePL\Codes\modelDis_Split"
TEST_DATA_PATH = r"E:\【论文撰写】\投稿\StrokePL\Codes\TrainTestData\test_data.csv"
SEQUENCE_LENGTH = 150
INPUT_FEATURES = 17

# TSNE配置
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_FIG_SIZE = (16, 12)
TEM_SESSION = [1,2,3]
TEM_SESSION_NAME = "1-2-3"
TEST_SESSION = [4]
TEST_SESSION_NAME = "4"
# 聚类配置
CLUSTER_NUM = 4
CLUSTER_RANDOM_STATE = 42

# ===================== 2. 加载并预处理测试数据 =====================
def load_and_preprocess_data(file_path):
    """加载测试数据并按Posture+pattern分组、UUID（用户）整理"""
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
            for uuid, user_df in pattern_df.groupby('UUID'):
                user_time_periods = user_df['TimePeriod'].unique()
                if not any(t in user_time_periods for t in TEM_SESSION):
                    continue
                
                template_df = user_df[user_df['TimePeriod'].isin(TEM_SESSION)].copy()
                test_df = user_df[user_df['TimePeriod'].isin(TEST_SESSION)].copy()
                
                if len(template_df) == 0 or len(test_df) == 0:
                    print(f"用户{uuid}缺少模板样本(TimePeriod={TEM_SESSION})或测试样本(TimePeriod={TEST_SESSION})，跳过")
                    continue
                
                # 处理模板样本
                template_samples = {}
                for sample_id, sample_df in template_df.groupby('Sample ID'):
                    sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                    seq_data = sample_df[feature_cols].values
                    if len(seq_data) < SEQUENCE_LENGTH:
                        pad_len = SEQUENCE_LENGTH - len(seq_data)
                        seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
                    else:
                        seq_data = seq_data[:SEQUENCE_LENGTH]
                    template_samples[sample_id] = seq_data
                
                # 处理测试样本
                test_periods = {}
                for time_period in TEST_SESSION:
                    tp_df = test_df[test_df['TimePeriod'] == time_period].copy()
                    if len(tp_df) == 0:
                        continue
                    tp_samples = {}
                    for sample_id, sample_df in tp_df.groupby('Sample ID'):
                        sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                        seq_data = sample_df[feature_cols].values
                        if len(seq_data) < SEQUENCE_LENGTH:
                            pad_len = SEQUENCE_LENGTH - len(seq_data)
                            seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
                        else:
                            seq_data = seq_data[:SEQUENCE_LENGTH]
                        tp_samples[sample_id] = seq_data
                    if tp_samples:
                        test_periods[time_period] = tp_samples
                
                if test_periods and template_samples:
                    user_samples[uuid] = {
                        "template_samples": template_samples,
                        "test_periods": test_periods
                    }
            
            if user_samples:
                groups[posture_name][pattern_name] = user_samples
    
    return groups, feature_cols

def load_trained_model():
    """从 ./model/ 加载训练好的模型（已修复：传入K和beta参数）"""
    """加载模型结构并载入权重"""
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
    """使用模型提取单个样本的特征向量"""
    sample_input = np.expand_dims(sample_data, axis=0)
    embedding = model.predict(sample_input, verbose=0)[0]
    # # 检查向量
    # print("===============================")
    # print("Embedding shape:", embedding.shape)
    # print("Embedding values (first 256):", embedding[:256])
    return np.array(embedding)

def get_template_centers(model, template_samples):
    """通过KMeans聚类从模板样本中生成中心特征向量"""
    template_embs = []
    for _, sample_data in template_samples.items():
        emb = get_sample_embedding(model, sample_data)
        template_embs.append(emb)
    template_embs = np.array(template_embs)
    
    if len(template_embs) < CLUSTER_NUM:
        repeat_times = int(np.ceil(CLUSTER_NUM / len(template_embs)))
        template_embs = np.repeat(template_embs, repeat_times, axis=0)[:CLUSTER_NUM]
    
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=CLUSTER_RANDOM_STATE)
    kmeans.fit(template_embs)
    centers = kmeans.cluster_centers_
    
    return centers

def calculate_min_euclidean_distance(test_emb, template_centers):
    """
    计算测试样本到模板中心的最小欧几里得距离
    距离越小 = 越相似
    """
    test_emb = np.array(test_emb)
    template_centers = np.array(template_centers)
    
    test_emb = test_emb.reshape(1, -1)
    distances = euclidean_distances(test_emb, template_centers)[0]
    return np.min(distances)


def calculate_eer(distances, labels):
    """
    基于欧几里得距离计算等错误率EER
    距离越小 → 越可能是同一用户（正样本 label=1）
    """
    if not distances or len(distances) != len(labels):
        return 1.0, 0.0
    
    thresholds = np.linspace(0, 1, 100)
    min_eer = 1.0
    best_threshold = 0.0
    
    for th in thresholds:
        # 距离 < 阈值 → 判定为正样本
        preds = [1 if d < th else 0 for d in distances]
        
        fp = 0
        fn = 0
        tn = 0
        tp = 0
        
        for label, pred in zip(labels, preds):
            if label == 1:
                if pred == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    tn += 1
        
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        current_eer_diff = abs(FAR - FRR)
        if current_eer_diff < min_eer:
            min_eer = (FAR + FRR) / 2
            best_threshold = th
    
    return min_eer, best_threshold

# ===================== 4. 按用户+TimePeriod计算EER（欧几里得距离版） =====================
def calculate_session_level_eer(model, posture_pattern_data):
    """按用户+TimePeriod维度计算EER（使用欧几里得距离）"""
    session_eer_list = []
    all_users = list(posture_pattern_data.keys())
    
    # 第一步：为所有用户提取特征向量，并生成模板中心
    user_emb_info = {}
    for user_id in all_users:
        user_data = posture_pattern_data[user_id]
        
        template_centers = get_template_centers(model, user_data["template_samples"])
        
        test_period_embs = {}
        for time_period, tp_samples in user_data["test_periods"].items():
            tp_embs = []
            for _, sample_data in tp_samples.items():
                emb = get_sample_embedding(model, sample_data)
                tp_embs.append(emb)
            test_period_embs[time_period] = tp_embs
        
        all_user_embs = []
        all_user_embs.extend(template_centers)
        for tp_embs in test_period_embs.values():
            all_user_embs.extend(tp_embs)
        
        user_emb_info[user_id] = {
            "template_centers": template_centers,
            "test_period_embs": test_period_embs,
            "all_embs": all_user_embs
        }
    
    # 第二步：为每个用户的每个TimePeriod计算EER
    for target_user in all_users:
        target_info = user_emb_info[target_user]
        target_template_centers = target_info["template_centers"]
        
        for time_period, tp_embs in target_info["test_period_embs"].items():
            if not tp_embs:
                print(f"用户{target_user}时段{time_period}无样本，跳过")
                continue
            
            # 正样本：同一用户，距离越小越相似
            pos_distances = []
            for test_emb in tp_embs:
                dist = calculate_min_euclidean_distance(test_emb, target_template_centers)
                pos_distances.append((dist, 1))
            
            # 负样本：其他用户
            neg_distances = []
            for other_user in all_users:
                if other_user == target_user:
                    continue
                other_embs = user_emb_info[other_user]["all_embs"]
                for emb in other_embs:
                    try:
                        neg_dist = calculate_min_euclidean_distance(emb, target_template_centers)
                        neg_distances.append((neg_dist, 0))
                    except Exception as e:
                        print(f"计算用户{other_user}样本距离失败: {e}，跳过该样本")
                        continue
            
            if not neg_distances:
                print(f"用户{target_user}时段{time_period}无负样本对，跳过")
                continue
            
            try:
                neg_distances = random.sample(neg_distances, len(pos_distances))
            except ValueError:
                neg_distances = neg_distances[:len(pos_distances)]
            
            all_pairs = pos_distances + neg_distances
            distances = [d for d, _ in all_pairs]
            labels = [l for _, l in all_pairs]
            
            eer, _ = calculate_eer(distances, labels)
            
            session_eer_list.append({
                "用户ID": target_user,
                "测试时段ID": time_period,
                "EER": round(eer, 4)
            })
    
    return session_eer_list

# ===================== 5. 提取用户特征向量用于TSNE =====================
def extract_user_embeddings(model, posture_pattern_data):
    """提取每个用户的所有样本特征向量"""
    user_embeddings = {}
    all_embeddings = []
    all_user_labels = []
    sample_ids = []
    
    for user_id, user_data in posture_pattern_data.items():
        user_sample_embs = []
        
        for sample_id, sample_data in user_data["template_samples"].items():
            emb = get_sample_embedding(model, sample_data)
            user_sample_embs.append(emb)
            all_embeddings.append(emb)
            all_user_labels.append(user_id)
            sample_ids.append(f"{user_id}_template_TP1+2_{sample_id}")
        
        for time_period, tp_samples in user_data["test_periods"].items():
            for sample_id, sample_data in tp_samples.items():
                embedding = get_sample_embedding(model, sample_data)
                user_sample_embs.append(embedding)
                all_embeddings.append(embedding)
                all_user_labels.append(user_id)
                sample_ids.append(f"{user_id}_TP{time_period}_Sample{sample_id}")
        
        user_embeddings[user_id] = np.array(user_sample_embs)
    
    all_embeddings = np.array(all_embeddings)
    all_user_labels = np.array(all_user_labels)
    
    return user_embeddings, all_embeddings, all_user_labels, sample_ids



# ===================== 7. 主流程执行 =====================
if __name__ == "__main__":
    groups, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
    model = load_trained_model()
    
    all_session_results = []
    for posture_name, pattern_dict in groups.items():
        for pattern_name, user_data in pattern_dict.items():
            group_full_name = f"{posture_name}_{pattern_name}"
            print(f"\n===== 处理分组: {group_full_name} =====")
            
            try:
                session_eer_list = calculate_session_level_eer(model, user_data)
                for res in session_eer_list:
                    res["Posture"] = posture_name
                    res["Pattern"] = pattern_name
                    res["分组名称"] = group_full_name
                all_session_results.extend(session_eer_list)
                
                print(f"分组{group_full_name} 时段级EER结果:")
                for res in session_eer_list:
                    print(f"  用户{res['用户ID']} 时段{res['测试时段ID']} EER: {res['EER']}")
                
                group_df = pd.DataFrame(session_eer_list)
                period_stats = group_df.groupby('测试时段ID')['EER'].agg(['mean', 'std']).round(4)
                print(f"\n分组{group_full_name} 各时段EER统计:")
                for period, stats in period_stats.iterrows():
                    mean_eer = stats['mean']
                    std_eer = stats['std']
                    std_eer = std_eer if not np.isnan(std_eer) else 0.0
                    print(f"  时段{period}: 均值={mean_eer}, 标准差={std_eer}")
            except Exception as e:
                print(f"分组{group_full_name}处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    results_df = pd.DataFrame(all_session_results)
    results_df = results_df[["Posture", "Pattern", "分组名称", "用户ID", "测试时段ID", "EER"]]
    results_df.to_csv(f"timeperiod_level_eer_results_{TEM_SESSION_NAME}_to_{TEST_SESSION_NAME}.csv", index=False, encoding="utf-8-sig")
    print(f"\n时段级EER结果已保存至 timeperiod_level_eer_results_{TEM_SESSION_NAME}_to_{TEST_SESSION_NAME}.csv")
    print(f"共生成 {len(results_df)} 条时段级测试结果")