import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# ===================== 1. 配置路径和参数 =====================
WEIGHTS_DIR = r"E:\论文撰写记录\投稿\StrokePL\Codes\model"
TEST_DATA_PATH = r"E:\论文撰写记录\投稿\StrokePL\Codes\datasets\test_data.csv"
SEQUENCE_LENGTH = 150
INPUT_FEATURES = 17

# TSNE配置
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_FIG_SIZE = (16, 12)

# 聚类配置
CLUSTER_NUM = 2
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
        'Time', 'X', 'Y', 'SizeMajor', 'SizeMinor', 'Orientation', 'Pressure', 'Size',
        'GravityX', 'GravityY', 'GravityZ',
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
                if not any(t in user_time_periods for t in [4]):
                    continue
                
                template_df = user_df[user_df['TimePeriod'].isin([1,2])].copy()
                test_df = user_df[user_df['TimePeriod'].isin([3])].copy()
                
                if len(template_df) == 0 or len(test_df) == 0:
                    print(f"用户{uuid}缺少模板样本(TimePeriod=1+2)或测试样本(3/4)，跳过")
                    continue
                
                # 处理模板样本
                template_samples = {}
                for sample_id, sample_df in template_df.groupby('Sample ID'):
                    sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                    seq_data = sample_df[feature_cols].values
                    template_samples[sample_id] = seq_data
                
                # 处理测试样本
                test_periods = {}
                for time_period in [3]:
                    tp_df = test_df[test_df['TimePeriod'] == time_period].copy()
                    if len(tp_df) == 0:
                        continue
                    tp_samples = {}
                    for sample_id, sample_df in tp_df.groupby('Sample ID'):
                        sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                        seq_data = sample_df[feature_cols].values
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
    """加载模型结构并载入权重"""
    from Models import SplitArchitecture
    descriptor = {}
    descriptor["SEQUENCE_LENGTH"] = SEQUENCE_LENGTH
    descriptor["INPUT_FEATURES"] = INPUT_FEATURES
    descriptor = SplitArchitecture.get_model(descriptor)
    m = descriptor["model"]
    path = "./model2/final_model_weights"
    m.load_weights(path)
   
    return m


def get_sample_embedding(model, sample_data):
    """使用模型提取单个样本的特征向量"""
    sample_input = np.expand_dims(sample_data, axis=0)
    embedding = model.predict(sample_input, verbose=0)[0]
    # 确保返回的是numpy数组（而非列表）
    return np.array(embedding)

def get_template_centers(model, template_samples):
    """通过KMeans聚类从模板样本中生成中心特征向量"""
    template_embs = []
    for _, sample_data in template_samples.items():
        emb = get_sample_embedding(model, sample_data)
        template_embs.append(emb)
    template_embs = np.array(template_embs)
    
    kmeans = KMeans(n_clusters=CLUSTER_NUM, random_state=CLUSTER_RANDOM_STATE)
    kmeans.fit(template_embs)
    centers = kmeans.cluster_centers_
    return centers

def calculate_max_cosine_similarity(test_emb, template_centers):
    """
    计算测试样本到模板中心的最大余弦相似度
    修复：确保输入都是numpy数组
    """
    # 强制转换为numpy数组（核心修复点）
    test_emb = np.array(test_emb)
    template_centers = np.array(template_centers)
    
    # 扩展维度以适配cosine_similarity输入格式
    test_emb = test_emb.reshape(1, -1)
    # 计算测试样本与每个中心的余弦相似度
    similarities = cosine_similarity(test_emb, template_centers)[0]
    # 返回最大相似度（值越大越相似）
    return np.max(similarities)

def calculate_embedding_similarity(model, sample1, sample2):
    """计算两个样本的余弦相似度"""
    emb1 = get_sample_embedding(model, sample1).reshape(1, -1)
    emb2 = get_sample_embedding(model, sample2).reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def calculate_eer(similarities, labels):
    """
    基于余弦相似度计算等错误率EER
    """
    if not similarities or len(similarities) != len(labels):
        return 1.0, 0.0
    
    # 余弦相似度取值范围是[-1, 1]
    thresholds = np.linspace(-1, 1, 200)
    min_eer = 1.0
    best_threshold = 0.0
    
    for th in thresholds:
        # 相似度 > 阈值 判定为正样本
        preds = [1 if s > th else 0 for s in similarities]
        
        fp = 0  # 负样本被错误判定为正样本
        fn = 0  # 正样本被错误判定为负样本
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

# ===================== 4. 按用户+TimePeriod计算EER（余弦相似度版） =====================
def calculate_session_level_eer(model, posture_pattern_data):
    """按用户+TimePeriod维度计算EER（使用余弦相似度）"""
    session_eer_list = []
    all_users = list(posture_pattern_data.keys())
    
    # 第一步：为所有用户提取特征向量，并生成模板中心
    user_emb_info = {}
    for user_id in all_users:
        user_data = posture_pattern_data[user_id]
        
        # 生成模板中心特征向量
        template_centers = get_template_centers(model, user_data["template_samples"])
        
        # 提取测试时段的特征向量
        test_period_embs = {}
        for time_period, tp_samples in user_data["test_periods"].items():
            tp_embs = []
            for _, sample_data in tp_samples.items():
                emb = get_sample_embedding(model, sample_data)
                tp_embs.append(emb)
            test_period_embs[time_period] = tp_embs
        
        # 提取该用户所有样本的特征向量（核心修复：不转换为list）
        all_user_embs = []
        # 直接添加numpy数组，不使用tolist()
        all_user_embs.extend(template_centers)  # 移除tolist()
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
            
            # 构建正样本对：测试样本到模板中心的最大余弦相似度
            pos_similarities = []
            for test_emb in tp_embs:
                sim = calculate_max_cosine_similarity(test_emb, target_template_centers)
                pos_similarities.append((sim, 1))
            
            # 构建负样本对：其他用户样本到该模板中心的最大余弦相似度
            neg_similarities = []
            for other_user in all_users:
                if other_user == target_user:
                    continue
                other_embs = user_emb_info[other_user]["all_embs"]
                for emb in other_embs:
                    # 增加异常处理
                    try:
                        neg_sim = calculate_max_cosine_similarity(emb, target_template_centers)
                        neg_similarities.append((neg_sim, 0))
                    except Exception as e:
                        print(f"计算用户{other_user}样本相似度失败: {e}，跳过该样本")
                        continue
            
            if not neg_similarities:
                print(f"用户{target_user}时段{time_period}无负样本对，跳过")
                continue
            
            # 平衡正负样本数量
            try:
                neg_similarities = random.sample(neg_similarities, len(pos_similarities))
            except ValueError:
                # 负样本数量不足时取全部
                neg_similarities = neg_similarities[:len(pos_similarities)]
            
            all_pairs = pos_similarities + neg_similarities
            similarities = [s for s, _ in all_pairs]
            labels = [l for _, l in all_pairs]
            
            # 计算该TimePeriod的EER
            eer, _ = calculate_eer(similarities, labels)
            
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
        
        # 加入模板样本
        for sample_id, sample_data in user_data["template_samples"].items():
            emb = get_sample_embedding(model, sample_data)
            user_sample_embs.append(emb)
            all_embeddings.append(emb)
            all_user_labels.append(user_id)
            sample_ids.append(f"{user_id}_template_TP1+2_{sample_id}")
        
        # 加入测试样本
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

# ===================== 6. 绘制TSNE图 =====================
def plot_tsne_for_walk_twoone(model, groups):
    """绘制pattern为"two_one"、posture为"walk"的TSNE图"""
    if "walk" not in groups or "two_one" not in groups["walk"]:
        raise ValueError("未找到walk_two-one分组数据")
    
    target_data = groups["walk"]["two_one"]
    user_embeddings, embeddings, user_labels, _ = extract_user_embeddings(model, target_data)
    
    # TSNE降维
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, 
                random_state=TSNE_RANDOM_STATE, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)
    
    # 保存TSNE数据
    tsne_data = pd.DataFrame({
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1],
        'user_id': user_labels,
        **{f'emb_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
    })
    tsne_data.to_csv("tsne_walk_twoone_data.csv", index=False, encoding="utf-8-sig")
    print("TSNE绘图数据已保存至 tsne_walk_twoone_data.csv")
    
    # 绘制TSNE图
    plt.figure(figsize=TSNE_FIG_SIZE)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                          c=range(len(user_labels)), cmap='tab10', s=150, alpha=0.9)
    
    plt.xlabel('', fontsize=32)
    plt.ylabel('t-SNE StrokePL', fontsize=32)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("tsne_walk_twoone.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("TSNE图已保存为 tsne_walk_twoone.png")
    
    return tsne_results, user_labels

# ===================== 7. 主流程执行 =====================
if __name__ == "__main__":
    # 步骤1：加载并预处理数据
    # ==========================
    # 正确
    # ==========================
    groups, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
    # 步骤2：加载模型
    model = load_trained_model()
    
    # 步骤3：遍历每个Posture+pattern分组计算EER
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
                
                # 打印当前分组结果
                print(f"分组{group_full_name} 时段级EER结果:")
                for res in session_eer_list:
                    print(f"  用户{res['用户ID']} 时段{res['测试时段ID']} EER: {res['EER']}")
                
                # 计算统计量
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
                # 打印详细的错误堆栈信息
                import traceback
                traceback.print_exc()
                continue
    
    # 步骤4：绘制TSNE图
    try:
        plot_tsne_for_walk_twoone(model, groups)
    except Exception as e:
        print(f"TSNE绘图失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 步骤5：保存结果
    results_df = pd.DataFrame(all_session_results)
    results_df = results_df[["Posture", "Pattern", "分组名称", "用户ID", "测试时段ID", "EER"]]
    results_df.to_csv("timeperiod_level_eer_results_cosine.csv", index=False, encoding="utf-8-sig")
    print("\n时段级EER结果（余弦相似度）已保存至 timeperiod_level_eer_results_cosine.csv")
    print(f"共生成 {len(results_df)} 条时段级测试结果")