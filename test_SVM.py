import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# ===================== 1. 配置路径和参数 =====================
WEIGHTS_DIR = r"E:\【论文撰写】\投稿\StrokePL\Codes\model"
TEST_DATA_PATH = r"E:\【论文撰写】\投稿\StrokePL\Codes\datasets\test_data.csv"
SEQUENCE_LENGTH = 150
INPUT_FEATURES = 17

# TSNE配置
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_FIG_SIZE = (16, 12)

# OVSVM配置
OVSVM_NU = 0.1  # 异常值比例参数
OVSVM_GAMMA = 'scale'  # 核函数参数

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
                
                # 使用TimePeriod 1,2,3作为训练样本，TimePeriod 4作为测试样本
                train_df = user_df[user_df['TimePeriod'].isin([3])].copy()
                test_df = user_df[user_df['TimePeriod'].isin([4])].copy()
                
                if len(train_df) == 0 or len(test_df) == 0:
                    print(f"用户{uuid}缺少训练样本(TimePeriod=1+2+3)或测试样本(4)，跳过")
                    continue
                
                # 处理训练样本（用于训练OVSVM）
                train_samples = []
                for sample_id, sample_df in train_df.groupby('Sample ID'):
                    sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                    seq_data = sample_df[feature_cols].values
                    if len(seq_data) < SEQUENCE_LENGTH:
                        pad_len = SEQUENCE_LENGTH - len(seq_data)
                        seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
                    else:
                        seq_data = seq_data[:SEQUENCE_LENGTH]
                    train_samples.append(seq_data)
                
                # 处理测试样本
                test_samples = []
                for sample_id, sample_df in test_df.groupby('Sample ID'):
                    sample_df = sample_df.sort_values('Time').reset_index(drop=True)
                    seq_data = sample_df[feature_cols].values
                    if len(seq_data) < SEQUENCE_LENGTH:
                        pad_len = SEQUENCE_LENGTH - len(seq_data)
                        seq_data = np.pad(seq_data, ((0, pad_len), (0, 0)), mode='constant')
                    else:
                        seq_data = seq_data[:SEQUENCE_LENGTH]
                    test_samples.append(seq_data)
                
                if test_samples and train_samples:
                    user_samples[uuid] = {
                        "train_samples": train_samples,
                        "test_samples": test_samples
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
    path = "./modelDis/final_model_weights"
    m.load_weights(path)
   
    return m

# ===================== 3. 核心函数：特征向量提取和OVSVM =====================
def get_sample_embedding(model, sample_data):
    """使用模型提取单个样本的特征向量"""
    sample_input = np.expand_dims(sample_data, axis=0)
    embedding = model.predict(sample_input, verbose=0)[0]
    return np.array(embedding)

def train_ovsvm(train_embs):
    """使用训练样本的特征向量训练One-Class SVM"""
    clf = OneClassSVM(nu=OVSVM_NU, kernel='rbf', gamma=OVSVM_GAMMA)
    clf.fit(train_embs)
    return clf

def predict_with_ovsvm(clf, test_emb):
    """使用OVSVM进行预测，返回决策分数（正值表示正常，负值表示异常）"""
    test_emb = test_emb.reshape(1, -1)
    score = clf.decision_function(test_emb)[0]
    return score

# ===================== 4. 计算EER（基于OVSVM） =====================
def calculate_eer_ovsvm(scores, labels):
    """
    基于OVSVM决策分数计算EER
    score > 0 → 预测为正常（同一用户）
    score < 0 → 预测为异常（不同用户）
    """
    if not scores or len(scores) != len(labels):
        return 1.0, 0.0
    
    thresholds = np.linspace(min(scores), max(scores), 200)
    min_eer = 1.0
    best_threshold = 0.0
    
    for th in thresholds:
        # score > th → 判定为正样本（同一用户）
        preds = [1 if s > th else 0 for s in scores]
        
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

# ===================== 5. 按用户计算EER（OVSVM版） =====================
def calculate_session_level_eer_ovsvm(model, posture_pattern_data):
    """按用户维度计算EER（使用OVSVM）"""
    session_eer_list = []
    all_users = list(posture_pattern_data.keys())
    total_pos_samples = 0
    total_neg_samples_before = 0
    total_neg_samples_after = 0
    
    # 第一步：为所有用户提取特征向量
    user_emb_info = {}
    for user_id in all_users:
        user_data = posture_pattern_data[user_id]
        
        # 提取训练样本特征（TimePeriod 1,2,3）
        train_embs = []
        for sample_data in user_data["train_samples"]:
            emb = get_sample_embedding(model, sample_data)
            train_embs.append(emb)
        
        # 提取测试样本特征（TimePeriod 4）
        test_embs = []
        for sample_data in user_data["test_samples"]:
            emb = get_sample_embedding(model, sample_data)
            test_embs.append(emb)
        
        # 收集所有样本特征（用于作为负样本）
        all_embs = train_embs + test_embs
        
        user_emb_info[user_id] = {
            "train_embs": train_embs,
            "test_embs": test_embs,
            "all_embs": all_embs
        }
    
    # 第二步：为每个用户训练OVSVM并计算EER
    for target_user in all_users:
        target_info = user_emb_info[target_user]
        target_train_embs = target_info["train_embs"]
        target_test_embs = target_info["test_embs"]
        
        if not target_train_embs or len(target_train_embs) < 2:
            print(f"用户{target_user}训练样本不足，跳过")
            continue
        
        # 训练OVSVM
        clf = train_ovsvm(np.array(target_train_embs))
        
        # 正样本：同一用户的测试样本（TimePeriod 4）
        pos_scores = []
        for test_emb in target_test_embs:
            score = predict_with_ovsvm(clf, test_emb)
            pos_scores.append((score, 1))
        
        # 负样本：其他用户的所有样本（训练+测试）
        neg_scores = []
        for other_user in all_users:
            if other_user == target_user:
                continue
            other_embs = user_emb_info[other_user]["all_embs"]
            for emb in other_embs:
                score = predict_with_ovsvm(clf, emb)
                neg_scores.append((score, 0))
        
        if not neg_scores:
            print(f"用户{target_user}无负样本，跳过")
            continue
        
        # 记录原始负样本数量
        original_neg_count = len(neg_scores)
        
        # ===== 关键：控制负样本数量与正样本数量相同 =====
        target_neg_count = len(pos_scores)
        if len(neg_scores) >= target_neg_count:
            # 负样本充足：随机采样至与正样本数量相同
            neg_scores = random.sample(neg_scores, target_neg_count)
        else:
            # 负样本不足：使用所有负样本（此时正负样本不均衡，但无法避免）
            print(f"警告：用户{target_user}负样本不足 ({len(neg_scores)} < {target_neg_count})")
        
        # 累计统计
        total_pos_samples += len(pos_scores)
        total_neg_samples_before += original_neg_count
        total_neg_samples_after += len(neg_scores)
        
        all_pairs = pos_scores + neg_scores
        scores = [s for s, _ in all_pairs]
        labels = [l for _, l in all_pairs]
        
        eer, _ = calculate_eer_ovsvm(scores, labels)
        
        session_eer_list.append({
            "用户ID": target_user,
            "训练样本数": len(target_train_embs),
            "测试样本数": len(target_test_embs),
            "正样本数": len(pos_scores),
            "负样本数(均衡前)": original_neg_count,
            "负样本数(均衡后)": len(neg_scores),
            "EER": round(eer, 4)
        })
    
    # 输出总体统计
    print(f"\n===== 正负样本统计 =====")
    print(f"总正样本数: {total_pos_samples}")
    print(f"总负样本数(均衡前): {total_neg_samples_before}")
    print(f"总负样本数(均衡后): {total_neg_samples_after}")
    print(f"正负样本比例(均衡后): 1:{total_neg_samples_after/total_pos_samples:.2f}")
    
    return session_eer_list

# ===================== 6. 提取用户特征向量用于TSNE =====================
def extract_user_embeddings(model, posture_pattern_data):
    """提取每个用户的所有样本特征向量"""
    user_embeddings = {}
    all_embeddings = []
    all_user_labels = []
    sample_ids = []
    
    for user_id, user_data in posture_pattern_data.items():
        user_sample_embs = []
        
        for idx, sample_data in enumerate(user_data["train_samples"]):
            emb = get_sample_embedding(model, sample_data)
            user_sample_embs.append(emb)
            all_embeddings.append(emb)
            all_user_labels.append(user_id)
            sample_ids.append(f"{user_id}_train_{idx+1}")
        
        for idx, sample_data in enumerate(user_data["test_samples"]):
            embedding = get_sample_embedding(model, sample_data)
            user_sample_embs.append(embedding)
            all_embeddings.append(embedding)
            all_user_labels.append(user_id)
            sample_ids.append(f"{user_id}_test_{idx+1}")
        
        user_embeddings[user_id] = np.array(user_sample_embs)
    
    all_embeddings = np.array(all_embeddings)
    all_user_labels = np.array(all_user_labels)
    
    return user_embeddings, all_embeddings, all_user_labels, sample_ids

# ===================== 7. 绘制TSNE图 =====================
def plot_tsne_for_walk_twoone(model, groups):
    """绘制pattern为"two_one"、posture为"walk"的TSNE图"""
    if "walk" not in groups or "two_one" not in groups["walk"]:
        raise ValueError("未找到walk_two-one分组数据")
    
    target_data = groups["walk"]["two_one"]
    user_embeddings, embeddings, user_labels, _ = extract_user_embeddings(model, target_data)
    
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, 
                random_state=TSNE_RANDOM_STATE, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)
    
    tsne_data = pd.DataFrame({
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1],
        'user_id': user_labels,
        **{f'emb_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
    })
    tsne_data.to_csv("tsne_walk_twoone_data.csv", index=False, encoding="utf-8-sig")
    print("TSNE绘图数据已保存至 tsne_walk_twoone_data.csv")
    
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

# ===================== 8. 主流程执行 =====================
if __name__ == "__main__":
    print("===== OVSVM测试模式 =====")
    print(f"OVSVM参数: nu={OVSVM_NU}, gamma={OVSVM_GAMMA}")
    print(f"训练样本时段: TimePeriod 1, 2, 3")
    print(f"测试样本时段: TimePeriod 4")
    print(f"负样本策略: 与正样本数量相同\n")
    
    groups, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
    model = load_trained_model()
    
    all_session_results = []
    for posture_name, pattern_dict in groups.items():
        for pattern_name, user_data in pattern_dict.items():
            group_full_name = f"{posture_name}_{pattern_name}"
            print(f"\n===== 处理分组: {group_full_name} =====")
            
            try:
                session_eer_list = calculate_session_level_eer_ovsvm(model, user_data)
                for res in session_eer_list:
                    res["Posture"] = posture_name
                    res["Pattern"] = pattern_name
                    res["分组名称"] = group_full_name
                all_session_results.extend(session_eer_list)
                
                print(f"\n分组{group_full_name} EER结果:")
                for res in session_eer_list:
                    print(f"  用户{res['用户ID'][:8]}... 正样本={res['正样本数']} 负样本={res['负样本数(均衡后)']} EER: {res['EER']}")
                
                group_df = pd.DataFrame(session_eer_list)
                mean_eer = group_df['EER'].mean().round(4)
                std_eer = group_df['EER'].std().round(4)
                print(f"\n分组{group_full_name} 平均EER: {mean_eer} ± {std_eer}")
            except Exception as e:
                print(f"分组{group_full_name}处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    try:
        plot_tsne_for_walk_twoone(model, groups)
    except Exception as e:
        print(f"TSNE绘图失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    results_df = pd.DataFrame(all_session_results)
    results_df = results_df[["Posture", "Pattern", "分组名称", "用户ID", "训练样本数", "测试样本数", 
                           "正样本数", "负样本数(均衡前)", "负样本数(均衡后)", "EER"]]
    results_df.to_csv("ovsvm_eer_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nOVSVM EER结果已保存至 ovsvm_eer_results.csv")
    print(f"共生成 {len(results_df)} 条测试结果")
    
    # 统计总体结果
    overall_mean = results_df['EER'].mean().round(4)
    overall_std = results_df['EER'].std().round(4)
    print(f"\n===== 总体统计 =====")
    print(f"总样本数: {len(results_df)}")
    print(f"平均EER: {overall_mean}")
    print(f"EER标准差: {overall_std}")
