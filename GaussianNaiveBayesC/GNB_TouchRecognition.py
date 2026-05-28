import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from fusion_feature_extractor import FusionFeatureExtractor

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子（保证采样可复现）
np.random.seed(42)

# ====================== 1. 数据集定义 ======================
class TouchSensorFusionDataset(Dataset):
    """安卓解锁触摸+传感器模式数据集（适配线段时间划分）"""
    def __init__(self, touch_data, sensor_data, 
                 feature_cols=['X', 'Y', 'Pressure', 'Size', 'Orientation']): 
        self.feature_cols = feature_cols
        self.touch_data = touch_data
        self.sensor_data = sensor_data
        self.fusion_extractor = FusionFeatureExtractor()
        self.uuid_samples = self._process_fusion_data()
        self.global_scaler = self._init_global_scaler()

    def _init_global_scaler(self):
        """初始化融合特征缩放器"""
        all_feats = []
        for uuid in self.uuid_samples:
            train_touch, train_sensor, train_patterns, test_touch, test_sensor, test_patterns = self.uuid_samples[uuid]
            train_feats = self.fusion_extractor.extract_batch_fusion_features(train_touch, train_sensor, train_patterns)
            test_feats = self.fusion_extractor.extract_batch_fusion_features(test_touch, test_sensor, test_patterns)
            all_user_feats = np.concatenate([train_feats, test_feats], axis=0)
            all_feats.append(all_user_feats)
        all_feats = np.concatenate(all_feats, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_feats)
        return scaler

    def _process_sensor_data(self, uuid, sample_id):
        """处理单个样本的传感器数据（新增Time列）"""
        sensor_sample = self.sensor_data[
            (self.sensor_data['UUID'] == uuid) & 
            (self.sensor_data['Sample ID'] == sample_id)
        ]
        sensor_dict = {}
        for sensor_type in ['Gravity', 'Accelerometer', 'Gyroscope']:
            sensor_type_data = sensor_sample[sensor_sample['SensorType'] == sensor_type][['Time', 'X', 'Y', 'Z']].values  # 新增Time列
            if len(sensor_type_data) == 0:
                # 构造空数据（Time=0, X/Y/Z=0）
                sensor_type_data = np.zeros((1, 4), dtype=np.float32)
            sensor_dict[sensor_type] = sensor_type_data
        return sensor_dict

    def _process_fusion_data(self):
        """处理触摸+传感器融合数据（传递pattern）"""
        touch_uuid_samples = defaultdict(list)
        unique_uuids = sorted(self.touch_data['UUID'].unique())
        
        for uuid, uuid_data in self.touch_data.groupby('UUID'):
            for sample_id, sample_data in uuid_data.groupby('Sample ID'):
                if len(sample_data) < 5:
                    continue
                time_period = sample_data['TimePeriod'].iloc[0]
                if time_period not in [1, 2, 3, 4]:
                    continue
                # 获取pattern（用于线段数验证）
                pattern = sample_data['pattern'].iloc[0]
                # 处理触摸特征（移除size_major/size_minor）
                touch_features = sample_data[self.feature_cols].values.astype(np.float32)
                # 处理传感器特征（带Time列）
                sensor_dict = self._process_sensor_data(uuid, sample_id)
                touch_uuid_samples[uuid].append((time_period, pattern, touch_features, sensor_dict))

        # 筛选满足4个时间段各5个样本的用户
        filtered_uuid_samples = {}
        for uuid in touch_uuid_samples:
            time_period_counts = defaultdict(int)
            for tp, _, _, _ in touch_uuid_samples[uuid]:
                time_period_counts[tp] += 1
            if (time_period_counts.get(1, 0) >=5 and time_period_counts.get(2,0)>=5 and
                time_period_counts.get(3,0)>=5 and time_period_counts.get(4,0)>=5):
                # 拆分训练/测试
                tp1 = [(tf, sd, p) for tp, p, tf, sd in touch_uuid_samples[uuid] if tp ==1][:5]
                tp2 = [(tf, sd, p) for tp, p, tf, sd in touch_uuid_samples[uuid] if tp ==2][:5]
                tp3 = [(tf, sd, p) for tp, p, tf, sd in touch_uuid_samples[uuid] if tp ==3][:5]
                tp4 = [(tf, sd, p) for tp, p, tf, sd in touch_uuid_samples[uuid] if tp ==4][:5]
                
                train_touch = [tf for tf, sd, p in tp2 + tp3]
                train_sensor = [sd for tf, sd, p in tp2 + tp3]
                train_patterns = [p for tf, sd, p in tp2 + tp3]  # 保存pattern
                
                # test_touch = [tf for tf, sd, p in tp3 + tp4]
                # test_sensor = [sd for tf, sd, p in tp3 + tp4]
                # test_patterns = [p for tf, sd, p in tp3 + tp4]

                test_touch = [tf for tf, sd, p in tp4]
                test_sensor = [sd for tf, sd, p in tp4]
                test_patterns = [p for tf, sd, p in tp4]
                
                filtered_uuid_samples[uuid] = (train_touch, train_sensor, train_patterns, 
                                               test_touch, test_sensor, test_patterns)
        return filtered_uuid_samples

    def _balance_samples(self, pos_samples, neg_samples, random_state=42):
        """
        平衡正负样本数量：负样本采样至与正样本数量一致
        参数：
            pos_samples: 正样本数组 (n_pos, feat_dim)
            neg_samples: 负样本数组 (n_neg, feat_dim)
        返回：
            balanced_neg_samples: 平衡后的负样本数组
        """
        n_pos = len(pos_samples)
        n_neg = len(neg_samples)
        
        if n_neg == 0:
            raise ValueError("负样本数量为0，无法平衡")
        
        # 固定随机种子保证可复现
        rng = np.random.RandomState(random_state)
        
        # 负样本数量 >= 正样本：随机采样n_pos个
        if n_neg >= n_pos:
            selected_idx = rng.choice(n_neg, size=n_pos, replace=False)
            balanced_neg = neg_samples[selected_idx]
        # 负样本数量 < 正样本：随机重复采样（兜底方案）
        else:
            selected_idx = rng.choice(n_neg, size=n_pos, replace=True)
            balanced_neg = neg_samples[selected_idx]
        
        return balanced_neg

    def get_user_data(self, uuid):
        """获取用户融合特征数据（传递pattern）+ 样本数量平衡"""
        # 合法用户数据（正样本）
        legal_train_touch, legal_train_sensor, legal_train_patterns, \
        legal_test_touch, legal_test_sensor, legal_test_patterns = self.uuid_samples[uuid]
        
        # 提取融合特征并缩放（传递pattern）
        legal_train = self.fusion_extractor.extract_batch_fusion_features(legal_train_touch, legal_train_sensor, legal_train_patterns)
        legal_test = self.fusion_extractor.extract_batch_fusion_features(legal_test_touch, legal_test_sensor, legal_test_patterns)
        
        legal_train = self.global_scaler.transform(legal_train)
        legal_test = self.global_scaler.transform(legal_test)

        # 非法用户数据（负样本）
        illegal_train_touch = []
        illegal_train_sensor = []
        illegal_train_patterns = []
        illegal_test_touch = []
        illegal_test_sensor = []
        illegal_test_patterns = []
        
        for other_uuid in self.uuid_samples:
            if other_uuid == uuid:
                continue
            ot_train_touch, ot_train_sensor, ot_train_patterns, \
            ot_test_touch, ot_test_sensor, ot_test_patterns = self.uuid_samples[other_uuid]
            
            illegal_train_touch.extend(ot_train_touch)
            illegal_train_sensor.extend(ot_train_sensor)
            illegal_train_patterns.extend(ot_train_patterns)
            
            illegal_test_touch.extend(ot_test_touch)
            illegal_test_sensor.extend(ot_test_sensor)
            illegal_test_patterns.extend(ot_test_patterns)
        
        # 提取非法用户融合特征
        illegal_train = self.fusion_extractor.extract_batch_fusion_features(illegal_train_touch, illegal_train_sensor, illegal_train_patterns)
        illegal_test = self.fusion_extractor.extract_batch_fusion_features(illegal_test_touch, illegal_test_sensor, illegal_test_patterns)
        
        illegal_train = self.global_scaler.transform(illegal_train)
        illegal_test = self.global_scaler.transform(illegal_test)

        # ========== 核心修改：平衡样本数量 ==========
        # 1. 训练负样本 与 训练正样本 数量一致
        if len(legal_train) > 0 and len(illegal_train) > 0:
            illegal_train_balanced = self._balance_samples(legal_train, illegal_train)
        else:
            illegal_train_balanced = illegal_train
        
        # 2. 测试负样本 与 训练正样本 数量一致
        if len(legal_train) > 0 and len(illegal_test) > 0:
            illegal_test_balanced = self._balance_samples(legal_train, illegal_test)
        else:
            illegal_test_balanced = illegal_test
        
        # 输出样本数量信息（验证平衡效果）
        print(f"样本平衡后 - 训练正样本：{len(legal_train)}, 训练负样本：{len(illegal_train_balanced)}")
        print(f"样本平衡后 - 测试正样本：{len(legal_test)}, 测试负样本：{len(illegal_test_balanced)}")
        
        return legal_train, legal_test, illegal_train_balanced, illegal_test_balanced

    def get_all_users(self):
        return list(self.uuid_samples.keys())
    
    def get_all_features_for_tsne(self):
        """获取所有用户的融合特征数据（用于TSNE可视化）"""
        all_feats = []
        all_labels = []
        all_user_ids = []
        i = 0
        for uuid in self.uuid_samples:
            i += 1
            legal_train, legal_test, illegal_train, illegal_test = self.get_user_data(uuid)
            # 合法用户（正样本）标记为1，非法用户（负样本）标记为0
            # 合并训练+测试的合法样本
            legal_feats = np.concatenate([legal_train, legal_test], axis=0)
            all_feats.append(legal_feats)
            all_user_ids.extend([i]*len(legal_feats))
            
        all_feats = np.concatenate(all_feats, axis=0)
        all_user_ids = np.array(all_user_ids)
        return all_feats, all_user_ids

    def __len__(self):
        return sum(20 for _ in self.uuid_samples.values())

    def __getitem__(self, idx):
        all_touch_feats = []
        for uuid in self.uuid_samples:
            train_touch, _, _, test_touch, _, _ = self.uuid_samples[uuid]
            all_user_touch = train_touch + test_touch
            for feat in all_user_touch:
                all_touch_feats.append(feat)
        return torch.tensor(all_touch_feats[idx], dtype=torch.float32)

# ====================== 2. 辅助函数+GNBTrainer类（无修改）======================
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    frr = 1 - tpr
    min_diff_idx = np.argmin(np.abs(fpr - frr))
    eer = (fpr[min_diff_idx] + frr[min_diff_idx]) / 2
    eer_threshold = thresholds[min_diff_idx]
    return eer, eer_threshold, fpr, frr

class GNBTrainer:
    def __init__(self):
        self.gnb = GaussianNB()
        self.alpha = 2.0

    def train(self, legal_train, illegal_train):
        X_train = np.concatenate([legal_train, illegal_train], axis=0)
        y_train = np.concatenate([
            np.ones(len(legal_train)), 
            np.zeros(len(illegal_train))
        ])
        self.gnb.fit(X_train, y_train)

    def predict_score(self, X):
        proba = self.gnb.predict_proba(X)
        return proba[:, 1]

    def dynamic_threshold(self, legal_train):
        train_scores = self.predict_score(legal_train)
        tau = np.mean(train_scores) - self.alpha * np.std(train_scores)
        return tau, train_scores

    def evaluate(self, legal_test, illegal_test, tau):
        legal_scores = self.predict_score(legal_test)
        illegal_scores = self.predict_score(illegal_test)
        y_true = np.concatenate([
            np.zeros_like(legal_scores),
            np.ones_like(illegal_scores)
        ])
        y_score = np.concatenate([
            1 - legal_scores,
            1 - illegal_scores
        ])
        y_pred = np.where(y_score > tau, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_normal = tn + fp
        total_abnormal = tp + fn
        
        FAR = fp / total_normal if total_normal > 0 else 0.0
        FRR = fn / total_abnormal if total_abnormal > 0 else 0.0
        TPR = tp / total_abnormal if total_abnormal > 0 else 0.0
        TNR = tn / total_normal if total_normal > 0 else 0.0
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 1.0
        
        if len(np.unique(y_true)) > 1:
            eer, eer_threshold, _, _ = calculate_eer(y_true, y_score)
        else:
            eer = 0.0
            eer_threshold = tau
        metrics = {
            "TPR": TPR, "TNR": TNR, "FAR": FAR, "FRR": FRR,
            "EER": eer, "EER_Threshold": eer_threshold, "AUC": auc, "Threshold": tau,
            "Mean Normal Score": np.mean(1 - legal_scores), 
            "Mean Abnormal Score": np.mean(1 - illegal_scores),
            "Normal Sample Count": len(legal_scores), 
            "Abnormal Sample Count": len(illegal_scores)
        }
        return metrics

class OVSVMTrainer:
    def __init__(self, kernel='rbf', nu=0.5, gamma='scale'):
        self.ovsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.alpha = 2.0

    def train(self, legal_train, illegal_train=None):
        self.ovsvm.fit(legal_train)

    def predict_score(self, X):
        return self.ovsvm.decision_function(X)

    def dynamic_threshold(self, legal_train):
        train_scores = self.predict_score(legal_train)
        tau = np.mean(train_scores) - self.alpha * np.std(train_scores)
        return tau, train_scores

    def evaluate(self, legal_test, illegal_test, tau):
        legal_scores = self.predict_score(legal_test)
        illegal_scores = self.predict_score(illegal_test)
        y_true = np.concatenate([
            np.zeros_like(legal_scores),
            np.ones_like(illegal_scores)
        ])
        y_score = np.concatenate([
            -legal_scores,
            -illegal_scores
        ])
        y_pred = np.where(y_score > -tau, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_normal = tn + fp
        total_abnormal = tp + fn
        
        FAR = fp / total_normal if total_normal > 0 else 0.0
        FRR = fn / total_abnormal if total_abnormal > 0 else 0.0
        TPR = tp / total_abnormal if total_abnormal > 0 else 0.0
        TNR = tn / total_normal if total_normal > 0 else 0.0
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 1.0
        
        if len(np.unique(y_true)) > 1:
            eer, eer_threshold, _, _ = calculate_eer(y_true, y_score)
        else:
            eer = 0.0
            eer_threshold = tau
        metrics = {
            "TPR": TPR, "TNR": TNR, "FAR": FAR, "FRR": FRR,
            "EER": eer, "EER_Threshold": eer_threshold, "AUC": auc, "Threshold": tau,
            "Mean Normal Score": np.mean(-legal_scores), 
            "Mean Abnormal Score": np.mean(-illegal_scores),
            "Normal Sample Count": len(legal_scores), 
            "Abnormal Sample Count": len(illegal_scores)
        }
        return metrics

# ====================== 4. 主执行流程 =======================
def main():
    # 1. 加载触摸和传感器数据集
    touch_data_path = r"E:\【论文撰写】\投稿\StrokePL\Codes\Tdatas\cleaned_touch_data.csv"
    sensor_data_path = r"E:\【论文撰写】\投稿\StrokePL\Codes\Tdatas\cleaned_sensor_data.csv"
    
    if not os.path.exists(touch_data_path):
        raise FileNotFoundError(f"触摸数据文件不存在：{touch_data_path}")
    if not os.path.exists(sensor_data_path):
        raise FileNotFoundError(f"传感器数据文件不存在：{sensor_data_path}")
    
    touch_df = pd.read_csv(touch_data_path)
    sensor_df = pd.read_csv(sensor_data_path)
    
    print(f"原始触摸数据集形状：{touch_df.shape}")
    print(f"原始传感器数据集形状：{sensor_df.shape}")
    print(f"触摸数据用户UUID数量：{touch_df['UUID'].nunique()}")
    print(f"传感器数据用户UUID数量：{sensor_df['UUID'].nunique()}")

    # 2. 初始化结果存储
    all_results = {}

    for (posture, pattern), group_touch_data in touch_df.groupby(['Posture', 'pattern']):
        group_sensor_data = sensor_df[
            (sensor_df['posture'] == posture) & 
            (sensor_df['pattern'] == pattern)
        ]
        print(f"\n========== 处理分组：Posture={posture}, Pattern={pattern} ==========")
        
        dataset = TouchSensorFusionDataset(group_touch_data, group_sensor_data)
        users = dataset.get_all_users()
        num_users = len(users)
        
        if num_users < 3:
            print(f"该分组有效用户数不足（仅{num_users}个），跳过")
            continue
        print(f"有效用户数：{num_users}")


        # 4. 逐用户训练和评估（同时运行GNB和OVSVM）
        group_user_metrics = {}
        for uuid in users:
            print(f"\n----- 训练合法用户：{uuid} -----")
            legal_train, legal_test, illegal_train, illegal_test = dataset.get_user_data(uuid)
            print(f"合法用户训练样本数：{len(legal_train)}, 测试样本数：{len(legal_test)}")
            print(f"非法用户训练样本数：{len(illegal_train)}, 测试样本数：{len(illegal_test)}")
            if len(legal_train) == 0:
                print(f"用户{uuid}训练样本不足，跳过")
                continue

            # GNB分类器
            gnb_trainer = GNBTrainer()
            gnb_trainer.train(legal_train, illegal_train)
            gnb_tau, gnb_train_scores = gnb_trainer.dynamic_threshold(legal_train)
            print(f"GNB动态阈值：{gnb_tau:.4f}（训练集分数均值：{np.mean(gnb_train_scores):.4f}，标准差：{np.std(gnb_train_scores):.4f}）")
            gnb_metrics = gnb_trainer.evaluate(legal_test, illegal_test, gnb_tau)
            group_user_metrics[(uuid, 'GNB')] = gnb_metrics
            print(f"GNB用户{uuid}评估结果：")
            print(f"  TPR(异常检测率): {gnb_metrics['TPR']:.4f}, TNR(正常识别率): {gnb_metrics['TNR']:.4f}")
            print(f"  FAR(错误接受率): {gnb_metrics['FAR']:.4f}, FRR(错误拒绝率): {gnb_metrics['FRR']:.4f}")
            print(f"  EER(等错误率): {gnb_metrics['EER']:.4f}, AUC: {gnb_metrics['AUC']:.4f}")

            # OVSVM分类器（使用legal_train训练）
            ovsvm_trainer = OVSVMTrainer()
            ovsvm_trainer.train(legal_train)
            ovsvm_tau, ovsvm_train_scores = ovsvm_trainer.dynamic_threshold(legal_train)
            print(f"OVSVM动态阈值：{ovsvm_tau:.4f}（训练集分数均值：{np.mean(ovsvm_train_scores):.4f}，标准差：{np.std(ovsvm_train_scores):.4f}）")
            ovsvm_metrics = ovsvm_trainer.evaluate(legal_test, illegal_test, ovsvm_tau)
            group_user_metrics[(uuid, 'OVSVM')] = ovsvm_metrics
            print(f"OVSVM用户{uuid}评估结果：")
            print(f"  TPR(异常检测率): {ovsvm_metrics['TPR']:.4f}, TNR(正常识别率): {ovsvm_metrics['TNR']:.4f}")
            print(f"  FAR(错误接受率): {ovsvm_metrics['FAR']:.4f}, FRR(错误拒绝率): {ovsvm_metrics['FRR']:.4f}")
            print(f"  EER(等错误率): {ovsvm_metrics['EER']:.4f}, AUC: {ovsvm_metrics['AUC']:.4f}")

        # 5. 计算分组平均指标
        if group_user_metrics:
            avg_tpr = np.mean([m['TPR'] for m in group_user_metrics.values()])
            avg_tnr = np.mean([m['TNR'] for m in group_user_metrics.values()])
            avg_far = np.mean([m['FAR'] for m in group_user_metrics.values()])
            avg_frr = np.mean([m['FRR'] for m in group_user_metrics.values()])
            avg_eer = np.mean([m['EER'] for m in group_user_metrics.values()])
            avg_auc = np.mean([m['AUC'] for m in group_user_metrics.values()])
            all_results[(posture, pattern)] = {
                "用户详细指标": group_user_metrics,
                "平均TPR": avg_tpr, 
                "平均TNR": avg_tnr, 
                "平均FAR": avg_far, 
                "平均FRR": avg_frr,
                "平均EER": avg_eer, 
                "平均AUC": avg_auc, 
                "有效用户数": num_users
            }

    # 6. 输出汇总结果
    print("\n========== 所有分组结果汇总 ==========")
    for (posture, pattern), res in all_results.items():
        print(f"\nPosture={posture}, Pattern={pattern}（有效用户数：{res['有效用户数']}）：")
        print(f"  平均异常检测率(TPR)：{res['平均TPR']:.4f}")
        print(f"  平均正常识别率(TNR)：{res['平均TNR']:.4f}")
        print(f"  平均错误接受率(FAR)：{res['平均FAR']:.4f}")
        print(f"  平均错误拒绝率(FRR)：{res['平均FRR']:.4f}")
        print(f"  平均等错误率(EER)：{res['平均EER']:.4f}")
        print(f"  平均AUC：{res['平均AUC']:.4f}")

    # 7. 保存结果到CSV（添加分类器列）
    result_rows = []
    for (posture, pattern), res in all_results.items():
        for (uuid, classifier), metrics in res['用户详细指标'].items():
            row = {
                "Posture": posture,
                "Pattern": pattern,
                "UUID": uuid,
                "Classifier": classifier,
                "TPR": metrics['TPR'],
                "TNR": metrics['TNR'],
                "FAR": metrics['FAR'],
                "FRR": metrics['FRR'],
                "EER": metrics['EER'],
                "AUC": metrics['AUC'],
                "Threshold": metrics['Threshold'],
                "Mean Normal Score": metrics['Mean Normal Score'],
                "Mean Abnormal Score": metrics['Mean Abnormal Score'],
                "Normal Sample Count": metrics['Normal Sample Count'],
                "Abnormal Sample Count": metrics['Abnormal Sample Count']
            }
            result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv("fusion_recognition_results_23_4.csv", index=False, encoding='utf-8')
    print("\n融合特征结果已保存到：fusion_recognition_results_23_4.csv")
    
if __name__ == "__main__":
    main()