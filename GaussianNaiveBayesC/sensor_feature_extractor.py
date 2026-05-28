import numpy as np
import math
from scipy import stats

class SensorFeatureExtractor:
    """基于传感器数据的特征提取器（提取全量数据特征，移除时间切片逻辑）"""
    def __init__(self):
        pass

    def extract_single_sensor_features(self, sensor_data):
        """
        提取单个传感器样本的特征（强制使用全量数据，忽略time_ranges）
        参数：
            sensor_data: 单样本传感器矩阵 (seq_len, 4) - Time/X/Y/Z
            time_ranges: 兼容旧参数（无实际作用，仅保留接口）
        返回：
            feat_vec: 传感器特征向量
        """
        # 强制使用全量数据，忽略time_ranges切片逻辑
        seg_data = sensor_data[:, 1:]  # 取X/Y/Z列，忽略Time列
        if len(seg_data) == 0:
            seg_data = np.zeros((1, 3), dtype=np.float32)  # 兜底空数据
        
        # 提取全量数据的特征
        feat_vec = self._extract_segment_features(seg_data)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_vec

    def _extract_segment_features(self, seg_data):
        """提取单段传感器数据的特征（增加空数据保护）"""
        x = seg_data[:, 0]
        y = seg_data[:, 1]
        z = seg_data[:, 2]
        
        # 1. 基础统计特征（增加空数据保护）
        def safe_stat(stat_func, arr):
            """安全统计函数：空数组返回0"""
            return stat_func(arr) if len(arr) > 0 else 0.0
        
        base_feats = [
            # X轴统计
            safe_stat(np.mean, x), safe_stat(np.std, x), safe_stat(np.min, x), safe_stat(np.max, x), 
            safe_stat(stats.skew, x), safe_stat(stats.kurtosis, x), safe_stat(np.median, x),
            safe_stat(np.ptp, x), safe_stat(lambda a: np.sum(np.abs(a - np.mean(a))), x),
            # Y轴统计
            safe_stat(np.mean, y), safe_stat(np.std, y), safe_stat(np.min, y), safe_stat(np.max, y), 
            safe_stat(stats.skew, y), safe_stat(stats.kurtosis, y), safe_stat(np.median, y),
            safe_stat(np.ptp, y), safe_stat(lambda a: np.sum(np.abs(a - np.mean(a))), y),
            # Z轴统计
            safe_stat(np.mean, z), safe_stat(np.std, z), safe_stat(np.min, z), safe_stat(np.max, z), 
            safe_stat(stats.skew, z), safe_stat(stats.kurtosis, z), safe_stat(np.median, z),
            safe_stat(np.ptp, z), safe_stat(lambda a: np.sum(np.abs(a - np.mean(a))), z),
            # 向量特征
            safe_stat(lambda a: np.mean(np.sqrt(a[:,0]**2 + a[:,1]**2 + a[:,2]**2)), seg_data),
            safe_stat(lambda a: np.max(np.sqrt(a[:,0]**2 + a[:,1]**2 + a[:,2]**2)), seg_data),
            safe_stat(lambda a: np.min(np.sqrt(a[:,0]**2 + a[:,1]**2 + a[:,2]**2)), seg_data),
            safe_stat(lambda a: stats.skew(np.sqrt(a[:,0]**2 + a[:,1]**2 + a[:,2]**2)), seg_data),
        ]
        
        # 2. 时域特征（增加空数据保护）
        diff_x = np.diff(x) if len(x) > 1 else np.array([])
        diff_y = np.diff(y) if len(y) > 1 else np.array([])
        diff_z = np.diff(z) if len(z) > 1 else np.array([])
        
        time_feats = [
            safe_stat(np.mean, diff_x), safe_stat(np.std, diff_x), safe_stat(lambda a: np.max(np.abs(a)), diff_x),
            safe_stat(np.mean, diff_y), safe_stat(np.std, diff_y), safe_stat(lambda a: np.max(np.abs(a)), diff_y),
            safe_stat(np.mean, diff_z), safe_stat(np.std, diff_z), safe_stat(lambda a: np.max(np.abs(a)), diff_z),
            # 符号变化率（空数组返回0）
            safe_stat(lambda a: np.sum(np.diff(np.sign(a))) / (2 * len(a)) if len(a) > 0 else 0.0, diff_x),
            safe_stat(lambda a: np.sum(np.diff(np.sign(a))) / (2 * len(a)) if len(a) > 0 else 0.0, diff_y),
            safe_stat(lambda a: np.sum(np.diff(np.sign(a))) / (2 * len(a)) if len(a) > 0 else 0.0, diff_z),
        ]
        
        # 3. 频域特征（增加空数据保护）
        freq_feats = [0.0] * 9
        if len(x) >= 8:
            fft_x = np.fft.fft(x)
            psd_x = np.abs(fft_x) ** 2 / len(x)
            fft_y = np.fft.fft(y)
            psd_y = np.abs(fft_y) ** 2 / len(y)
            fft_z = np.fft.fft(z)
            psd_z = np.abs(fft_z) ** 2 / len(z)
            
            freq_feats = [
                safe_stat(np.mean, psd_x[:len(psd_x)//2]), 
                safe_stat(np.mean, psd_y[:len(psd_y)//2]), 
                safe_stat(np.mean, psd_z[:len(psd_z)//2]),
                safe_stat(np.max, psd_x[:len(psd_x)//2]), 
                safe_stat(np.max, psd_y[:len(psd_y)//2]), 
                safe_stat(np.max, psd_z[:len(psd_z)//2]),
                # 重心频率（避免除0）
                safe_stat(lambda a: np.sum(np.arange(len(a)) * a) / (np.sum(a) + 1e-8), psd_x[:len(psd_x)//2]),
                safe_stat(lambda a: np.sum(np.arange(len(a)) * a) / (np.sum(a) + 1e-8), psd_y[:len(psd_y)//2]),
                safe_stat(lambda a: np.sum(np.arange(len(a)) * a) / (np.sum(a) + 1e-8), psd_z[:len(psd_z)//2]),
            ]
        
        # 合并特征
        seg_feat = np.concatenate([base_feats, time_feats, freq_feats]).astype(np.float32)
        return seg_feat

    def extract_batch_sensor_features(self, batch_sensor_data, batch_time_ranges=None):
        """
        提取批量传感器样本特征（全量数据）
        参数：
            batch_sensor_data: 批量传感器矩阵 (n_samples, seq_len, 4) - Time/X/Y/Z
            batch_time_ranges: 兼容旧参数（无实际作用）
        返回：
            batch_feat_vecs: 批量特征向量
        """
        batch_feat_vecs = []
        for sample in batch_sensor_data:
            feat_vec = self.extract_single_sensor_features(sample)
            batch_feat_vecs.append(feat_vec)
        
        return np.array(batch_feat_vecs)

    def extract_multi_sensor_features(self, gravity_data, accel_data, gyro_data, time_ranges=None):
        """
        提取多传感器融合特征（全量数据）
        参数：
            gravity_data: 重力传感器数据 (n_samples, seq_len, 4)
            accel_data: 加速度传感器数据 (n_samples, seq_len, 4)
            gyro_data: 陀螺仪传感器数据 (n_samples, seq_len, 4)
            time_ranges: 兼容旧参数（无实际作用）
        返回：
            fusion_feats: 融合特征
        """
        gravity_feats = self.extract_batch_sensor_features(gravity_data)
        accel_feats = self.extract_batch_sensor_features(accel_data)
        gyro_feats = self.extract_batch_sensor_features(gyro_data)
        
        fusion_feats = np.concatenate([gravity_feats, accel_feats, gyro_feats], axis=1)
        return fusion_feats