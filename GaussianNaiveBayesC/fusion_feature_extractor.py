import numpy as np
from touch_feature_extractor import TouchFeatureExtractor
from sensor_feature_extractor import SensorFeatureExtractor

class FusionFeatureExtractor:
    """触摸+传感器融合特征提取器（移除线段时间切片，提取全量传感器特征）"""
    def __init__(self):
        self.touch_extractor = TouchFeatureExtractor()
        self.sensor_extractor = SensorFeatureExtractor()

    def extract_single_sample_fusion_features(self, touch_features, sensor_dict, pattern=None):
        """
        提取单个样本的融合特征（全量传感器数据，不按线段时间切片）
        参数：
            touch_features: 触摸特征矩阵 (seq_len, 5) - X/Y/Pressure/Size/Orientation
            sensor_dict: 传感器字典 
                {
                    'Gravity': (seq_len, 4),  # Time/X/Y/Z
                    'Accelerometer': (seq_len, 4),
                    'Gyroscope': (seq_len, 4)
                }
            pattern: 图案名称（用于验证线段数，仅触摸特征使用）
        返回：
            fusion_feat: 融合特征向量
        """
        # 仅提取触摸特征（移除seg_time_ranges返回值，需同步修改touch_extractor）
        # 若touch_extractor的extract_single_sample_features仍返回两个值，改为：
        # touch_feat, _ = self.touch_extractor.extract_single_sample_features(touch_features, pattern)
        touch_feat = self.touch_extractor.extract_single_sample_features(touch_features)
        
        # 提取全量传感器特征（不再传递time_ranges，使用默认全量逻辑）
        gravity_feat = self.sensor_extractor.extract_single_sensor_features(sensor_dict['Gravity'])
        accel_feat = self.sensor_extractor.extract_single_sensor_features(sensor_dict['Accelerometer'])
        gyro_feat = self.sensor_extractor.extract_single_sensor_features(sensor_dict['Gyroscope'])
        
        # 融合特征
        fusion_feat = np.concatenate([
            touch_feat,
            gravity_feat,
            accel_feat,
            gyro_feat
        ]).astype(np.float32)
        
        fusion_feat = np.nan_to_num(fusion_feat, nan=0.0, posinf=0.0, neginf=0.0)
        return fusion_feat

    def extract_batch_fusion_features(self, batch_touch_features, batch_sensor_dicts, patterns=None):
        """
        提取批量样本的融合特征（全量传感器数据）
        参数：
            batch_touch_features: 批量触摸特征 (n_samples, seq_len, 5)
            batch_sensor_dicts: 批量传感器字典列表
            patterns: 批量图案名称列表（仅触摸特征使用）
        返回：
            batch_fusion_feats: 批量融合特征
        """
        batch_fusion_feats = []
        patterns = patterns if patterns is not None else [None]*len(batch_touch_features)
        
        for touch_feat, sensor_dict, pattern in zip(batch_touch_features, batch_sensor_dicts, patterns):
            fusion_feat = self.extract_single_sample_fusion_features(touch_feat, sensor_dict, pattern)
            batch_fusion_feats.append(fusion_feat)
        
        return np.array(batch_fusion_feats)