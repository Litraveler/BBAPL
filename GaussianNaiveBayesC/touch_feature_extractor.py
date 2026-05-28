import numpy as np
import math
from scipy import stats

class TouchFeatureExtractor:
    """基于IEEE 2019论文的触摸特征提取器（适配线段时间划分+移除size_major/size_minor）"""
    def __init__(self):
        self.angle_threshold = 20  # 线段切割夹角阈值（度）

    def get_pattern_segment_num(self, pattern):
        """根据pattern获取预期线段数"""
        if pattern.startswith('two'):
            return 2
        elif pattern.startswith('three'):
            return 3
        elif pattern.startswith('four'):
            return 4
        else:
            return None  # 未知pattern

    def segment_cutter(self, x, y, time_s, expected_seg_num=None):
        """
        线段切割器：根据触摸点坐标+时间戳切割线段，支持按pattern验证线段数
        参数：
            x/y: 触摸坐标序列
            time_s: 时间戳序列（秒）
            expected_seg_num: 预期线段数（从pattern获取）
        返回：
            segments: [(x_seg, y_seg, time_seg_start, time_seg_end), ...]
        """
        if len(x) < 2 or len(y) < 2:
            return [(x, y, time_s[0], time_s[-1])] if len(x) > 0 else []
        
        segments = []
        current_seg_x = [x[0]]
        current_seg_y = [y[0]]
        current_seg_start = time_s[0]
        
        # 核心切割逻辑
        for i in range(1, len(x)-1):
            vec1 = np.array([x[i]-x[i-1], y[i]-y[i-1]])
            vec2 = np.array([x[i+1]-x[i], y[i+1]-y[i]])
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = math.degrees(np.arccos(cos_theta))
            
            if angle > self.angle_threshold:
                current_seg_x.append(x[i])
                current_seg_y.append(y[i])
                current_seg_end = time_s[i]
                segments.append((np.array(current_seg_x), np.array(current_seg_y), current_seg_start, current_seg_end))
                current_seg_x = [x[i]]
                current_seg_y = [y[i]]
                current_seg_start = time_s[i]
            else:
                current_seg_x.append(x[i])
                current_seg_y.append(y[i])
        
        # 处理最后一段
        current_seg_x.append(x[-1])
        current_seg_y.append(y[-1])
        current_seg_end = time_s[-1]
        segments.append((np.array(current_seg_x), np.array(current_seg_y), current_seg_start, current_seg_end))
        
        # 按pattern验证并修正线段数
        if expected_seg_num is not None and len(segments) != expected_seg_num:
            segments = self.adjust_segments(segments, expected_seg_num, x, y, time_s)
        
        return segments

    def adjust_segments(self, segments, expected_num, x, y, time_s):
        """调整线段数至预期值（不足则拆分最长段，多余则合并最短段）"""
        current_num = len(segments)
        
        # 不足：拆分最长线段
        while current_num < expected_num:
            # 找到最长线段
            seg_lengths = [np.sqrt((seg[0][-1]-seg[0][0])**2 + (seg[1][-1]-seg[1][0])**2) for seg in segments]
            longest_idx = np.argmax(seg_lengths)
            longest_seg = segments[longest_idx]
            
            # 拆分最长线段（取中点）
            seg_x, seg_y, seg_start, seg_end = longest_seg
            mid_idx = len(seg_x) // 2
            seg1_x = seg_x[:mid_idx+1]
            seg1_y = seg_y[:mid_idx+1]
            seg1_start = seg_start
            seg1_end = time_s[np.where(time_s >= seg_start)[0][mid_idx]]
            
            seg2_x = seg_x[mid_idx:]
            seg2_y = seg_y[mid_idx:]
            seg2_start = seg1_end
            seg2_end = seg_end
            
            # 替换原线段
            segments.pop(longest_idx)
            segments.insert(longest_idx, (seg2_x, seg2_y, seg2_start, seg2_end))
            segments.insert(longest_idx, (seg1_x, seg1_y, seg1_start, seg1_end))
            current_num += 1
        
        # 多余：合并最短线段
        while current_num > expected_num:
            # 找到最短的相邻线段对
            min_pair_len = float('inf')
            min_pair_idx = 0
            for i in range(len(segments)-1):
                seg1_len = np.sqrt((segments[i][0][-1]-segments[i][0][0])**2 + (segments[i][1][-1]-segments[i][1][0])**2)
                seg2_len = np.sqrt((segments[i+1][0][-1]-segments[i+1][0][0])**2 + (segments[i+1][1][-1]-segments[i+1][1][0])**2)
                total_len = seg1_len + seg2_len
                if total_len < min_pair_len:
                    min_pair_len = total_len
                    min_pair_idx = i
            
            # 合并相邻线段
            seg1 = segments[min_pair_idx]
            seg2 = segments[min_pair_idx+1]
            merged_x = np.concatenate([seg1[0], seg2[0][1:]])
            merged_y = np.concatenate([seg1[1], seg2[1][1:]])
            merged_start = seg1[2]
            merged_end = seg2[3]
            
            # 替换原线段对
            segments.pop(min_pair_idx+1)
            segments.pop(min_pair_idx)
            segments.insert(min_pair_idx, (merged_x, merged_y, merged_start, merged_end))
            current_num -= 1
        
        return segments

    def extract_single_sample_features(self, features, pattern=None):
        """
        提取单个触摸样本的特征（移除size_major/size_minor，记录线段时间戳）
        参数：
            features: 单样本特征矩阵 (seq_len, 5) - X/Y/Pressure/Size/Orientation
            pattern: 图案名称（用于验证线段数）
        返回：
            feat_vec: 触摸特征向量
            seg_time_ranges: 线段时间范围 [(start1, end1), (start2, end2), ...]
        """
        # 解包字段（移除size_major/size_minor）
        x = features[:, 0]
        y = features[:, 1]
        pressure = features[:, 2]
        size = features[:, 3]
        orientation = features[:, 4]
        seq_len = len(x)
        
        if seq_len < 2:
            return np.zeros(116, dtype=np.float32), []  # 调整特征维度
        
        # 时间序列（纳秒转秒）
        time_ns = np.arange(seq_len)
        time_s = time_ns / 1e9
        
        # 获取预期线段数
        expected_seg_num = self.get_pattern_segment_num(pattern) if pattern else None
        
        # 切割线段（带时间戳）
        segments = self.segment_cutter(x, y, time_s, expected_seg_num)
        seg_num_te = [len(sx) for sx, _, _, _ in segments]
        seg_time_ranges = [(seg[2], seg[3]) for seg in segments]  # 线段时间范围
        
        # ====================== 1. 触摸事件数特征 ======================
        num_te = seq_len
        paper_te_feats = [
            num_te, np.mean(seg_num_te), np.std(seg_num_te), np.max(seg_num_te), np.min(seg_num_te)
        ]

        # ====================== 2. 触摸压力特征 ======================
        paper_press_feats = [
            np.mean(pressure), np.max(pressure), np.min(pressure), np.std(pressure),
            stats.skew(pressure), stats.kurtosis(pressure)
        ]

        # ====================== 3. 触摸面积特征（仅保留Size） ======================
        paper_size_feats = [
            np.mean(size), np.max(size), np.min(size), np.std(size)
        ]

        # ====================== 4. 滑动速度特征 ======================
        step_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        time_diff = np.diff(time_s)
        time_diff[time_diff == 0] = 1e-9
        step_speed = step_dist / time_diff
        total_dist = np.sum(step_dist)
        total_time = time_s[-1] - time_s[0]
        avg_speed = total_dist / total_time
        
        paper_speed_feats = [
            avg_speed, np.mean(step_speed), np.max(step_speed), np.min(step_speed),
            np.std(step_speed), stats.skew(step_speed)
        ]

        # ====================== 5. 触摸方向特征 ======================
        dir_vec = np.array([x[-1]-x[0], y[-1]-y[0]])
        dir_angle = math.degrees(np.arctan2(dir_vec[1], dir_vec[0])) if np.linalg.norm(dir_vec) > 0 else 0
        paper_orient_feats = [
            np.mean(orientation), np.max(orientation), np.min(orientation), np.std(orientation),
            dir_angle, np.mean(np.diff(orientation)), np.std(np.diff(orientation))
        ]

        # ====================== 6. Segment级特征 ======================
        seg_dist = [np.sqrt((sx[-1]-sx[0])**2 + (sy[-1]-sy[0])**2) for sx, sy, _, _ in segments]
        seg_time = [seg_end - seg_start for _, _, seg_start, seg_end in segments]
        seg_speed = [d/t if t>1e-9 else 0 for d, t in zip(seg_dist, seg_time)]

        seg_feats = [
            len(segments),
            np.mean(seg_dist) if segments else 0, np.max(seg_dist) if segments else 0, np.min(seg_dist) if segments else 0,
            np.mean(seg_speed) if segments else 0, np.max(seg_speed) if segments else 0,
            np.mean(seg_num_te) if segments else 0, np.std(seg_num_te) if segments else 0
        ]

        # ====================== 7. 线段间夹角特征 ======================
        angle_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
        if len(segments) > 1:
            seg_angles = []
            for i in range(len(segments)-1):
                seg1 = np.array([segments[i][0][-1]-segments[i][0][0], segments[i][1][-1]-segments[i][1][0]])
                seg2 = np.array([segments[i+1][0][-1]-segments[i+1][0][0], segments[i+1][1][-1]-segments[i+1][1][0]])
                cos_theta = np.dot(seg1, seg2) / (np.linalg.norm(seg1) * np.linalg.norm(seg2) + 1e-8)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                seg_angles.append(math.degrees(np.arccos(cos_theta)))
            angle_feats = [
                np.mean(seg_angles), np.std(seg_angles), np.max(seg_angles),
                np.min(seg_angles), stats.skew(seg_angles)
            ]

        # ====================== 8. 轨迹统计特征 ======================
        track_feats = [
            total_dist,
            np.mean(step_dist), np.max(step_dist), np.min(step_dist), np.std(step_dist),
            len(np.unique(np.round(x, 1))),
            len(np.unique(np.round(y, 1))),
            stats.skew(x), stats.skew(y), stats.kurtosis(x), stats.kurtosis(y),
            np.ptp(x), np.ptp(y),
            np.mean(np.abs(x - np.mean(x))), np.mean(np.abs(y - np.mean(y)))
        ]

        # ====================== 合并特征 ======================
        feat_vec = np.concatenate([
            paper_te_feats, paper_press_feats, paper_size_feats,
            paper_speed_feats, paper_orient_feats, seg_feats,
            angle_feats, track_feats
        ]).astype(np.float32)

        # 处理异常值
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return feat_vec

    def extract_batch_features(self, batch_features, patterns=None):
        """
        提取批量样本特征
        参数：
            batch_features: 批量触摸特征 (n_samples, seq_len, 5)
            patterns: 批量图案名称列表 [pattern1, pattern2, ...]
        返回：
            batch_feat_vecs: 批量特征向量
            batch_seg_time_ranges: 批量线段时间范围
        """
        batch_feat_vecs = []
        batch_seg_time_ranges = []
        patterns = patterns if patterns is not None else [None]*len(batch_features)
        
        for sample, pattern in zip(batch_features, patterns):
            feat_vec, seg_time_ranges = self.extract_single_sample_features(sample, pattern)
            batch_feat_vecs.append(feat_vec)
            batch_seg_time_ranges.append(seg_time_ranges)
        
        return np.array(batch_feat_vecs), batch_seg_time_ranges