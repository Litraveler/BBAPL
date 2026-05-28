import numpy as np
import random
import tensorflow as tf
import conf

class CurriculumSetsGenerator:
    def __init__(self, descriptor, x, K, delay, max_neighbours):
        self.descriptor = descriptor
        self.x = x                # {posture:{pattern:{is_cross:{group_id:{sample_id: array}}}}}
        self.K = K
        self.epoch = 0
        self.current_num_cross = 0      # 整体batch中的跨会话用户数（课程学习）
        self.num_same_pattern = 0       # 同一模式用户数（课程学习）
        self.cross_in_same_target = 0   # 基准模式内跨会话用户目标数（分段固定）
        self.users = {}                 # group_id -> samples
        self.mode_to_users = {}         # (posture, pattern) -> [(group_id, is_cross), ...]
        self.cross_group_ids = []       # 所有跨会话用户ID
        self.non_cross_group_ids = []   # 所有非跨会话用户ID

    def initialize(self):
        """构建索引，提取所有用户及其所属模式"""
        self.users = {}
        self.mode_to_users = {}
        self.cross_group_ids = []
        self.non_cross_group_ids = []

        for posture, patterns in self.x.items():
            for pattern, cross_dict in patterns.items():
                for is_cross, group_dict in cross_dict.items():
                    for group_id, samples in group_dict.items():
                        # 保存用户样本
                        self.users[group_id] = samples
                        # 记录跨会话标记
                        if is_cross == 1:
                            self.cross_group_ids.append(group_id)
                        else:
                            self.non_cross_group_ids.append(group_id)
                        # 记录模式到用户映射
                        key = (posture, pattern)
                        if key not in self.mode_to_users:
                            self.mode_to_users[key] = []
                        self.mode_to_users[key].append((group_id, is_cross))

        self.cross_group_ids = list(set(self.cross_group_ids))
        self.non_cross_group_ids = list(set(self.non_cross_group_ids))

        print(f"[CurriculumSets] 总用户数: {len(self.users)}")
        print(f"[CurriculumSets] 跨会话用户数: {len(self.cross_group_ids)}")
        print(f"[CurriculumSets] 非跨会话用户数: {len(self.non_cross_group_ids)}")
        print(f"[CurriculumSets] 姿态-模式组合数: {len(self.mode_to_users)}")

    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()

    def on_epoch_begin(self, epoch, logs=None):
        print("[CurriculumSets] EPOCH " + str(epoch))
        self.epoch = epoch

        # 1. 整体跨会话用户数课程：从1开始，每50轮+1，最多5
        if epoch < 50:
            self.current_num_cross = 3
        elif epoch < 100:
            self.current_num_cross = 4
        elif epoch < 150:
            self.current_num_cross = 5
        else:
            self.current_num_cross = 5

        # 2. 同一模式用户数课程：从2开始，每50轮+2，最多10
        if epoch < 50:
            self.num_same_pattern = 1
        elif epoch < 100:
            self.num_same_pattern = 2
        else:
            self.num_same_pattern = 3

        # 3. 基准模式中跨会话用户目标数（分段固定）
        if epoch < 50:
            self.cross_in_same_target = 1
        elif epoch < 100:
            self.cross_in_same_target = 2
        else:
            self.cross_in_same_target = 3

        # 边界检查
        if self.cross_in_same_target > self.num_same_pattern:
            self.cross_in_same_target = self.num_same_pattern
        if self.cross_in_same_target > len(self.cross_group_ids):
            self.cross_in_same_target = len(self.cross_group_ids)
        if self.current_num_cross > len(self.cross_group_ids):
            self.current_num_cross = len(self.cross_group_ids)
        if self.num_same_pattern > len(self.users):
            self.num_same_pattern = len(self.users)

        print(f"[CurriculumSets] 本轮参数：总跨会话={self.current_num_cross}，同一模式用户数={self.num_same_pattern}，基准模式内跨会话目标={self.cross_in_same_target}")

    def _sample_users_from_mode(self, mode_key, num_users, target_cross=None):
        """
        从指定模式中随机选取用户
        :param mode_key: (posture, pattern)
        :param num_users: 需要选取的用户数
        :param target_cross: 希望选取的跨会话用户数，None表示不限制
        :return: (selected_users, actual_cross)  selected_users: list of (group_id, is_cross)
        """
        users_list = self.mode_to_users.get(mode_key, [])
        if not users_list:
            return [], 0

        # 分离跨会话和非跨会话
        cross_users = [u for u in users_list if u[1] == 1]
        non_cross_users = [u for u in users_list if u[1] == 0]

        if target_cross is None:
            # 无限制，直接随机选num_users个用户
            if len(users_list) < num_users:
                selected = random.choices(users_list, k=num_users)
            else:
                selected = random.sample(users_list, num_users)
            actual_cross = sum(1 for u in selected if u[1] == 1)
            return selected, actual_cross

        # 有跨会话数量要求
        target_cross = min(target_cross, len(cross_users))
        target_non_cross = num_users - target_cross
        if target_non_cross < 0:
            target_non_cross = 0
            target_cross = num_users
        target_non_cross = min(target_non_cross, len(non_cross_users))

        selected = []
        if target_cross > 0:
            selected.extend(random.sample(cross_users, target_cross))
        if target_non_cross > 0:
            selected.extend(random.sample(non_cross_users, target_non_cross))

        # 如果总数不足，用剩余用户补充（允许重复）
        if len(selected) < num_users:
            remaining = num_users - len(selected)
            all_available = cross_users + non_cross_users
            if len(all_available) == 0:
                return [], 0
            extra = random.choices(all_available, k=remaining)
            selected.extend(extra)

        return selected, target_cross

    def _sample_users_from_others(self, exclude_users, num_users, target_cross):
        """
        从所有用户中排除已选用户后，选取指定数量的用户，并尽量满足跨会话用户数
        :param exclude_users: 已选用户的group_id集合
        :param num_users: 需要选取的用户数
        :param target_cross: 希望选取的跨会话用户数
        :return: selected_users list of (group_id, is_cross)
        """
        # 获取所有未被排除的用户
        all_users = []
        for gid in self.users:
            if gid in exclude_users:
                continue
            is_cross = 1 if gid in self.cross_group_ids else 0
            all_users.append((gid, is_cross))

        if len(all_users) == 0:
            return []

        # 分离跨会话和非跨会话
        cross_available = [u for u in all_users if u[1] == 1]
        non_cross_available = [u for u in all_users if u[1] == 0]

        target_cross = min(target_cross, len(cross_available))
        target_non_cross = num_users - target_cross
        if target_non_cross < 0:
            target_non_cross = 0
            target_cross = num_users
        target_non_cross = min(target_non_cross, len(non_cross_available))

        selected = []
        if target_cross > 0:
            selected.extend(random.sample(cross_available, target_cross))
        if target_non_cross > 0:
            selected.extend(random.sample(non_cross_available, target_non_cross))

        # 如果仍不足，从剩余中补充
        if len(selected) < num_users:
            remaining = num_users - len(selected)
            all_available = cross_available + non_cross_available
            if len(all_available) == 0:
                return []
            extra = random.choices(all_available, k=remaining)
            selected.extend(extra)

        return selected

    def get_random_sets_batch(self):
        """生成一个batch，包含K个用户，整体跨会话数为self.current_num_cross，
           基准模式中的跨会话数为self.cross_in_same_target"""
        # 1. 随机选择一个姿态和模式作为基准模式
        mode_keys = list(self.mode_to_users.keys())
        if not mode_keys:
            raise RuntimeError("没有可用的姿态-模式组合")
        base_mode = random.choice(mode_keys)
        print(f"长度：{len(self.non_cross_group_ids)}")
        if len(self.non_cross_group_ids) > 0:
            # 2. 从基准模式中选取 self.num_same_pattern 个用户，目标跨会话数为 self.cross_in_same_target
            same_users, actual_cross_same = self._sample_users_from_mode(base_mode, self.num_same_pattern, self.cross_in_same_target)

            if not same_users:
                # 基准模式没有用户，回退：从所有用户中随机选 self.num_same_pattern 个
                print(f"警告：基准模式{base_mode}无用户，将从所有用户中随机选取")
                all_users = [(gid, 1 if gid in self.cross_group_ids else 0) for gid in self.users]
                if len(all_users) == 0:
                    raise RuntimeError("无可用用户")
                same_users = random.choices(all_users, k=self.num_same_pattern) if len(all_users) < self.num_same_pattern else random.sample(all_users, self.num_same_pattern)
                actual_cross_same = sum(1 for _, is_cross in same_users if is_cross == 1)

            # 3. 计算剩余需要选取的用户数和跨会话用户数
            remaining_users = self.K - len(same_users)
            remaining_cross = self.current_num_cross - actual_cross_same

            if remaining_users < 0:
                # 基准模式选取的用户数超过K，截断
                same_users = same_users[:self.K]
                remaining_users = 0
                remaining_cross = 0

            # 4. 从其他用户中选取剩余部分，并尽量满足剩余的跨会话需求
            exclude_ids = {uid for uid, _ in same_users}
            other_users = []
            if remaining_users > 0:
                other_users = self._sample_users_from_others(exclude_ids, remaining_users, remaining_cross)

            # 5. 合并所有用户
            all_selected = same_users + other_users

            # 如果总数不足K，重复补充（极端情况）
            while len(all_selected) < self.K:
                all_users_list = [(gid, 1 if gid in self.cross_group_ids else 0) for gid in self.users]
                if all_users_list:
                    all_selected.append(random.choice(all_users_list))
                else:
                    break
        else:
            # 2. 从所有用户中随机选 self.K 个（non_cross_group_ids为空时）
            # self.users是字典，需要先转换为(gid, is_cross)列表
            all_users_list = [(gid, 1 if gid in self.cross_group_ids else 0) for gid in self.users]
            if not all_users_list:
                raise RuntimeError("无可用用户")
            if len(all_users_list) < self.K:
                # 用户不足，重复选择
                all_selected = random.choices(all_users_list, k=self.K)
            else:
                all_selected = random.sample(all_users_list, k=self.K)

        # 打乱顺序
        random.shuffle(all_selected)

        # 6. 为每个用户选取样本
        X = []
        Y = []
        for i, (group_id, is_cross) in enumerate(all_selected):
            samples_dict = self.users[group_id]
            if not samples_dict:
                continue
            sample_arrays = list(samples_dict.values())
            random.shuffle(sample_arrays)
            for sample_arr in sample_arrays[:conf.N]:
                X.append(sample_arr)
                Y.append(i)

        if len(X) == 0:
            raise RuntimeError("没有生成任何样本，请检查数据")

        X = np.stack(X)
        Y = np.stack(Y)
        return X, Y
    def __call__(self):
        """无限生成器，每次返回一个batch"""
        while True:
            yield self.get_random_sets_batch()

class GeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_begin(self, epoch, logs=None):
        self.generator.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end(epoch, logs)


def get_generator(descriptor, x):
    print("    CurriculumSets")
    retval = CurriculumSetsGenerator(descriptor, x, conf.K, conf.CURRICULUM_DELAY, conf.CURRICULUM_MAX_NEIGHBOURS)
    retval.initialize()

    tc = GeneratorCallback(retval)
    return retval(), retval, tc



