import  os;
import  tensorflow_addons   as tfa;

import  sys;
import  tensorflow                  as      tf;
from    tensorflow.keras.losses     import  Loss;
from    tensorflow_addons.losses    import  metric_learning;

import  conf;

# @tf.function 是 TensorFlow 2.x 的一个装饰器，用来把Python 函数编译成TensorFlow 计算图（Graph）
# 大幅加速执行（尤其是大量矩阵运算）支持自动并行、XLA 编译、跨设备部署  天然开启图优化（常量折叠、算子融合等）
@tf.function
def calculate_set2set_loss(y_true, y_pred, K, p, beta, margin) -> tf.Tensor:
    labels = tf.convert_to_tensor(y_true, name="labels")
    # 网络输出的embedding特征向量
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")
    tf.print("\n[Loss] embedding shape:", tf.shape(embeddings), 
             "mean:", tf.reduce_mean(embeddings), 
             "std:", tf.math.reduce_std(embeddings),
             "max:", tf.reduce_max(embeddings),
             "min:", tf.reduce_min(embeddings),
             "margin:", margin,
             output_stream=sys.stdout)

    # 精度处理，中间计算统一成 float32 保证数值稳定
    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    );
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    );

    # 两两计算欧式距离矩阵 K*N
    pdist_matrix = metric_learning.pairwise_distance(
        precise_embeddings, squared=False
    );

    retval = 0.0;
    total_pairs = 0;

    for i in range(0, K - 1):
        for j in range(i + 1, K):
            posA = conf.N * i;
            posB = conf.N * j;
            # 集合A的内部距离并在最后维度扩展一维
            eA = tf.expand_dims(pdist_matrix[posA:posA + conf.N,posA: posA + conf.N],axis=-1);
            # 后面要跟“跨集合距离”逐元素相减，需要三维对齐。
            eA = tf.tile(eA, [1, 1, conf.N]);
            # 集合A和集合B的距离
            eB = tf.expand_dims(pdist_matrix[posA:posA + conf.N:,posB:posB + conf.N],axis=1);
            # 在第二维拓展
            eB = tf.tile(eB, [1, conf.N, 1]);
            # 内部距离 − 跨集合距离 + margin（动态margin）
            m = eA - eB + margin;
            m = tf.maximum(m, 0);
            # p为掩码，只在 i<j 的位置为 1，避免重复惩罚。
            m = tf.multiply(m,p);
            #把所有违反量累加得到 l，再累加到总 loss retval。
            l = tf.reduce_sum(m);
                
            retval += l;
            total_pairs += 1;
      
    retval /= conf.N * conf.N * (conf.N - 1) / 2;
    retval /= total_pairs;
    
    # 半径惩罚项，让同一集合的样本尽量紧密分布
    total_radius = 0.0;
    for i in range(0, K):
        # 取出属于集合 i 的 N 个向量
        legitimate_embeddings = precise_embeddings[conf.N * i:conf.N * i + conf.N]

        # ====================== 核心功能：统计每个向量的0数量并写入txt ======================
        # 计算：conf.N 个向量，每个向量中 0 的数量
        zero_counts = tf.reduce_sum(tf.cast(tf.equal(legitimate_embeddings, 0.0), tf.int32), axis=1)

        tf.print(
            f"\n===== 集合 {i} | 共 {conf.N} 个向量 =====",
            output_stream="file://zero_counts.txt",
            end=""
        )
        # 遍历每个向量的0数量，逐行打印（写入文件）
        for idx in tf.range(tf.shape(zero_counts)[0]):
            tf.print(
                zero_counts[idx],
                output_stream="file://zero_counts.txt",
                summarize=-1
            )
        # ==================================================================================
        
        # 求这 N 个向量的中心（平均向量）
        centroid = tf.reduce_mean(legitimate_embeddings, axis=0);
        # 每个样本到中心的 L2 距离，形状 [N]
        distances = tf.norm(legitimate_embeddings - centroid, axis=1)
        mean_distance = tf.reduce_mean(distances);
        total_radius += mean_distance;

    total_radius /= K;
    total_penalty = 0.0;
    for i in range(0, K):
        legitimate_embeddings = precise_embeddings[conf.N * i:conf.N * i + conf.N];
        centroid = tf.reduce_mean(legitimate_embeddings, axis=0);
        distances = tf.norm(legitimate_embeddings - centroid, axis=1)
        mean_distance = tf.reduce_mean(distances);
        total_penalty += tf.math.abs(mean_distance / total_radius - 1.0);
    
    total_penalty /= K;
    total_penalty *= beta;

    tf.print(tf.strings.format("[Set2Set]    K={}    beta={}    Lsm={}    Lrp={}", [K, beta, retval, total_penalty]), output_stream=sys.stdout);
    return retval + total_penalty;



class Set2SetLoss(Loss):
    def __init__(self, K, beta):
        super(Set2SetLoss, self).__init__()
        self.K = K;
        self.beta = beta;
        self.current_epoch = 0;  # 当前训练轮次

        # 掩码正方体[N,N,N]
        p = tf.zeros([conf.N, conf.N, conf.N], dtype=float)

        for i in range(0,conf.N):
          for j in range(i + 1,conf.N):
            for k in range(0,conf.N):
              position = [i, j, k]
              new_value = 1.0
              index = tf.constant([position])
              update = tf.constant([new_value])
              p = tf.tensor_scatter_nd_update(p, index, update);

        self.p = p;

    def _get_dynamic_margin(self):
        """
        根据训练轮次动态计算margin：
        - epoch < 50:   margin = 0.5
        - 50 <= epoch < 100: margin = 0.45
        - 100 <= epoch < 150: margin = 0.4
        - 150 <= epoch < 200: margin = 0.35
        - epoch >= 200: margin = 0.3
        每50轮减少0.05
        """
        if self.current_epoch < 50:
            return 0.2
        elif self.current_epoch < 100:
            return 0.15
        elif self.current_epoch < 150:
            return 0.1
        elif self.current_epoch < 200:
            return 0.08
        else:
            return 0.05

    def reset_epoch_cache(self, epoch):
        """重置epoch缓存，供回调函数调用"""
        self.current_epoch = epoch
        current_margin = self._get_dynamic_margin()
        print(f"\n[Set2SetLoss] 当前epoch: {epoch}, 使用margin: {current_margin}")

    def call(self, y_true, y_pred):
        margin = self._get_dynamic_margin()
        return calculate_set2set_loss(y_true, y_pred, self.K, self.p, self.beta, margin);


def get_loss():
    print("    Set2Set (beta=" + str(conf.BETA) + ") - 动态margin模式");
    return Set2SetLoss(conf.K, conf.BETA);

