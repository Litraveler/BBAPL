import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def radians_to_degrees(radians):
    return 180 / math.pi * radians

# 在高维空间中，标准正态分布的点在归一化后会均匀分布在单位球面上
def generate_spherical_coord_uniform(dimensions, size=1000):
    '''
    在 d 维空间中生成均匀分布的点。
    @param dimensions : 维度数量
    @param size : 样本数量
    @return x : d 维空间中的 size 个点
    '''
    x = np.random.normal(size=(size, dimensions))
    x = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x

def compute_pairwise_distance(x):
    ''' @Deprecated
    使用角度距离代替
    '''
    return np.matmul(x, x.T)

def upper_bound(dimensions, angle):
    s = math.sin(angle)
    factor1 = (1 + s) / (2 * s)
    factor2 = (1 - s) / (2 * s)
    factor = factor1 * np.log2(factor1) - factor2 * np.log2(factor2)
    exp = dimensions * factor
    return math.pow(2, exp)

def skip_diag_strided(A):
    ''' 从矩阵中移除对角线元素的代码，来自 Stack Overflow
    <url id="cvl74ds5rbsfvs4tm1tg" type="url" status="parsed" title="Just a moment..." wc="161">https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array</url>
    '''
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

def angular_distance_custom(y_true, y_pred):
    ''' 自定义角度距离函数
    '''
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)
    norm_true = tf.norm(y_true, axis=-1)
    norm_pred = tf.norm(y_pred, axis=-1)
    cosine_similarity = dot_product / (norm_true * norm_pred)
    angular_distance = tf.acos(tf.clip_by_value(cosine_similarity, -1.0, 1.0))
    return angular_distance


def compute_far_ifo_dimensions(thrs, dim_start=3, dim_end=128, nb_points=10000, batch_size=1000, plotting=False):
    '''
    计算不同维度下的 FAR（False Acceptance Rate）。
    Args:
        thrs: 要考虑的阈值列表
        dim_start: 最低维度
        dim_end: 最高维度（dim_end > dim_start）
        nb_points: 每个维度下生成的点的数量
        batch_size: 每个批次的点的数量
        plotting: 是否绘图
    Returns:
        FAR 的对数值数组，形状为 [dim_end - dim_start, len(thrs)]
    '''
    fars = np.empty((dim_end - dim_start, len(thrs)))
    for dim in range(dim_start, dim_end):
        # 在不同维度下生成固定数量的点
        x = generate_spherical_coord_uniform(dim, nb_points)

        # 分批计算角度距离
        dist = np.zeros((nb_points, nb_points))
        for i in range(0, nb_points, batch_size):
            for j in range(0, nb_points, batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = x[j:j + batch_size]
                batch_dist = angular_distance_custom(batch_x[:, None, :], batch_y[None, :, :]).numpy()
                dist[i:i + batch_size, j:j + batch_size] = batch_dist

        dist = skip_diag_strided(dist)
        dist = dist.ravel()

        for i in range(len(thrs)):
            fp = (dist < thrs[i]).sum()
            far_log = np.log10(np.maximum(fp, 1e-12)) - np.log10(len(dist))
            fars[dim - dim_start, i] = far_log

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(range(dim_start, dim_end), fars)
        ax.legend(np.arccos(1 - thrs) * 180 / np.pi)
        ax.set_xlabel('# dimensions')
        ax.set_ylabel('FAR')
        fig.tight_layout()
        fig.savefig('../../results/figures/far_ifo_dimenions_multiple_thr.png')
    return fars

def convert_to_angle(cosdist):
    return np.arccos(1 - cosdist)

# 示例调用
if __name__ == "__main__":
    thrs = [0.1, 0.2, 0.3]
    fars = compute_far_ifo_dimensions(thrs, dim_start=3, dim_end=10, nb_points=1000, plotting=True)
    print(fars)