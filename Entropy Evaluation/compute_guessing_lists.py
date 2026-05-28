import tensorflow as tf
import numpy as np
from scipy.sparse import coo_array, csr_array

@tf.function
def angular_distance_2tensors(feature1: tf.types.experimental.TensorLike, feature2: tf.types.experimental.TensorLike):
    """Computes the angular distance matrix.
    output[i, j] = 1 - cosine_similarity(feature[i, :], feature[j, :])
    Args:
      feature1: 2-D Tensor of size `[number of data, feature dimension]`.
      feature2: 2-D Tensor of size `[number of data, feature dimension]`.
    Returns:
      angular_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    # normalize input
    feature1 = tf.math.l2_normalize(feature1, axis=1)
    feature2 = tf.math.l2_normalize(feature2, axis=1)

    # create adjacency matrix of cosine similarity, the value range is 0-2
    angular_distances = 1 - tf.matmul(feature1, feature2, transpose_b=True)

    # ensure all distances > 1e-16
    angular_distances = tf.maximum(angular_distances, 1e-16)
    count_greater_than_0_1 = tf.reduce_sum(tf.cast(angular_distances > 0.1, tf.int32))

    tf.print("Count of values greater than 0.1:", count_greater_than_0_1)

    return angular_distances


def compute_coo_arr(centroids, query_templates, thr):
    '''
    The threshold

    Args :
        centroids: [num_centroids, template_length]
        query_templates: [num_query_templates, template_length]
        thr: the threshold
    Retruns:
        coo_arr: [num_query_templates, num_centroids]
        If the distance between the heart templates is less than the threshold thr,
           the corresponding position in the sparse matrix is marked as 1, otherwise it is not marked.
    '''
    centroids = tf.convert_to_tensor(centroids, name='centroids')

    # Batch processing query template
    step_size = 1000
    row_indx = []
    col_indx = []
    shape = 0
    for i in range(0, query_templates.shape[0], step_size):
        qt = tf.convert_to_tensor(query_templates[i*step_size : (i+1)*step_size], name='query_templates')

        # compute distance
        pdist = angular_distance_2tensors(qt, centroids).numpy()

        # compute the sparse array
        indexes = (pdist < thr).nonzero()
        row_indx.append(indexes[0]+i*step_size)
        col_indx.append(indexes[1])
        shape += len(indexes[0])

    coo_arr = coo_array(
            (np.ones((shape,), dtype=np.int8),
            (np.concatenate(row_indx), np.concatenate(col_indx)))
            )

    return coo_arr

def compute_guessing_list(csr_arr):
    '''
    Compute a guessing list, the guessing list is used to compute the guessing metrics and contains
    the successrate of each guess from most successfull to least successfull.
    WARNING: we do not return the success rate, but the number of accounts broken
    Args:
        coo_arr: a sparse array in csr format of shape [query templates (gueses), number of centroids]
    Returns:
        guessing list: a list with the number of acounts broke in each guess (should be normalized to compute guessing metrics!)
    '''
    results = []
    mmax = 2 #just an initialization value
    while mmax > 0:
        ssum = csr_arr.sum(axis=1)
        mmax = np.max(ssum)
        argmmax = np.argmax(ssum)
        results.append(mmax) # append the current maximum number of broken centroids

        inv_csr_arr = csr_array(np.ones(csr_arr.getrow(argmmax).get_shape()), dtype=np.int8)\
                        - csr_arr.getrow(argmmax)
        csr_arr = csr_arr * inv_csr_arr
        csr_arr.eliminate_zeros()

    results = results[:-1] # delete last 0
    return results
