
import numpy as np


# see https://github.com/pl8787/MatchPyramid-TensorFlow
def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
    stride1 = 1.0 * max_len1 / len1_one
    stride2 = 1.0 * max_len2 / len2_one
    idx1_one = np.arange(max_len1) / stride1
    idx2_one = np.arange(max_len2) / stride2
    mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
    index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
    return index_one


def dynamic_pooling_index(len1, len2, max_len1, max_len2):
    index = np.zeros((len(len1), max_len1, max_len2, 3), dtype=int)
    for i in range(len(len1)):
        index[i] = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
    return index
