
import tensorflow as tf


def cosine_similarity(v1, v2, aggregation=True):
    v1_n = tf.nn.l2_normalize(v1, dim=1)
    v2_n = tf.nn.l2_normalize(v2, dim=1)
    if aggregation:
        s = tf.reduce_sum(v1_n * v2_n, axis=1, keep_dims=True)
    else:
        s = v1_n * v2_n
    return s


def dot_product(v1, v2, aggregation=True):
    if aggregation:
        s = tf.reduce_sum(v1 * v2, axis=1, keep_dims=True)
    else:
        s = v1 * v2
    return s


def euclidean_distance(v1, v2, aggregation=True):
    if aggregation:
        s = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1, keep_dims=True))
    else:
        s = tf.abs(v1 - v2)
    return s


def euclidean_score(v1, v2, aggregation=True):
    s = euclidean_distance(v1, v2, aggregation)
    return 1. / (1. + s)


def canberra_score(v1, v2, aggregation=True):
    if aggregation:
        s = tf.reduce_sum(tf.abs(v1 - v2) / (v1 + v2), axis=1, keep_dims=True)
    else:
        s = tf.abs(v1 - v2) / (v1 + v2)
    return s