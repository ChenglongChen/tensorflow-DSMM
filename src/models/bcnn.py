
from copy import copy
import tensorflow as tf
import numpy as np

from inputs.dynamic_pooling import dynamic_pooling_index
from models.base_model import BaseModel
from tf_common import metrics


class BCNNBaseModel(BaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(BCNNBaseModel, self).__init__(params, logger, init_embedding_matrix)


    def _init_tf_vars(self):
        super(BCNNBaseModel, self)._init_tf_vars()
        self.dpool_index_word = tf.placeholder(tf.int32, shape=[None, self.params["max_seq_len_word"],
                                                                self.params["max_seq_len_word"], 3],
                                               name="dpool_index_word")
        self.dpool_index_char = tf.placeholder(tf.int32, shape=[None, self.params["max_seq_len_char"],
                                                                self.params["max_seq_len_char"], 3],
                                               name="dpool_index_char")


    def _padding(self, x, name):
        # x: [batch, s, d, 1]
        # x => [batch, s+w*2-2, d, 1]
        w = self.params["bcnn_filter_size"]
        return tf.pad(x, np.array([[0, 0], [w - 1, w - 1], [0, 0], [0, 0]]), "CONSTANT", name)


    def _make_attention_matrix(self, x1, x2):
        # x1: [batch, s1, d, 1]
        # x2: [batch, s2, d, 1]
        # match score
        if "euclidean" in self.params["bcnn_match_score_type"]:
            # x1 => [batch, s1, 1, d]
            # x2 => [batch, 1, s2, d]
            x1_ = tf.transpose(x1, perm=[0, 1, 3, 2])
            x2_ = tf.transpose(x2, perm=[0, 3, 1, 2])
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1_ - x2_), axis=-1))
            if "exp" in self.params["bcnn_match_score_type"]:
                # exp(-euclidean / (2. * beta)) (producenan)
                # from Convolutional Neural Network for Paraphrase Identification
                beta = 2.
                att = tf.exp(-euclidean / (2. * beta))
            else:
                # euclidean distance (produce nan)
                att = 1. / (1. + euclidean)
        elif self.params["bcnn_match_score_type"] == "cosine":
            # cosine similarity
            x1_ = tf.nn.l2_normalize(x1, dim=2)
            x2_ = tf.nn.l2_normalize(x2, dim=2)
            sim = tf.einsum("abcd,aecd->abe", x1_, x2_) # value in [-1, 1]
            att = (1. + sim) / 2. # value in [0, 1]
        return att


    def _convolution(self, x, d, name, reuse=False):
        # conv: [batch, s+w-1, 1, d]
        conv = tf.layers.conv2d(
            inputs=x,
            filters=self.params["bcnn_num_filters"],
            kernel_size=(self.params["bcnn_filter_size"], d),
            padding="valid",
            activation=self.params["bcnn_activation"],
            strides=1,
            reuse=reuse,
            name=name)

        # [batch, s+w-1, d, 1]
        return tf.transpose(conv, perm=[0, 1, 3, 2])


    def _w_ap(self, x, attention, name):
        # x: [batch, s+w-1, d, 1]
        # attention: [batch, s+w-1]
        if attention is not None:
            attention = tf.expand_dims(tf.expand_dims(attention, axis=-1), axis=-1)
            x2 = x * attention
        else:
            x2 = x
        w_ap = tf.layers.average_pooling2d(
            inputs=x2,
            pool_size=(self.params["bcnn_filter_size"], 1),
            strides=1,
            padding="valid",
            name=name)
        if attention is not None:
            w_ap = w_ap * self.params["bcnn_filter_size"]

        return w_ap


    def _all_ap(self, x, seq_len, name):
        if "input" in name:
            pool_width = seq_len
            d = self.params["embedding_dim"]
        else:
            pool_width = seq_len + self.params["bcnn_filter_size"] - 1
            d = self.params["bcnn_num_filters"]

        all_ap = tf.layers.average_pooling2d(
            inputs=x,
            pool_size=(pool_width, 1),
            strides=1,
            padding="valid",
            name=name)
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])

        return all_ap_reshaped


    def _expand_input(self, x1, x2, att_mat, seq_len, d, name):
        # att_mat: [batch, s, s]
        aW = tf.get_variable(name=name, shape=(seq_len, d))

        # [batch, s, s] * [s,d] => [batch, s, d]
        # expand dims => [batch, s, d, 1]
        x1_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", att_mat, aW), -1)
        x2_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW), -1)

        # [batch, s, d, 2]
        x1 = tf.concat([x1, x1_a], axis=3)
        x2 = tf.concat([x2, x2_a], axis=3)

        return x1, x2


    def _bcnn_cnn_layer(self, x1, x2, seq_len, d, name, dpool_index, granularity="word"):
        return None, None, None, None, None


    def _mp_cnn_layer(self, cross, dpool_index, filters, kernel_size, pool_size, strides, name):
        cross_conv = tf.layers.conv2d(
            inputs=cross,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=self.params["bcnn_mp_activation"],
            strides=1,
            reuse=False,
            name=name+"cross_conv")
        if self.params["bcnn_mp_dynamic_pooling"] and dpool_index is not None:
            cross_conv = tf.gather_nd(cross_conv, dpool_index)
        cross_pool = tf.layers.max_pooling2d(
            inputs=cross_conv,
            pool_size=pool_size,
            strides=strides,
            padding="valid",
            name=name+"cross_pool")
        return cross_pool

    def _bcnn_semantic_feature_layer(self, seq_left, seq_right, dpool_index=None, granularity="word"):
        name = self.model_name + granularity
        seq_len = self.params["max_seq_len_%s" % granularity]
        # [batch, s, d] => [batch, s, d, 1]
        seq_left = tf.expand_dims(seq_left, axis=-1)
        seq_right = tf.expand_dims(seq_right, axis=-1)

        left_ap_list = [None] * (self.params["bcnn_num_layers"] + 1)
        right_ap_list = [None] * (self.params["bcnn_num_layers"] + 1)
        left_ap_list[0] = self._all_ap(x=seq_left, seq_len=seq_len, name=name + "global_pooling_input_left")
        right_ap_list[0] = self._all_ap(x=seq_right, seq_len=seq_len, name=name + "global_pooling_input_right")

        x1 = seq_left
        x2 = seq_right
        d = self.params["embedding_dim"]
        outputs = []
        for layer in range(self.params["bcnn_num_layers"]):
            x1, left_ap_list[layer + 1], x2, right_ap_list[layer + 1], att_pooled = self._bcnn_cnn_layer(x1=x1, x2=x2,
                                                                                                         seq_len=seq_len,
                                                                                                         d=d,
                                                                                                         name=name + "cnn_layer_%d" % (
                                                                                                                 layer + 1),
                                                                                                         dpool_index=dpool_index,
                                                                                                         granularity=granularity)
            d = self.params["bcnn_num_filters"]
            if self.params["bcnn_mp_att_pooling"] and att_pooled is not None:
                outputs.append(att_pooled)

        for l, r in zip(left_ap_list, right_ap_list):
            outputs.append(metrics.cosine_similarity(l, r, self.params["similarity_aggregation"]))
            outputs.append(metrics.dot_product(l, r, self.params["similarity_aggregation"]))
            outputs.append(metrics.euclidean_distance(l, r, self.params["similarity_aggregation"]))
        return tf.concat(outputs, axis=-1)


    def _get_attention_matrix_pooled_features(self, att_mat, seq_len, dpool_index, granularity, name):
        # get attention matrix pooled features (as in sec. 5.3.1)
        att_mat0 = tf.expand_dims(att_mat, axis=3)
        # conv-pool layer 1
        filters = self.params["bcnn_mp_num_filters"][0]
        kernel_size = self.params["bcnn_mp_filter_sizes"][0]
        # seq_len = seq_len + self.params["bcnn_filter_size"] - 1
        pool_size0 = self.params["bcnn_mp_pool_sizes_%s" % granularity][0]
        pool_sizes = [seq_len / pool_size0, seq_len / pool_size0]
        strides = [seq_len / pool_size0, seq_len / pool_size0]
        conv1 = self._mp_cnn_layer(att_mat0, dpool_index, filters, kernel_size, pool_sizes, strides,
                                   name=self.model_name + name + granularity + "1")
        conv1_flatten = tf.reshape(conv1, [-1, self.params["mp_num_filters"][0] * (pool_size0 * pool_size0)])

        # conv-pool layer 2
        filters = self.params["bcnn_mp_num_filters"][1]
        kernel_size = self.params["bcnn_mp_filter_sizes"][1]
        pool_size1 = self.params["bcnn_mp_pool_sizes_%s" % granularity][1]
        pool_sizes = [pool_size0 / pool_size1, pool_size0 / pool_size1]
        strides = [pool_size0 / pool_size1, pool_size0 / pool_size1]
        conv2 = self._mp_cnn_layer(conv1, None, filters, kernel_size, pool_sizes, strides,
                                   name=self.model_name + name + granularity + "2")
        conv2_flatten = tf.reshape(conv2, [-1, self.params["mp_num_filters"][1] * (pool_size1 * pool_size1)])

        return conv2_flatten


    def _get_feed_dict(self, X, idx, Q, construct_neg=False, training=False, symmetric=False):
        feed_dict = super(BCNNBaseModel, self)._get_feed_dict(X, idx, Q, construct_neg, training, symmetric)
        if self.params["mp_dynamic_pooling"]:
            dpool_index_word = dynamic_pooling_index(feed_dict[self.seq_len_word_left],
                                                          feed_dict[self.seq_len_word_right],
                                                          self.params["max_seq_len_word"],
                                                          self.params["max_seq_len_word"])
            dpool_index_char = dynamic_pooling_index(feed_dict[self.seq_len_char_left],
                                                          feed_dict[self.seq_len_char_right],
                                                          self.params["max_seq_len_char"],
                                                          self.params["max_seq_len_char"])
            feed_dict.update({
                self.dpool_index_word: dpool_index_word,
                self.dpool_index_char: dpool_index_char,
            })
        return feed_dict


    def _get_matching_features(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])

            with tf.name_scope("word_network"):
                if self.params["attend_method"] == "context-attention":
                    emb_seq_word_left, enc_seq_word_left, att_seq_word_left, sem_seq_word_left, \
                    emb_seq_word_right, enc_seq_word_right, att_seq_word_right, sem_seq_word_right = \
                        self._interaction_semantic_feature_layer(
                            self.seq_word_left,
                            self.seq_word_right,
                            self.seq_len_word_left,
                            self.seq_len_word_right,
                            granularity="word")
                else:
                    emb_seq_word_left, enc_seq_word_left, att_seq_word_left, sem_seq_word_left = \
                        self._semantic_feature_layer(
                            self.seq_word_left,
                            self.seq_len_word_left,
                            granularity="word", reuse=False)
                    emb_seq_word_right, enc_seq_word_right, att_seq_word_right, sem_seq_word_right = \
                        self._semantic_feature_layer(
                            self.seq_word_right,
                            self.seq_len_word_right,
                            granularity="word", reuse=True)
                sim_word = self._bcnn_semantic_feature_layer(emb_seq_word_left, emb_seq_word_right, self.dpool_index_word, granularity="word")

            with tf.name_scope("char_network"):
                if self.params["attend_method"] == "context-attention":
                    emb_seq_char_left, enc_seq_char_left, att_seq_char_left, sem_seq_char_left, \
                    emb_seq_char_right, enc_seq_char_right, att_seq_char_right, sem_seq_char_right = \
                        self._interaction_semantic_feature_layer(
                            self.seq_char_left,
                            self.seq_char_right,
                            self.seq_len_char_left,
                            self.seq_len_char_right,
                            granularity="char")
                else:
                    emb_seq_char_left, enc_seq_char_left, att_seq_char_left, sem_seq_char_left = \
                        self._semantic_feature_layer(
                            self.seq_char_left,
                            self.seq_len_char_left,
                            granularity="char", reuse=False)
                    emb_seq_char_right, enc_seq_char_right, att_seq_char_right, sem_seq_char_right = \
                        self._semantic_feature_layer(
                            self.seq_char_right,
                            self.seq_len_char_right,
                            granularity="char", reuse=True)
                sim_char = self._bcnn_semantic_feature_layer(emb_seq_char_left, emb_seq_char_right, self.dpool_index_char, granularity="char")

            with tf.name_scope("matching_features"):
                matching_features_word = sim_word
                matching_features_char = sim_char

        return matching_features_word, matching_features_char


class BCNN(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        p = copy(params)
        p["model_name"] = p["model_name"] + "bcnn"
        super(BCNN, self).__init__(p, logger, init_embedding_matrix)


    def _bcnn_cnn_layer(self, x1, x2, seq_len, d, name, dpool_index=None, granularity="word"):
        # x1, x2 = [batch, s, d, 1]
        # att_mat0: [batch, s, s]
        att_mat0 = self._make_attention_matrix(x1, x2)
        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        left_attention, right_attention = None, None

        left_wp = self._w_ap(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._all_ap(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._w_ap(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._all_ap(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        # get attention matrix pooled features (as in sec. 5.3.1)
        att_mat0_pooled = self._get_attention_matrix_pooled_features(att_mat0, seq_len, dpool_index, granularity, name+"att_pooled")

        return left_wp, left_ap, right_wp, right_ap, att_mat0_pooled


class ABCNN1(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        p = copy(params)
        p["model_name"] = p["model_name"] + "abcnn1"
        super(ABCNN1, self).__init__(p, logger, init_embedding_matrix)


    def _bcnn_cnn_layer(self, x1, x2, seq_len, d, name, dpool_index=None, granularity="word"):
        # x1, x2 = [batch, s, d, 1]
        # att_mat0: [batch, s, s]
        att_mat0 = self._make_attention_matrix(x1, x2)
        x1, x2 = self._expand_input(x1, x2, att_mat0, seq_len, d, name=name+"expand_input")

        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        left_attention, right_attention = None, None

        left_wp = self._w_ap(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._all_ap(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._w_ap(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._all_ap(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        # get attention matrix pooled features (as in sec. 5.3.1)
        att_mat0_pooled = self._get_attention_matrix_pooled_features(att_mat0, seq_len, dpool_index, granularity, name+"att_pooled")

        return left_wp, left_ap, right_wp, right_ap, att_mat0_pooled


class ABCNN2(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        p = copy(params)
        p["model_name"] = p["model_name"] + "abcnn2"
        super(ABCNN2, self).__init__(p, logger, init_embedding_matrix)


    def _bcnn_cnn_layer(self, x1, x2, seq_len, d, name, dpool_index=None, granularity="word"):
        # x1, x2 = [batch, s, d, 1]
        att_mat0 = self._make_attention_matrix(x1, x2)
        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        # [batch, s+w-1, s+w-1]
        att_mat1 = self._make_attention_matrix(left_conv, right_conv)
        # [batch, s+w-1], [batch, s+w-1]
        left_attention, right_attention = tf.reduce_sum(att_mat1, axis=2), tf.reduce_sum(att_mat1, axis=1)

        left_wp = self._w_ap(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._all_ap(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._w_ap(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._all_ap(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        # get attention matrix pooled features (as in sec. 5.3.1)
        att_mat0_pooled = self._get_attention_matrix_pooled_features(att_mat0, seq_len, dpool_index, granularity, name+"att_pooled")

        return left_wp, left_ap, right_wp, right_ap, att_mat0_pooled


class ABCNN3(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        p = copy(params)
        p["model_name"] = p["model_name"] + "abcnn3"
        super(ABCNN3, self).__init__(p, logger, init_embedding_matrix)


    def _bcnn_cnn_layer(self, x1, x2, seq_len, d, name, dpool_index=None, granularity="word"):
        # x1, x2 = [batch, s, d, 1]
        # att_mat0: [batch, s, s
        att_mat0 = self._make_attention_matrix(x1, x2)
        x1, x2 = self._expand_input(x1, x2, att_mat0, seq_len, d, name=name + "expand_input")

        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        # [batch, s+w-1, s+w-1]
        att_mat1 = self._make_attention_matrix(left_conv, right_conv)
        # [batch, s+w-1], [batch, s+w-1]
        left_attention, right_attention = tf.reduce_sum(att_mat1, axis=2), tf.reduce_sum(att_mat1, axis=1)

        left_wp = self._w_ap(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._all_ap(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._w_ap(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._all_ap(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        # get attention matrix pooled features (as in sec. 5.3.1)
        att_mat0_pooled = self._get_attention_matrix_pooled_features(att_mat0, seq_len, dpool_index, granularity, name+"att_pooled")

        return left_wp, left_ap, right_wp, right_ap, att_mat0_pooled
