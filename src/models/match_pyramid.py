
from copy import copy
import tensorflow as tf

from inputs.dynamic_pooling import dynamic_pooling_index
from models.base_model import BaseModel


class MatchPyramidBaseModel(BaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        super(MatchPyramidBaseModel, self).__init__(params, logger, init_embedding_matrix)


    def _init_tf_vars(self):
        super(MatchPyramidBaseModel, self)._init_tf_vars()
        self.dpool_index_word = tf.placeholder(tf.int32, shape=[None, self.params["max_seq_len_word"],
                                                                self.params["max_seq_len_word"], 3],
                                               name="dpool_index_word")
        self.dpool_index_char = tf.placeholder(tf.int32, shape=[None, self.params["max_seq_len_char"],
                                                                self.params["max_seq_len_char"], 3],
                                               name="dpool_index_char")


    def _get_match_matrix(self, seq_left, emb_seq_left, enc_seq_left, seq_right, emb_seq_right, enc_seq_right,
                          granularity="word"):
        # 1. word embedding
        # 1.1 dot product: [batchsize, s1, s2, 1]
        match_matrix_dot_product = tf.expand_dims(
            tf.einsum("abd,acd->abc", emb_seq_left, emb_seq_right), axis=-1)
        # 1.2 identity: [batchsize, s1, s2, 1]
        match_matrix_identity = tf.expand_dims(tf.cast(
            tf.equal(
                tf.expand_dims(seq_left, 2),
                tf.expand_dims(seq_right, 1)
            ), tf.float32), axis=-1)

        # 2. compressed word embedding
        eW = tf.get_variable("eW_%s" % (self.model_name + granularity),
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32),
                             dtype=tf.float32,
                             shape=[self.params["embedding_dim_%s" % granularity],
                                    self.params["embedding_dim_compressed"]])
        emb_seq_com_left = tf.einsum("abd,dc->abc", emb_seq_left, eW)
        emb_seq_com_right = tf.einsum("abd,dc->abc", emb_seq_right, eW)
        # 2.1 dot product: [batchsize, s1, s2, 1]
        match_matrix_dot_product_com = tf.expand_dims(
            tf.einsum("abd,acd->abc", emb_seq_com_left, emb_seq_com_right), axis=-1)
        # 2.2 element product: [batchsize, s1, s2, d]
        match_matrix_element_product_com = tf.expand_dims(emb_seq_com_left, 2) * tf.expand_dims(
            emb_seq_com_right, 1)
        # 2.3 element concat: [batchsize, s1, s2, 2*d]
        match_matrix_element_concat_com = tf.concat([
            tf.tile(tf.expand_dims(emb_seq_com_left, 2), [1, 1, self.params["max_seq_len_%s" % granularity], 1]),
            tf.tile(tf.expand_dims(emb_seq_com_right, 1), [1, self.params["max_seq_len_%s" % granularity], 1, 1]),
        ], axis=-1)

        # 3. contextual word embedding
        # 3.1 dot product: [batchsize, s1, s2, 1]
        match_matrix_dot_product_ctx = tf.expand_dims(
            tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right), axis=-1)
        # 2.2 element product: [batchsize, s1, s2, d]
        match_matrix_element_product_ctx = tf.expand_dims(enc_seq_left, 2) * tf.expand_dims(
            enc_seq_right, 1)
        # 2.3 element concat: [batchsize, s1, s2, 2*d]
        match_matrix_element_concat_ctx = tf.concat([
            tf.tile(tf.expand_dims(enc_seq_left, 2), [1, 1, self.params["max_seq_len_%s" % granularity], 1]),
            tf.tile(tf.expand_dims(enc_seq_right, 1), [1, self.params["max_seq_len_%s" % granularity], 1, 1]),
        ], axis=-1)

        match_matrix = tf.concat([
            match_matrix_dot_product,
            match_matrix_identity,
            match_matrix_dot_product_com,
            match_matrix_element_product_com,
            match_matrix_element_concat_com,
            match_matrix_dot_product_ctx,
            match_matrix_element_product_ctx,
            match_matrix_element_concat_ctx,
        ], axis=-1)
        return match_matrix


    def _mp_cnn_layer(self, cross, dpool_index, filters, kernel_size, pool_size, strides, name):
        cross_conv = tf.layers.conv2d(
            inputs=cross,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=self.params["mp_activation"],
            strides=1,
            reuse=False,
            name=name+"cross_conv")
        if self.params["mp_dynamic_pooling"] and dpool_index is not None:
            cross_conv = tf.gather_nd(cross_conv, dpool_index)
        cross_pool = tf.layers.max_pooling2d(
            inputs=cross_conv,
            pool_size=pool_size,
            strides=strides,
            padding="valid",
            name=name+"cross_pool")
        return cross_pool


    def _mp_semantic_feature_layer(self, match_matrix, dpool_index, granularity="word"):

        # conv-pool layer 1
        filters = self.params["mp_num_filters"][0]
        kernel_size = self.params["mp_filter_sizes"][0]
        seq_len = self.params["max_seq_len_%s" % granularity]
        pool_size0 = self.params["mp_pool_sizes_%s" % granularity][0]
        pool_sizes = [seq_len / pool_size0, seq_len / pool_size0]
        strides = [seq_len / pool_size0, seq_len / pool_size0]
        conv1 = self._mp_cnn_layer(match_matrix, dpool_index, filters, kernel_size, pool_sizes, strides, name=self.model_name+granularity+"1")
        conv1_flatten = tf.reshape(conv1, [-1, self.params["mp_num_filters"][0] * (pool_size0 * pool_size0)])

        # conv-pool layer 2
        filters = self.params["mp_num_filters"][1]
        kernel_size = self.params["mp_filter_sizes"][1]
        pool_size1 = self.params["mp_pool_sizes_%s" % granularity][1]
        pool_sizes = [pool_size0 / pool_size1, pool_size0 / pool_size1]
        strides = [pool_size0 / pool_size1, pool_size0 / pool_size1]
        conv2 = self._mp_cnn_layer(conv1, None, filters, kernel_size, pool_sizes, strides, name=self.model_name + granularity + "2")
        conv2_flatten = tf.reshape(conv2, [-1, self.params["mp_num_filters"][1] * (pool_size1 * pool_size1)])

        # cross = tf.concat([conv1_flatten, conv2_flatten], axis=-1)

        return conv2_flatten


    def _get_feed_dict(self, X, idx, Q, construct_neg=False, training=False, symmetric=False):
        feed_dict = super(MatchPyramidBaseModel, self)._get_feed_dict(X, idx, Q, construct_neg, training, symmetric)
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


class MatchPyramid(MatchPyramidBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        p["model_name"] = p["model_name"] + "match_pyramid"
        super(MatchPyramid, self).__init__(p, logger, init_embedding_matrix)


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
                match_matrix_word = tf.einsum("abd,acd->abc", emb_seq_word_left, emb_seq_word_right)
                match_matrix_word = tf.expand_dims(match_matrix_word, axis=-1)
                sim_word = self._mp_semantic_feature_layer(match_matrix_word, self.dpool_index_word,
                                                             granularity="word")

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
                match_matrix_char = tf.einsum("abd,acd->abc", emb_seq_char_left, emb_seq_char_right)
                match_matrix_char = tf.expand_dims(match_matrix_char, axis=-1)
                sim_char = self._mp_semantic_feature_layer(match_matrix_char, self.dpool_index_char,
                                                             granularity="char")
            with tf.name_scope("matching_features"):
                matching_features_word = sim_word
                matching_features_char = sim_char

        return matching_features_word, matching_features_char


class GMatchPyramid(MatchPyramidBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        # model config
        p.update({
            "model_name": p["model_name"] + "g_match_pyramid",
            "encode_method": "textcnn",
            "attend_method": ["ave", "max", "min", "self-attention"],

            # cnn
            "cnn_num_layers": 1,
            "cnn_num_filters": 32,
            "cnn_filter_sizes": [1, 2, 3],
            "cnn_timedistributed": False,
            "cnn_activation": tf.nn.relu,
            "cnn_gated_conv": True,
            "cnn_residual": True,

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [64 * 4, 64 * 2, 64],
            "fc_dropouts": [0, 0, 0],
        })
        super(GMatchPyramid, self).__init__(p, logger, init_embedding_matrix)


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

                match_matrix_word = self._get_match_matrix(self.seq_word_left, emb_seq_word_left, enc_seq_word_left,
                                                           self.seq_word_right, emb_seq_word_right, enc_seq_word_right,
                                                           granularity="word")
                sim_word = self._mp_semantic_feature_layer(match_matrix_word, self.dpool_index_word, granularity="word")

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

                match_matrix_char = self._get_match_matrix(self.seq_char_left, emb_seq_char_left, enc_seq_char_left,
                                                           self.seq_char_right, emb_seq_char_right, enc_seq_char_right,
                                                           granularity="char")
                sim_char = self._mp_semantic_feature_layer(match_matrix_char, self.dpool_index_char,
                                                             granularity="char")

            with tf.name_scope("matching_features"):
                matching_features_word = sim_word
                matching_features_char = sim_char

        return matching_features_word, matching_features_char
