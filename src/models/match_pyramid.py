
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


    def _interaction_feature_layer(self, enc_seq_left, enc_seq_right, dpool_index, granularity="word"):
        cross = tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right)
        cross = tf.expand_dims(cross, axis=3)
        cross_conv = tf.layers.conv2d(
            inputs=cross,
            filters=self.params["mp_num_filters"],
            kernel_size=self.params["mp_filter_sizes"],
            padding="same",
            activation=tf.nn.relu,
            strides=1,
            reuse=False,
            name=self.model_name+"cross_conv_%s" % granularity)
        if self.params["mp_dynamic_pooling"]:
            cross_conv = tf.gather_nd(cross_conv, dpool_index)
        pool_size = self.params["mp_pool_size_%s" % granularity]
        cross_pool = tf.layers.max_pooling2d(
            inputs=cross_conv,
            pool_size=[self.params["max_seq_len_%s" % granularity] / pool_size,
                       self.params["max_seq_len_%s" % granularity] / pool_size],
            strides=[self.params["max_seq_len_%s" % granularity] / pool_size,
                     self.params["max_seq_len_%s" % granularity] / pool_size],
            padding="valid",
            name=self.model_name+"cross_pool_%s" % granularity)

        cross = tf.reshape(cross_pool, [-1, self.params["mp_num_filters"] * (pool_size * pool_size)])

        return cross


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
        super(MatchPyramid, self).__init__(params, logger, init_embedding_matrix)


    def _build_model(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])

            with tf.name_scope("word_network"):
                sem_seq_word_left, enc_seq_word_left = self._semantic_feature_layer(self.seq_word_left, granularity="word", reuse=False, return_enc=True)
                sem_seq_word_right, enc_seq_word_right = self._semantic_feature_layer(self.seq_word_right, granularity="word", reuse=True, return_enc=True)
                cross_word = self._interaction_feature_layer(enc_seq_word_left, enc_seq_word_right, self.dpool_index_word, granularity="word")

            with tf.name_scope("char_network"):
                sem_seq_char_left, enc_seq_char_left = self._semantic_feature_layer(self.seq_char_left, granularity="char", reuse=False, return_enc=True)
                sem_seq_char_right, enc_seq_char_right = self._semantic_feature_layer(self.seq_char_right, granularity="char", reuse=True, return_enc=True)
                cross_char = self._interaction_feature_layer(enc_seq_char_left, enc_seq_char_right,
                                                             self.dpool_index_char, granularity="char")

            with tf.name_scope("prediction"):
                out_0 = tf.concat([cross_word, cross_char], axis=-1)
                out = self._mlp_layer(out_0, fc_type=self.params["fc_type"],
                                      hidden_units=self.params["fc_hidden_units"],
                                      dropouts=self.params["fc_dropouts"],
                                      scope_name=self.model_name + "mlp",
                                      reuse=False)
                logits = tf.layers.dense(out, 1, activation=None,
                                         kernel_initializer=tf.glorot_uniform_initializer(
                                             seed=self.params["random_seed"]),
                                         name=self.model_name + "logits")
                logits = tf.squeeze(logits, axis=1)
                proba = tf.nn.sigmoid(logits)

        return logits, proba