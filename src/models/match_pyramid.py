
import numpy as np
import tensorflow as tf

from models.base_model import BaseModel
from tf_common.nn_module import dense_block, resnet_block


class MatchPyramidBaseModel(BaseModel):
    def __init__(self, model_name, params, logger, threshold, calibration_factor, training=True,
                 word_embedding_matrix=None, char_embedding_matrix=None):
        super(MatchPyramidBaseModel, self).__init__(model_name, params, logger, threshold, calibration_factor, training,
                                            word_embedding_matrix, char_embedding_matrix)


    def _init_tf_vars(self):
        super(MatchPyramidBaseModel, self)._init_tf_vars()
        self.dpool_index_word = tf.placeholder(tf.int32, shape=[None, self.params["max_sequence_length_word"],
                                                                self.params["max_sequence_length_word"], 3],
                                               name='dpool_index_word')
        self.dpool_index_char = tf.placeholder(tf.int32, shape=[None, self.params["max_sequence_length_char"],
                                                                self.params["max_sequence_length_char"], 3],
                                               name='dpool_index_char')


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
            name="cross_conv_%s" % granularity)
        # dynamic pooling
        cross_conv_expand = tf.gather_nd(cross_conv, dpool_index)
        pool_size = self.params["mp_pool_size_%s" % granularity]
        cross_pool = tf.layers.max_pooling2d(
            inputs=cross_conv_expand,
            pool_size=[self.params["max_sequence_length_%s" % granularity] / pool_size,
                       self.params["max_sequence_length_%s" % granularity] / pool_size],
            strides=[self.params["max_sequence_length_%s" % granularity] / pool_size,
                     self.params["max_sequence_length_%s" % granularity] / pool_size],
            padding='valid',
            name="cross_pool_%s" % granularity)

        cross = tf.reshape(cross_pool, [-1, self.params["mp_num_filters"] * (pool_size * pool_size)])

        return cross


    # see https://github.com/pl8787/MatchPyramid-TensorFlow
    def _dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = (np.arange(max_len1)/stride1).astype(int)
            idx2_one = (np.arange(max_len2)/stride2).astype(int)
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        index_append = index.append
        for i in range(len(len1)):
            index_append(dpool_index_(i, len1[i], len2[i]))
        return np.array(index)


    def _get_feed_dict(self, X, idx, Q, construct_neg=False, training=False, symmetric=False):
        feed_dict = super(MatchPyramidBaseModel, self)._get_feed_dict(X, idx, Q, construct_neg, training, symmetric)
        dpool_index_word = self._dynamic_pooling_index(feed_dict[self.seq_len_word_left],
                                                      feed_dict[self.seq_len_word_right],
                                                      self.params["max_sequence_length_word"],
                                                      self.params["max_sequence_length_word"])
        dpool_index_char = self._dynamic_pooling_index(feed_dict[self.seq_len_char_left],
                                                      feed_dict[self.seq_len_char_right],
                                                      self.params["max_sequence_length_char"],
                                                      self.params["max_sequence_length_char"])
        feed_dict.update({
            self.dpool_index_word: dpool_index_word,
            self.dpool_index_char: dpool_index_char,
        })
        return feed_dict


class MatchPyramid(MatchPyramidBaseModel):
    def __init__(self, model_name, params, logger, threshold, calibration_factor, training=True,
                 word_embedding_matrix=None, char_embedding_matrix=None):
        super(MatchPyramid, self).__init__(model_name, params, logger, threshold, calibration_factor, training,
                                            word_embedding_matrix, char_embedding_matrix)


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
                hidden_units = self.params["fc_hidden_units"]
                dropouts = self.params["fc_dropouts"]
                if self.params["fc_type"] == "fc":
                    out = dense_block(out_0, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                                      scope_name=self.model_name + "mlp", reuse=False, training=self.training,
                                      seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    out = dense_block(out_0, hidden_units=hidden_units, dropouts=dropouts, densenet=True,
                                      scope_name=self.model_name + "mlp", reuse=False, training=self.training,
                                      seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    out = resnet_block(out_0, hidden_units=hidden_units, dropouts=dropouts, cardinality=1,
                                       dense_shortcut=True, training=self.training,
                                       seed=self.params["random_seed"], scope_name=self.model_name + "mlp", reuse=False)
                logits = tf.layers.dense(out, 1, activation=None,
                                         kernel_initializer=tf.glorot_uniform_initializer(
                                             seed=self.params["random_seed"]),
                                         name=self.model_name + "logits")
                logits = tf.squeeze(logits, axis=1)
                proba = tf.nn.sigmoid(logits)

            with tf.name_scope("loss"):
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
                loss = tf.reduce_mean(loss, name="log_loss")
                if self.params["l2_lambda"] > 0:
                    l2_losses = tf.add_n(
                        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.params[
                                    "l2_lambda"]
                    loss = loss + l2_losses

        return loss, logits, proba