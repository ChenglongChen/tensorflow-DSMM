
import tensorflow as tf

from models.bcnn import ABCNN3
from models.match_pyramid import MatchPyramidBaseModel


class DSMM(MatchPyramidBaseModel, ABCNN3):
    def __init__(self, params, logger, init_embedding_matrix=None):
        super(DSMM, self).__init__(params, logger, init_embedding_matrix)


    def _build_model(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])

            with tf.name_scope("word_network"):
                sem_seq_word_left, enc_seq_word_left = self._semantic_feature_layer(self.seq_word_left, granularity="word", reuse=False, return_enc=True)
                sem_seq_word_right, enc_seq_word_right = self._semantic_feature_layer(self.seq_word_right, granularity="word", reuse=True, return_enc=True)

                #### matching
                # cosine similarity
                # sem_seq_word_left = tf.nn.l2_normalize(sem_seq_word_left, dim=1)
                # sem_seq_word_right = tf.nn.l2_normalize(sem_seq_word_right, dim=1)
                sim_word = sem_seq_word_left * sem_seq_word_right

                # diff
                diff_word = tf.square(sem_seq_word_left - sem_seq_word_right)

                # fm
                tmp = tf.concat([enc_seq_word_left, enc_seq_word_right], axis=1)
                sum_squared = tf.square(tf.reduce_sum(tmp, axis=1))
                squared_sum = tf.reduce_sum(tf.square(tmp), axis=1)
                fm_word = 0.5 * (sum_squared - squared_sum)

                # match pyramid
                mp_word = self._interaction_feature_layer(enc_seq_word_left, enc_seq_word_right, self.dpool_index_word, granularity="word")

                # abcnn
                abcnn_word = super(ABCNN3, self)._interaction_feature_layer(enc_seq_word_left, enc_seq_word_right, granularity="word")

                # dense
                deep_in_word = tf.concat([sem_seq_word_left, sem_seq_word_right], axis=-1)
                deep_word = self._mlp_layer(deep_in_word, fc_type=self.params["fc_type"],
                                            hidden_units=self.params["fc_hidden_units"],
                                            dropouts=self.params["fc_dropouts"],
                                            scope_name=self.model_name + "deep_word",
                                            reuse=False)

            with tf.name_scope("char_network"):
                sem_seq_char_left, enc_seq_char_left = self._semantic_feature_layer(self.seq_char_left, granularity="char", reuse=False, return_enc=True)
                sem_seq_char_right, enc_seq_char_right = self._semantic_feature_layer(self.seq_char_right, granularity="char", reuse=True, return_enc=True)

                #### matching
                # cosine similarity
                # sem_seq_char_left = tf.nn.l2_normalize(sem_seq_char_left, dim=1)
                # sem_seq_char_right = tf.nn.l2_normalize(sem_seq_char_right, dim=1)
                sim_char = sem_seq_char_left * sem_seq_char_right

                # diff
                diff_char = tf.square(sem_seq_char_left - sem_seq_char_right)

                # fm
                tmp = tf.concat([enc_seq_char_left, enc_seq_char_right], axis=1)
                sum_squared = tf.square(tf.reduce_sum(tmp, axis=1))
                squared_sum = tf.reduce_sum(tf.square(tmp), axis=1)
                fm_char = 0.5 * (sum_squared - squared_sum)

                # match pyramid
                mp_char = self._interaction_feature_layer(enc_seq_char_left, enc_seq_char_right,
                                                             self.dpool_index_char,
                                                             granularity="char")

                # abcnn
                abcnn_char = super(ABCNN3, self)._interaction_feature_layer(enc_seq_char_left, enc_seq_char_right,
                                                                                  granularity="char")

                # dense
                deep_in_char = tf.concat([sem_seq_char_left, sem_seq_char_right], axis=-1)
                deep_char = self._mlp_layer(deep_in_char, fc_type=self.params["fc_type"],
                                            hidden_units=self.params["fc_hidden_units"],
                                            dropouts=self.params["fc_dropouts"],
                                            scope_name=self.model_name + "deep_char",
                                            reuse=False)

            with tf.name_scope("prediction"):
                out_0 = tf.concat([
                    sim_word, diff_word, mp_word, abcnn_word, deep_word,
                    sim_char, diff_char, mp_char, abcnn_char, deep_char,
                ], axis=-1)
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