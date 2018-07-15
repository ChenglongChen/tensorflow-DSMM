
from copy import copy
import tensorflow as tf

from models.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
from models.esim import ESIMDecAttBaseModel
from models.match_pyramid import MatchPyramidBaseModel
from tf_common import metrics
from tf_common.nn_module import mlp_layer


class DSMM(MatchPyramidBaseModel, ESIMDecAttBaseModel, BCNN):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        p["model_name"] = p["model_name"] + "dsmm"
        super(DSMM, self).__init__(p, logger, init_embedding_matrix)


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

                #### matching
                # match score
                sim_word = tf.concat([
                    metrics.cosine_similarity(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    metrics.dot_product(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    metrics.euclidean_distance(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    # self._canberra_score(sem_seq_word_left, sem_seq_word_right),
                ], axis=-1)

                # match pyramid
                match_matrix_word = self._get_match_matrix(self.seq_word_left, emb_seq_word_left, enc_seq_word_left,
                                                           self.seq_word_right, emb_seq_word_right, enc_seq_word_right,
                                                           granularity="word")
                mp_word = self._mp_semantic_feature_layer(match_matrix_word,
                                                          self.dpool_index_word,
                                                          granularity="word")

                # esim
                esim_word = self._esim_semantic_feature_layer(emb_seq_word_left,
                                                              emb_seq_word_right,
                                                              self.seq_len_word_left,
                                                              self.seq_len_word_right,
                                                              granularity="word")

                # bcnn
                bcnn_word = self._bcnn_semantic_feature_layer(emb_seq_word_left,
                                                              emb_seq_word_right,
                                                              granularity="word")

                # dense
                deep_in_word = tf.concat([sem_seq_word_left, sem_seq_word_right], axis=-1)
                deep_word = mlp_layer(deep_in_word, fc_type=self.params["fc_type"],
                                      hidden_units=self.params["fc_hidden_units"],
                                      dropouts=self.params["fc_dropouts"],
                                      scope_name=self.model_name + "deep_word",
                                      reuse=False,
                                      training=self.training,
                                      seed=self.params["random_seed"])

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

                # match score
                sim_char = tf.concat([
                    metrics.cosine_similarity(sem_seq_char_left, sem_seq_char_right, self.params["similarity_aggregation"]),
                    metrics.dot_product(sem_seq_char_left, sem_seq_char_right, self.params["similarity_aggregation"]),
                    metrics.euclidean_distance(sem_seq_char_left, sem_seq_char_right, self.params["similarity_aggregation"]),
                    # self._canberra_score(sem_seq_char_left, sem_seq_char_right),
                ], axis=-1)

                # match pyramid
                match_matrix_char = self._get_match_matrix(self.seq_char_left, emb_seq_char_left, enc_seq_char_left,
                                                           self.seq_char_right, emb_seq_char_right, enc_seq_char_right,
                                                           granularity="char")
                mp_char = self._mp_semantic_feature_layer(match_matrix_char,
                                                          self.dpool_index_char,
                                                          granularity="char")

                # esim
                esim_char = self._esim_semantic_feature_layer(emb_seq_char_left,
                                                              emb_seq_char_right,
                                                              self.seq_len_char_left,
                                                              self.seq_len_char_right,
                                                              granularity="char")

                # bcnn
                bcnn_char = self._bcnn_semantic_feature_layer(emb_seq_char_left,
                                                              emb_seq_char_right,
                                                              granularity="char")

                # dense
                deep_in_char = tf.concat([sem_seq_char_left, sem_seq_char_right], axis=-1)
                deep_char = mlp_layer(deep_in_char, fc_type=self.params["fc_type"],
                                      hidden_units=self.params["fc_hidden_units"],
                                      dropouts=self.params["fc_dropouts"],
                                      scope_name=self.model_name + "deep_char",
                                      reuse=False,
                                      training=self.training,
                                      seed=self.params["random_seed"])

            with tf.name_scope("matching_features"):
                matching_features_word = tf.concat([
                    sim_word, mp_word, esim_word, bcnn_word, deep_word,# sem_seq_word_left, sem_seq_word_right,
                ], axis=-1)
                matching_features_char = tf.concat([
                    sim_char, mp_char, esim_char, bcnn_char, deep_char,# sem_seq_char_left, sem_seq_char_right,
                ], axis=-1)

        return matching_features_word, matching_features_char
