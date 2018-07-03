
from copy import copy
import tensorflow as tf

from models.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
from models.esim import ESIMBaseModel
from models.match_pyramid import MatchPyramidBaseModel


class DSMM(MatchPyramidBaseModel, ESIMBaseModel, BCNN):
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
                # cosine similarity
                if self.params["similarity_aggregation"]:
                    sim_word = tf.concat([
                        self._cosine_similarity(sem_seq_word_left, sem_seq_word_right),
                        self._euclidean_distance(sem_seq_word_left, sem_seq_word_right),
                        # self._canberra_score(sem_seq_word_left, sem_seq_word_right),
                    ], axis=-1)
                else:
                    sim_word = tf.concat([
                        sem_seq_word_left * sem_seq_word_right,
                        tf.abs(sem_seq_word_left - sem_seq_word_right),
                        # tf.abs(sem_seq_word_left - sem_seq_word_right) / (sem_seq_word_left + sem_seq_word_right),
                    ], axis=-1)

                # match pyramid
                match_matrix_word = self._get_match_matrix(self.seq_word_left, emb_seq_word_left, enc_seq_word_left,
                                                           self.seq_word_right, emb_seq_word_right, enc_seq_word_right,
                                                           granularity="word")
                mp_word = self._interaction_feature_layer(match_matrix_word, self.dpool_index_word,
                                                             granularity="word")

                # esim
                esim_word = super(ESIMBaseModel, self)._interaction_semantic_feature_layer(self.seq_word_left,
                                                                                           self.seq_word_right,
                                                                                           self.seq_len_word_left,
                                                                                           self.seq_len_word_right,
                                                                                           granularity="word")

                # abcnn
                abcnn_word = super(BCNN, self)._interaction_feature_layer(emb_seq_word_left, emb_seq_word_right, granularity="word")

                # dense
                deep_in_word = tf.concat([sem_seq_word_left, sem_seq_word_right], axis=-1)
                deep_word = self._mlp_layer(deep_in_word, fc_type=self.params["fc_type"],
                                            hidden_units=self.params["fc_hidden_units"],
                                            dropouts=self.params["fc_dropouts"],
                                            scope_name=self.model_name + "deep_word",
                                            reuse=False)

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

                # cosine similarity
                if self.params["similarity_aggregation"]:
                    sim_char = tf.concat([
                        self._cosine_similarity(sem_seq_char_left, sem_seq_char_right),
                        self._euclidean_distance(sem_seq_char_left, sem_seq_char_right),
                        # self._canberra_score(sem_seq_char_left, sem_seq_char_right),
                    ], axis=-1)
                else:
                    sim_char = tf.concat([
                        sem_seq_char_left * sem_seq_char_right,
                        tf.abs(sem_seq_char_left - sem_seq_char_right),
                        # tf.abs(sem_seq_char_left - sem_seq_char_right) / (sem_seq_char_left + sem_seq_char_right),
                    ], axis=-1)

                # match pyramid
                match_matrix_char = self._get_match_matrix(self.seq_char_left, emb_seq_char_left, enc_seq_char_left,
                                                           self.seq_char_right, emb_seq_char_right, enc_seq_char_right,
                                                           granularity="char")
                mp_char = self._interaction_feature_layer(match_matrix_char, self.dpool_index_char,
                                                             granularity="char")

                # esim
                esim_char = super(ESIMBaseModel, self)._interaction_semantic_feature_layer(self.seq_char_left, self.seq_char_right,
                                                                                           self.seq_len_char_left, self.seq_len_char_right,
                                                                                           granularity="char")

                # abcnn
                abcnn_char = super(BCNN, self)._interaction_feature_layer(emb_seq_char_left, emb_seq_char_right,
                                                                                  granularity="char")

                # dense
                deep_in_char = tf.concat([sem_seq_char_left, sem_seq_char_right], axis=-1)
                deep_char = self._mlp_layer(deep_in_char, fc_type=self.params["fc_type"],
                                            hidden_units=self.params["fc_hidden_units"],
                                            dropouts=self.params["fc_dropouts"],
                                            scope_name=self.model_name + "deep_char",
                                            reuse=False)

            with tf.name_scope("matching_features"):
                matching_features = tf.concat([
                    sim_word, mp_word, esim_word, abcnn_word, deep_word,# sem_seq_word_left, sem_seq_word_right,
                    sim_char, mp_char, esim_char, abcnn_char, deep_char,# sem_seq_char_left, sem_seq_char_right,
                ], axis=-1)

        return matching_features