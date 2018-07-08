
from copy import copy
import tensorflow as tf

from models.base_model import BaseModel
from tf_common import metrics


class DSSMBaseModel(BaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        super(DSSMBaseModel, self).__init__(params, logger, init_embedding_matrix)


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
                # match score
                sim_word = tf.concat([
                    metrics.cosine_similarity(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    metrics.dot_product(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    metrics.euclidean_distance(sem_seq_word_left, sem_seq_word_right, self.params["similarity_aggregation"]),
                    # self._canberra_score(sem_seq_word_left, sem_seq_word_right),
                ], axis=-1)

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

            with tf.name_scope("matching_features"):
                matching_features_word = sim_word
                matching_features_char = sim_char

        return matching_features_word, matching_features_char


class DSSM(DSSMBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        # model config
        p.update({
            "model_name": p["model_name"] + "dssm",
            "encode_method": "fasttext",
            "attend_method": ["ave", "max", "min", "self-scalar-attention"],

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [64 * 4, 64 * 2, 64],
            "fc_dropouts": [0, 0, 0],
        })
        super(DSSM, self).__init__(p, logger, init_embedding_matrix)


class CDSSM(DSSMBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        # model config
        p.update({
            "model_name": p["model_name"] + "cdssm",
            "encode_method": "textcnn",
            "attend_method": ["ave", "max", "min", "self-scalar-attention"],

            # cnn
            "cnn_num_layers": 1,
            "cnn_num_filters": 32,
            "cnn_filter_sizes": [1, 2, 3],
            "cnn_timedistributed": False,
            "cnn_activation": tf.nn.relu,
            "cnn_gated_conv": False,
            "cnn_residual": False,

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [64 * 4, 64 * 2, 64],
            "fc_dropouts": [0, 0, 0],
        })
        super(CDSSM, self).__init__(p, logger, init_embedding_matrix)


class RDSSM(DSSMBaseModel):
    def __init__(self, params, logger, init_embedding_matrix=None):
        p = copy(params)
        # model config
        p.update({
            "model_name": p["model_name"] + "rdssm",
            "encode_method": "textbirnn",
            "attend_method": ["ave", "max", "min", "self-scalar-attention"],

            # rnn
            "rnn_num_units": 32,
            "rnn_cell_type": "gru",
            "rnn_num_layers": 1,

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [64 * 4, 64 * 2, 64],
            "fc_dropouts": [0, 0, 0],
        })
        super(RDSSM, self).__init__(p, logger, init_embedding_matrix)
