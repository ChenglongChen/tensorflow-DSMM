
import tensorflow as tf

from models.base_model import BaseModel


class DSSMBaseModel(BaseModel):
    def __init__(self, model_name, params, logger, threshold=0.5, calibration_factor=1., training=True,
                 init_embedding_matrix=None):
        super(DSSMBaseModel, self).__init__(model_name, params, logger, threshold, calibration_factor, training,
                                            init_embedding_matrix)


    def _build_model(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])

            with tf.name_scope("word_network"):
                sem_seq_word_left = self._semantic_feature_layer(self.seq_word_left, granularity="word", reuse=False)
                sem_seq_word_right = self._semantic_feature_layer(self.seq_word_right, granularity="word", reuse=True)
                sim_word = self._cosine_similarity(sem_seq_word_left, sem_seq_word_right)

            with tf.name_scope("char_network"):
                sem_seq_char_left = self._semantic_feature_layer(self.seq_char_left, granularity="char", reuse=False)
                sem_seq_char_right = self._semantic_feature_layer(self.seq_char_right, granularity="char", reuse=True)
                sim_char = self._cosine_similarity(sem_seq_char_left, sem_seq_char_right)

            with tf.name_scope("prediction"):
                out = tf.concat([sim_word, sim_char], axis=-1)
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
                        [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * self.params[
                                    "l2_lambda"]
                    loss = loss + l2_losses

        return loss, logits, proba


class DSSM(DSSMBaseModel):
    def __init__(self, model_name, params, logger, threshold=0.5, calibration_factor=1., training=True,
                 init_embedding_matrix=None):
        # model config
        params.update({
            "encode_method": "fasttext",
            "attend_method": "ave",

            # embedding dim
            "embedding_dim_word": 300,
            "embedding_dim_char": 300,
            "embedding_dim": 300,

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [300, 128],
            "fc_dropouts": [0, 0],
        })
        super(DSSM, self).__init__(model_name, params, logger, threshold, calibration_factor, training,
                                            init_embedding_matrix)


class CDSSM(DSSMBaseModel):
    def __init__(self, model_name, params, logger, threshold=0.5, calibration_factor=1., training=True,
                 init_embedding_matrix=None):
        # model config
        params.update({
            "encode_method": "textcnn",
            "attend_method": "max",

            # cnn
            "cnn_num_filters": 300,
            "cnn_filter_sizes": [3],
            "cnn_timedistributed": False,

            # fc block
            "fc_type": "fc",
            "fc_hidden_units": [128],
            "fc_dropouts": [0],
        })
        super(CDSSM, self).__init__(model_name, params, logger, threshold, calibration_factor, training,
                                            init_embedding_matrix)

