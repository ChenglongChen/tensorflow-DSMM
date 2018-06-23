
import time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from utils import os_utils
from tf_common.nadam import NadamOptimizer
from tf_common.nn_module import word_dropout, dense_block, resnet_block
from tf_common.nn_module import encode, attend


class BaseModel(object):
    def __init__(self, model_name, params, logger, threshold, calibration_factor, training=True,
                 word_embedding_matrix=None, char_embedding_matrix=None):
        self.model_name = model_name
        self.params = params
        self.logger = logger
        self.threshold = threshold
        self.calibration_factor = calibration_factor
        self.word_embedding_matrix = word_embedding_matrix
        self.char_embedding_matrix = char_embedding_matrix
        self.calibration_model = None
        os_utils._makedirs(self.params["offline_model_dir"], force=training)

        self._init_tf_vars()
        self.loss, self.logits, self.proba = self._build_model()
        self.train_op = self._get_train_op()

        self.sess, self.saver = self._init_session()
        
        
    def _init_tf_vars(self):
        #### training flag
        self.training = tf.placeholder(tf.bool, shape=[], name="training")
        #### labels
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        #### word
        self.seq_word_left = tf.placeholder(tf.int32, shape=[None, None], name="seq_word_left")
        self.seq_word_right = tf.placeholder(tf.int32, shape=[None, None], name="seq_word_right")
        #### char
        self.seq_char_left = tf.placeholder(tf.int32, shape=[None, None], name="seq_char_left")
        self.seq_char_right = tf.placeholder(tf.int32, shape=[None, None], name="seq_char_right")
        #### word len
        self.seq_len_word_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_left")
        self.seq_len_word_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_right")
        #### char len
        self.seq_len_char_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_left")
        self.seq_len_char_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_right")

        #### training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                        self.params["decay_steps"], self.params["decay_rate"])


    def _semantic_feature_layer(self, seq_input, granularity="word", reuse=False, return_enc=False):
        assert granularity in ["char", "word"]
        #### embed
        if granularity == "word":
            if self.word_embedding_matrix is None:
                std = 0.1
                minval = -std
                maxval = std
                emb_matrix = tf.Variable(
                    tf.random_uniform([self.params["max_num_%s"%granularity] + 1, self.params["embedding_dim_%s"%granularity]],
                                      minval, maxval,
                                      seed=self.params["random_seed"],
                                      dtype=tf.float32))
            else:
                emb_matrix = tf.Variable(self.word_embedding_matrix,
                                              trainable=self.params["embedding_trainable"])
        elif granularity == "char":
            if self.char_embedding_matrix is None:
                std = 0.1
                minval = -std
                maxval = std
                emb_matrix = tf.Variable(
                    tf.random_uniform([self.params["max_num_%s"%granularity] + 1, self.params["embedding_dim_%s"%granularity]],
                                      minval, maxval,
                                      seed=self.params["random_seed"],
                                      dtype=tf.float32))
            else:
                emb_matrix = tf.Variable(self.char_embedding_matrix,
                                              trainable=self.params["embedding_trainable"])

        emb_seq = tf.nn.embedding_lookup(emb_matrix, seq_input)

        #### dropout
        emb_seq = word_dropout(emb_seq,
                                         training=self.training,
                                         dropout=self.params["embedding_dropout"],
                                         seed=self.params["random_seed"])

        #### encode
        enc_seq = encode(emb_seq, method=self.params["encode_method"], params=self.params,
                                   scope_name=self.model_name + "enc_seq_%s"%granularity, reuse=reuse)

        #### attend
        feature_dim = self.params["encode_dim"]
        context = None
        att_seq = attend(enc_seq, context=context, feature_dim=feature_dim,
                                   method=self.params["attend_method"],
                                   scope_name=self.model_name + "att_seq_%s"%granularity,
                                   reuse=reuse)

        #### MLP nonlinear projection
        hidden_units = self.params["fc_hidden_units"]
        dropouts = self.params["fc_dropouts"]
        if self.params["fc_type"] == "fc":
            sem_seq = dense_block(att_seq, hidden_units=hidden_units, dropouts=dropouts,
                                             densenet=False, scope_name=self.model_name + "sem_seq_%s"%granularity,
                                             reuse=reuse,
                                             training=self.training, seed=self.params["random_seed"])
        elif self.params["fc_type"] == "densenet":
            sem_seq = dense_block(att_seq, hidden_units=hidden_units, dropouts=dropouts,
                                             densenet=True, scope_name=self.model_name + "sem_seq_%s"%granularity,
                                             reuse=reuse,
                                             training=self.training, seed=self.params["random_seed"])
        elif self.params["fc_type"] == "resnet":
            sem_seq = resnet_block(att_seq, hidden_units=hidden_units, dropouts=dropouts,
                                              cardinality=1, dense_shortcut=True, training=self.training,
                                              reuse=reuse,
                                              seed=self.params["random_seed"],
                                              scope_name=self.model_name + "sem_seq_%s"%granularity)
        if return_enc:
            return sem_seq, enc_seq
        else:
            return sem_seq


    def _cosine_similarity(self, v1, v2):
        v1_n = tf.nn.l2_normalize(v1, dim=1)
        v2_n = tf.nn.l2_normalize(v2, dim=1)
        s = tf.reduce_sum(v1_n * v2_n, axis=1, keep_dims=True)
        return s


    def _euclidean_score(self, v1, v2):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
        return 1. / (1. + euclidean)


    def _build_model(self):
        pass


    def _get_train_op(self):
        with tf.name_scope("optimization"):
            if self.params["optimizer_type"] == "nadam":
                optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                beta2=self.params["beta2"], epsilon=1e-8,
                                                schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                        beta2=self.params["beta2"], epsilon=1e-8)
    
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        return train_op


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 1})
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver


    def save_session(self):
        self.saver.save(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def restore_session(self):
        self.saver.restore(self.sess, self.params["offline_model_dir"] + "/model.checkpoint")


    def _get_batch_index(self, seq, step):
        n = len(seq)
        res = []
        for i in range(0, n, step):
            res.append(seq[i:i + step])
        # last batch
        if len(res) * step < n:
            res.append(seq[len(res) * step:])
        return res


    def _get_pos_neg_ind(self, label):
        length = len(label)
        pos_ind_tmp = np.where(label == 1)[0]
        inds = np.zeros((len(pos_ind_tmp) * length, 2), dtype=int)
        inds[:, 0] = np.repeat(pos_ind_tmp, length)
        inds[:, 1] = list(range(length)) * len(pos_ind_tmp)
        mask = inds[:, 0] != inds[:, 1]
        pos_ind = inds[mask, 0]
        neg_ind = inds[mask, 1]
        return pos_ind, neg_ind


    def _get_feed_dict(self, X, idx, Q, construct_neg=False, training=False, symmetric=False):
        if training:

            if construct_neg:
                q1 = X["q1"][idx]
                q2 = X["q2"][idx]
                # for label=1 sample, construct negative sample within batch
                pos_ind, neg_ind = self._get_pos_neg_ind(X["label"][idx])
                # original & symmetric
                feed_dict = {
                    self.seq_word_left: np.vstack([Q["words"][q1],
                                                   Q["words"][X["q1"][idx[pos_ind]]],
                                                   Q["words"][X["q1"][idx[neg_ind]]],
                                                   Q["words"][q2],
                                                   Q["words"][X["q2"][idx[neg_ind]]],
                                                   Q["words"][X["q2"][idx[pos_ind]]]
                                                   ]),
                    self.seq_word_right: np.vstack([Q["words"][q2],
                                                    Q["words"][X["q2"][idx[neg_ind]]],
                                                    Q["words"][X["q2"][idx[pos_ind]]],
                                                    Q["words"][q1],
                                                    Q["words"][X["q1"][idx[pos_ind]]],
                                                    Q["words"][X["q1"][idx[neg_ind]]],
                                                    ]),
                    self.seq_char_left: np.vstack([Q["chars"][q1],
                                                   Q["chars"][X["q1"][idx[pos_ind]]],
                                                   Q["chars"][X["q1"][idx[neg_ind]]],
                                                   Q["chars"][q2],
                                                   Q["chars"][X["q2"][idx[neg_ind]]],
                                                   Q["chars"][X["q2"][idx[pos_ind]]]
                                                   ]),
                    self.seq_char_right: np.vstack([Q["chars"][q2],
                                                    Q["chars"][X["q2"][idx[neg_ind]]],
                                                    Q["chars"][X["q2"][idx[pos_ind]]],
                                                    Q["chars"][q1],
                                                    Q["chars"][X["q1"][idx[pos_ind]]],
                                                    Q["chars"][X["q1"][idx[neg_ind]]]
                                                    ]),
                    self.labels: np.hstack([X["label"][idx],
                                            np.zeros(len(pos_ind)),
                                            np.zeros(len(pos_ind)),
                                            X["label"][idx],
                                            np.zeros(len(pos_ind)),
                                            np.zeros(len(pos_ind))
                                            ]),
                    self.training: training,
                }
            else:
                q1 = X["q1"][idx]
                q2 = X["q2"][idx]
                feed_dict = {
                    self.seq_word_left: np.vstack([Q["words"][q1],
                                                   Q["words"][q2],
                                                   ]),
                    self.seq_word_right: np.vstack([Q["words"][q2],
                                                    Q["words"][q1],
                                                    ]),
                    self.seq_char_left: np.vstack([Q["chars"][q1],
                                                   Q["chars"][q2],
                                                   ]),
                    self.seq_char_right: np.vstack([Q["chars"][q2],
                                                    Q["chars"][q1],
                                                    ]),
                    self.seq_len_word_left: np.hstack([Q["sequence_length_word"][q1],
                                                       Q["sequence_length_word"][q2],
                                                       ]),
                    self.seq_len_word_right: np.hstack([Q["sequence_length_word"][q2],
                                                        Q["sequence_length_word"][q1],
                                                        ]),
                    self.seq_len_char_left: np.hstack([Q["sequence_length_char"][q1],
                                                       Q["sequence_length_char"][q2],
                                                       ]),
                    self.seq_len_char_right: np.hstack([Q["sequence_length_char"][q2],
                                                        Q["sequence_length_char"][q1],
                                                        ]),
                    self.labels: np.hstack([X["label"][idx],
                                            X["label"][idx],
                                            ]),
                    self.training: training,
                }
        elif not symmetric:
            q1 = X["q1"][idx]
            q2 = X["q2"][idx]
            feed_dict = {
                self.seq_word_left: Q["words"][q1],
                self.seq_word_right: Q["words"][q2],
                self.seq_char_left: Q["chars"][q1],
                self.seq_char_right: Q["chars"][q2],
                self.seq_len_word_left: Q["sequence_length_word"][q1],
                self.seq_len_word_right: Q["sequence_length_word"][q2],
                self.seq_len_char_left: Q["sequence_length_char"][q1],
                self.seq_len_char_right: Q["sequence_length_char"][q2],
                self.labels: X["label"][idx],
                self.training: training,
            }
        else:
            q1 = X["q1"][idx]
            q2 = X["q2"][idx]
            feed_dict = {
                self.seq_word_left: np.vstack([Q["words"][q1],
                                               Q["words"][q2],
                                               ]),
                self.seq_word_right: np.vstack([Q["words"][q2],
                                                Q["words"][q1],
                                                ]),
                self.seq_char_left: np.vstack([Q["chars"][q1],
                                               Q["chars"][q2],
                                               ]),
                self.seq_char_right: np.vstack([Q["chars"][q2],
                                                Q["chars"][q1],
                                                ]),
                self.seq_len_word_left: np.hstack([Q["sequence_length_word"][q1],
                                                   Q["sequence_length_word"][q2],
                                                   ]),
                self.seq_len_word_right: np.hstack([Q["sequence_length_word"][q2],
                                                    Q["sequence_length_word"][q1],
                                                    ]),
                self.seq_len_char_left: np.hstack([Q["sequence_length_char"][q1],
                                                   Q["sequence_length_char"][q2],
                                                   ]),
                self.seq_len_char_right: np.hstack([Q["sequence_length_char"][q2],
                                                    Q["sequence_length_char"][q1],
                                                    ]),
                self.labels: np.hstack([X["label"][idx],
                                        X["label"][idx],
                                        ]),
                self.training: training,
            }
        return feed_dict


    def fit(self, X, Q, validation_data, shuffle=False):
        start_time = time.time()
        l = X["label"].shape[0]
        self.logger.info("fit on %d sample" % l)
        self.logger.info("mean: %.5f"%np.mean(validation_data["label"]))
        train_idx_shuffle = np.arange(l)
        total_loss = 0.
        loss_decay = 0.9
        total_batch = 0
        for epoch in range(self.params["epoch"]):
            self.logger.info("epoch: %d" % (epoch + 1))
            np.random.seed(epoch)
            if shuffle:
                np.random.shuffle(train_idx_shuffle)
            batches = self._get_batch_index(train_idx_shuffle, self.params["batch_size"])
            for i, idx in enumerate(batches):
                feed_dict = self._get_feed_dict(X, idx, Q, construct_neg=self.params["construct_neg"], training=True)
                loss, lr, opt = self.sess.run((self.loss, self.learning_rate, self.train_op), feed_dict=feed_dict)
                total_loss = loss_decay * total_loss + (1. - loss_decay) * loss
                total_batch += 1
                if validation_data is not None and total_batch % self.params["eval_every_num_update"] == 0:
                    y_valid = validation_data["label"]
                    y_proba = self.predict_proba(validation_data, Q)
                    valid_loss = log_loss(y_valid, y_proba, eps=1e-15)
                    # y_proba_cal = self.predict_calibration_proba(validation_data, Q)
                    y_proba_cal = y_proba
                    valid_loss_cal = log_loss(y_valid, y_proba_cal, eps=1e-15)
                    self.logger.info(
                        "[epoch-%d, batch-%d] train-loss=%.5f, valid-loss=%.5f, valid-loss-cal=%.5f, valid-proba=%.5f, predict-proba=%.5f, predict-proba-cal=%.5f, lr=%.5f [%.1f s]" % (
                            epoch + 1, total_batch, total_loss, valid_loss, valid_loss_cal,
                            np.mean(y_valid), np.mean(y_proba), np.mean(y_proba_cal), lr, time.time() - start_time))
                else:
                    self.logger.info("[epoch-%d, batch-%d] train-loss=%.5f, lr=%.5f [%.1f s]" % (
                        epoch + 1, total_batch, total_loss,
                        lr, time.time() - start_time))


    def predict_calibration_proba(self, X, Q):
        y_logit = self.predict_logit(X, Q)
        y_valid = X["label"]
        if self.calibration_model is None:
            self.calibration_model = LogisticRegression()
            self.calibration_model.fit(y_logit, y_valid)
        y_proba = self.calibration_model.predict_proba(y_logit)
        return y_proba


    def predict_logit(self, X, Q):
        l = X["label"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx, Q, training=False, symmetric=True)
            pred = self.sess.run((self.logits), feed_dict=feed_dict)
            n = int(pred.shape[0]/2)
            pred = (pred[:n] + pred[n:])/2.
            y_pred_append(pred)
        y_pred = np.hstack(y_pred).reshape((-1, 1)).astype(np.float64)
        return y_pred


    def predict_proba(self, X, Q):
        l = X["label"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx, Q, training=False, symmetric=True)
            pred = self.sess.run((self.proba), feed_dict=feed_dict)
            n = int(pred.shape[0] / 2)
            pred = (pred[:n] + pred[n:]) / 2.
            y_pred_append(pred)
        y_pred = np.hstack(y_pred).reshape((-1, 1)).astype(np.float64)
        return y_pred


    def predict(self, X, Q):
        proba = self.predict_proba(X, Q)
        y = np.array(proba > self.threshold, dtype=int)
        return y
