
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

import utils
from nadam import NadamOptimizer
from nn_module import word_dropout, dense_block, resnet_block
from nn_module import encode, attend


class BaseModel(object):
    def __init__(self, model_name, params, logger, threshold, training=True):
        self.model_name = model_name
        self.params = params
        self.logger = logger
        self.threshold = threshold
        utils._makedirs(self.params["offline_model_dir"], force=training)

        self._init_tf_vars()
        self.loss, self.proba = self._build_model()
        # self.loss = self._get_loss()
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
        #### pinyin
        self.seq_pinyin_left = tf.placeholder(tf.int32, shape=[None, None], name="seq_pinyin_left")
        self.seq_pinyin_right = tf.placeholder(tf.int32, shape=[None, None], name="seq_pinyin_right")
        
        #### training
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.params["init_lr"], self.global_step,
                                                        self.params["decay_steps"], self.params["decay_rate"])


    def _build_model(self):
        return None, None


    def _get_loss(self):
        with tf.name_scope("loss"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(loss, name="loss")
        return loss


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


    def _get_feed_dict(self, X, idx, training=False, symmetric=False):
        if training:
            # for label=1 sample, construct negative sample within batch
            pos_ind, neg_ind = self._get_pos_neg_ind(X["label"][idx])

            # original & symmetric
            feed_dict = {
                self.seq_word_left: np.vstack([X["seq_word_left"][idx],
                                               X["seq_word_left"][idx[pos_ind]],
                                               X["seq_word_left"][idx[neg_ind]],
                                               X["seq_word_right"][idx],
                                               X["seq_word_right"][idx[neg_ind]],
                                               X["seq_word_right"][idx[pos_ind]]
                                               ]),
                self.seq_word_right: np.vstack([X["seq_word_right"][idx],
                                                X["seq_word_right"][idx[neg_ind]],
                                                X["seq_word_right"][idx[pos_ind]],
                                                X["seq_word_left"][idx],
                                                X["seq_word_left"][idx[pos_ind]],
                                                X["seq_word_left"][idx[neg_ind]],
                                                ]),
                self.seq_char_left: np.vstack([X["seq_char_left"][idx],
                                                X["seq_char_left"][idx[pos_ind]],
                                                X["seq_char_left"][idx[neg_ind]],
                                               X["seq_char_right"][idx],
                                               X["seq_char_right"][idx[neg_ind]],
                                               X["seq_char_right"][idx[pos_ind]]
                                               ]),
                self.seq_char_right: np.vstack([X["seq_char_right"][idx],
                                               X["seq_char_right"][idx[neg_ind]],
                                               X["seq_char_right"][idx[pos_ind]],
                                                X["seq_char_left"][idx],
                                                X["seq_char_left"][idx[pos_ind]],
                                                X["seq_char_left"][idx[neg_ind]]
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
        elif not symmetric:
            feed_dict = {
                self.seq_word_left: X["seq_word_left"][idx],
                self.seq_word_right: X["seq_word_right"][idx],
                self.seq_char_left: X["seq_char_left"][idx],
                self.seq_char_right: X["seq_char_right"][idx],
                self.labels: X["label"][idx],
                self.training: training,
            }
        else:
            feed_dict = {
                self.seq_word_left: X["seq_word_right"][idx],
                self.seq_word_right: X["seq_word_left"][idx],
                self.seq_char_left: X["seq_char_right"][idx],
                self.seq_char_right: X["seq_char_left"][idx],
                self.labels: X["label"][idx],
                self.training: training,
            }
        return feed_dict


    def fit(self, X, validation_data, shuffle=False):
        start_time = time.time()
        l = X["label"].shape[0]
        self.logger.info("fit on %d sample" % l)
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
                feed_dict = self._get_feed_dict(X, idx, training=True)
                loss, lr, opt = self.sess.run((self.loss, self.learning_rate, self.train_op), feed_dict=feed_dict)
                total_loss = loss_decay * total_loss + (1. - loss_decay) * loss
                total_batch += 1
                if total_batch % self.params["eval_every_num_update"] == 0:
                    y_proba = self.predict_proba(validation_data)
                    y_valid = validation_data["label"]
                    precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)
                    f1 = 2. * (precision * recall) / (precision + recall)
                    f1[precision + recall == 0] = 0
                    self.threshold = thresholds[np.argmax(f1)]
                    f1 = np.max(f1)
                    self.logger.info(
                        "[epoch-%d, batch-%d] train-loss=%.5f, valid-f1=%.5f, lr=%.5f, threshold=%.5f [%.1f s]" % (
                            epoch + 1, total_batch, total_loss, f1,
                            lr, self.threshold, time.time() - start_time))
                else:
                    self.logger.info("[epoch-%d, batch-%d] train-loss=%.5f, lr=%.5f [%.1f s]" % (
                        epoch + 1, total_batch, total_loss,
                        lr, time.time() - start_time))


    def predict_proba(self, X):
        l = X["label"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size"])
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            # original
            feed_dict = self._get_feed_dict(X, idx, training=False, symmetric=False)
            pred_1 = self.sess.run((self.proba), feed_dict=feed_dict)
            # symmetric
            feed_dict = self._get_feed_dict(X, idx, training=False, symmetric=True)
            pred_2 = self.sess.run((self.proba), feed_dict=feed_dict)
            pred = (pred_1 + pred_2) / 2.
            y_pred_append(pred)
        y_pred = np.hstack(y_pred).reshape((-1, 1))
        return y_pred


    def predict(self, X):
        proba = self.predict_proba(X)
        y = np.array(proba > self.threshold, dtype=int)
        return y


class SemanticMatchingModel(BaseModel):
    def __init__(self, model_name, params, logger, threshold, training=True):
        super(SemanticMatchingModel, self).__init__(model_name, params, logger, threshold, training)

    def _build_model(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])
            
            with tf.name_scope("word_network"):
                #### embed
                std = 0.1
                minval = -std
                maxval = std
                emb_word_matrix = tf.Variable(
                    tf.random_uniform([self.params["max_num_words"] + 1, self.params["embedding_word_dim"]],
                                      minval, maxval,
                                      seed=self.params["random_seed"],
                                      dtype=tf.float32))

                emb_seq_word_left = tf.nn.embedding_lookup(emb_word_matrix, self.seq_word_left)
                emb_seq_word_right = tf.nn.embedding_lookup(emb_word_matrix, self.seq_word_right)

                #### dropout
                emb_seq_word_left = word_dropout(emb_seq_word_left,
                                                 training=self.training,
                                                 dropout=self.params["embedding_dropout"],
                                                seed=self.params["random_seed"])
                emb_seq_word_right = word_dropout(emb_seq_word_right,
                                                  training=self.training,
                                                  dropout=self.params["embedding_dropout"],
                                                  seed=self.params["random_seed"])

                #### encode
                enc_seq_word_left = encode(emb_seq_word_left, method=self.params["encode_method"], params=self.params,
                                           scope_name=self.model_name + "enc_seq_word", reuse=False)
                enc_seq_word_right = encode(emb_seq_word_right, method=self.params["encode_method"], params=self.params,
                                            scope_name=self.model_name + "enc_seq_word", reuse=True)

                #### attend
                feature_dim = self.params["encode_dim"]
                context = tf.reduce_mean(enc_seq_word_right, axis=1)
                att_seq_word_left = attend(enc_seq_word_left, context=None, feature_dim=feature_dim,
                                           method=self.params["attend_method"],
                                           scope_name=self.model_name + "att_seq_word",
                                           reuse=False)
                context = tf.reduce_mean(enc_seq_word_left, axis=1)
                att_seq_word_right = attend(enc_seq_word_right, context=None, feature_dim=feature_dim,
                                            method=self.params["attend_method"],
                                            scope_name=self.model_name + "att_seq_word",
                                            reuse=True)

                #### matching
                # fm
                tmp = tf.concat([enc_seq_word_left, enc_seq_word_right], axis=1)
                sum_squared = tf.square(tf.reduce_sum(tmp, axis=1))
                squared_sum = tf.reduce_sum(tf.square(tmp), axis=1)
                fm_word = 0.5 * (sum_squared - squared_sum)

                # cosine similarity
                att_seq_word_norm_left = tf.nn.l2_normalize(att_seq_word_left, dim=1)
                att_seq_word_norm_right = tf.nn.l2_normalize(att_seq_word_right, dim=1)
                # att_seq_word_norm_left = att_seq_word_left
                # att_seq_word_norm_right = att_seq_word_right
                sim_word = att_seq_word_norm_left * att_seq_word_norm_right

                # diff
                diff_word = tf.abs(att_seq_word_norm_left - att_seq_word_norm_right)

                # dense
                deep_in_word = tf.concat([att_seq_word_norm_left, att_seq_word_norm_right], axis=-1)
                hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
                dropouts = [self.params["fc_dropout"]] * len(hidden_units)
                if self.params["fc_type"] == "fc":
                    deep_out_word = dense_block(deep_in_word, hidden_units=hidden_units, dropouts=dropouts,
                                                densenet=False,
                                                scope_name="deep_out_word", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    deep_out_word = dense_block(deep_in_word, hidden_units=hidden_units, dropouts=dropouts,
                                                densenet=True,
                                                scope_name="deep_out_word", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    deep_out_word = resnet_block(deep_in_word, hidden_units=hidden_units, dropouts=dropouts,
                                                 cardinality=1,
                                                 dense_shortcut=True, training=self.training,
                                                 seed=self.params["random_seed"], scope_name="deep_out_word", reuse=False)

            with tf.name_scope("char_network"):
                #### embed
                std = 0.1
                minval = -std
                maxval = std
                emb_char_matrix = tf.Variable(
                    tf.random_uniform([self.params["max_num_chars"] + 1, self.params["embedding_char_dim"]],
                                      minval, maxval,
                                      seed=self.params["random_seed"],
                                      dtype=tf.float32))

                emb_seq_char_left = tf.nn.embedding_lookup(emb_char_matrix, self.seq_char_left)
                emb_seq_char_right = tf.nn.embedding_lookup(emb_char_matrix, self.seq_char_right)

                #### dropout
                emb_seq_char_left = word_dropout(emb_seq_char_left,
                                                 training=self.training,
                                                 dropout=self.params["embedding_dropout"],
                                                 seed=self.params["random_seed"])
                emb_seq_char_right = word_dropout(emb_seq_char_right,
                                                  training=self.training,
                                                  dropout=self.params["embedding_dropout"],
                                                  seed=self.params["random_seed"])

                #### encode
                enc_seq_char_left = encode(emb_seq_char_left, method=self.params["encode_method"], params=self.params,
                                           scope_name=self.model_name + "enc_seq_char", reuse=False)
                enc_seq_char_right = encode(emb_seq_char_right, method=self.params["encode_method"], params=self.params,
                                            scope_name=self.model_name + "enc_seq_char", reuse=True)

                #### attend
                feature_dim = self.params["encode_dim"]
                context = tf.reduce_mean(enc_seq_char_right, axis=1)
                att_seq_char_left = attend(enc_seq_char_left, context=None, feature_dim=feature_dim,
                                           method=self.params["attend_method"],
                                           scope_name=self.model_name + "att_seq_char",
                                           reuse=False)
                context = tf.reduce_mean(enc_seq_char_left, axis=1)
                att_seq_char_right = attend(enc_seq_char_right, context=None, feature_dim=feature_dim,
                                            method=self.params["attend_method"],
                                            scope_name=self.model_name + "att_seq_char",
                                            reuse=True)

                #### matching
                # fm
                tmp = tf.concat([enc_seq_char_left, enc_seq_char_right], axis=1)
                sum_squared = tf.square(tf.reduce_sum(tmp, axis=1))
                squared_sum = tf.reduce_sum(tf.square(tmp), axis=1)
                fm_char = 0.5 * (sum_squared - squared_sum)

                # cosine similarity
                att_seq_char_norm_left = tf.nn.l2_normalize(att_seq_char_left, dim=1)
                att_seq_char_norm_right = tf.nn.l2_normalize(att_seq_char_right, dim=1)
                # att_seq_char_norm_left = att_seq_char_left
                # att_seq_char_norm_right = att_seq_char_right
                sim_char = att_seq_char_norm_left * att_seq_char_norm_right

                # diff
                diff_char = tf.abs(att_seq_char_norm_left - att_seq_char_norm_right)

                # dense
                deep_in_char = tf.concat([att_seq_char_norm_left, att_seq_char_norm_right], axis=-1)
                hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
                dropouts = [self.params["fc_dropout"]] * len(hidden_units)
                if self.params["fc_type"] == "fc":
                    deep_out_char = dense_block(deep_in_char, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                                           scope_name="deep_out_char", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    deep_out_char = dense_block(deep_in_char, hidden_units=hidden_units, dropouts=dropouts, densenet=True,
                                                scope_name="deep_out_char", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    deep_out_char = resnet_block(deep_in_char, hidden_units=hidden_units, dropouts=dropouts, cardinality=1,
                                            dense_shortcut=True, training=self.training,
                                            seed=self.params["random_seed"], scope_name="deep_out_char", reuse=False)

            with tf.name_scope("prediction"):
                out_0 = tf.concat([
                    sim_word, diff_word, fm_word, deep_out_word, att_seq_word_norm_left, att_seq_word_norm_right,
                    sim_char, diff_char, fm_char, deep_out_char, att_seq_char_norm_left, att_seq_char_norm_right,
                ], axis=-1)
                hidden_units = [self.params["fc_dim"] * 4, self.params["fc_dim"] * 2, self.params["fc_dim"]]
                dropouts = [self.params["fc_dropout"]] * len(hidden_units)
                if self.params["fc_type"] == "fc":
                    out = dense_block(out_0, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                                      scope_name="mlp", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    out = dense_block(out_0, hidden_units=hidden_units, dropouts=dropouts, densenet=True,
                                      scope_name="mlp", reuse=False, training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    out = resnet_block(out_0, hidden_units=hidden_units, dropouts=dropouts, cardinality=1,
                                       dense_shortcut=True, training=self.training,
                                       seed=self.params["random_seed"], scope_name="mlp", reuse=False)
                logits = tf.layers.dense(out, 1, activation=None,
                                         kernel_initializer=tf.glorot_uniform_initializer(seed=self.params["random_seed"]),
                                         name="logit")
                logits = tf.squeeze(logits, axis=1)
                proba = tf.nn.sigmoid(logits)

            with tf.name_scope("loss"):
                log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
                log_loss = tf.reduce_mean(log_loss, name="log_loss")
                l2_losses = tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.params["l2_lambda"]
                loss = log_loss + l2_losses

        return loss, proba
