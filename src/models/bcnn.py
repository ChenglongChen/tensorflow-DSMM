
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel


class BCNNBaseModel(BaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(BCNNBaseModel, self).__init__(params, logger, init_embedding_matrix)


    def _padding(self, x, name):
        # x: [batch, s, d, 1]
        # x => [batch, s+w*2-2, d, 1]
        w = self.params["bcnn_filter_sizes"]
        return tf.pad(x, np.array([[0, 0], [w - 1, w - 1], [0, 0], [0, 0]]), "CONSTANT", name)


    def _make_attention_mat(self, x1, x2):
        # x1: [batch, s1, d, 1]
        # x2: [batch, s2, d, 1]
        # match score
        if "euclidean" in self.params["bcnn_match_score_type"]:
            # x1 => [batch, s1, 1, d]
            # x2 => [batch, 1, s2, d]
            x1_ = tf.transpose(x1, perm=[0, 1, 3, 2])
            x2_ = tf.transpose(x2, perm=[0, 3, 1, 2])
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1_ - x2_), axis=-1))
            if "exp" in self.params["bcnn_match_score_type"]:
                # exp(-euclidean / (2. * beta)) (producenan)
                # from Convolutional Neural Network for Paraphrase Identification
                beta = 2.
                att = tf.exp(-euclidean / (2. * beta))
            else:
                # euclidean distance (produce nan)
                att = 1. / (1. + euclidean)
        elif self.params["bcnn_match_score_type"] == "cosine":
            # cosine similarity
            x1_ = tf.nn.l2_normalize(x1, dim=2)
            x2_ = tf.nn.l2_normalize(x2, dim=2)
            sim = tf.einsum("abcd,aecd->abe", x1_, x2_) # value in [-1, 1]
            att = (1. + sim) / 2. # value in [0, 1]
        return att


    def _convolution(self, x, d, name, reuse=False):
        # conv: [batch, s+w-1, 1, d]
        conv = tf.layers.conv2d(
            inputs=x,
            filters=self.params["bcnn_num_filters"],
            kernel_size=(self.params["bcnn_filter_sizes"], d),
            padding="valid",
            activation=self.params["bcnn_activation"],
            strides=1,
            reuse=reuse,
            name=name)

        # [batch, s+w-1, d, 1]
        return tf.transpose(conv, perm=[0, 1, 3, 2])


    def _attention_pooling(self, x, attention, name):
        # x: [batch, s+w-1, d, 1]
        # attention: [batch, s+w-1]
        if attention is not None:
            attention = tf.expand_dims(tf.expand_dims(attention, axis=-1), axis=-1)
            x2 = x * attention
        else:
            x2 = x
        w_ap = tf.layers.average_pooling2d(
            inputs=x2,
            pool_size=(self.params["bcnn_filter_sizes"], 1),
            strides=1,
            padding="valid",
            name=name)
        if attention is not None:
            w_ap = w_ap * self.params["bcnn_filter_sizes"]

        return w_ap


    def _global_pooling(self, x, seq_len, name):
        if "input" in name:
            pool_width = seq_len
            d = self.params["embedding_dim"]
        else:
            pool_width = seq_len + self.params["bcnn_filter_sizes"] - 1
            d = self.params["bcnn_num_filters"]

        all_ap = tf.layers.average_pooling2d(
            inputs=x,
            pool_size=(pool_width, 1),
            strides=1,
            padding="valid",
            name=name)
        all_ap_reshaped = tf.reshape(all_ap, [-1, d])

        return all_ap_reshaped


    def _expand_input(self, x1, x2, seq_len, d, name):
        aW = tf.get_variable(name=name, shape=(seq_len, d))

        # [batch, s, s]
        att_mat = self._make_attention_mat(x1, x2)

        # [batch, s, s] * [s,d] => [batch, s, d]
        # expand dims => [batch, s, d, 1]
        x1_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", att_mat, aW), -1)
        x2_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW), -1)

        # [batch, s, d, 2]
        x1 = tf.concat([x1, x1_a], axis=3)
        x2 = tf.concat([x2, x2_a], axis=3)

        return x1, x2


    def _cnn_layer(self, x1, x2, seq_len, d, name):
        return None, None, None, None


    def _bcnn_interaction_feature_layer(self, enc_seq_left, enc_seq_right, granularity="word"):
        name = self.model_name + granularity
        seq_len = self.params["max_seq_len_%s"%granularity]
        # [batch, s, d] => [batch, s, d, 1]
        enc_seq_left = tf.expand_dims(enc_seq_left, axis=-1)
        enc_seq_right = tf.expand_dims(enc_seq_right, axis=-1)

        LO_0 = self._global_pooling(x=enc_seq_left, seq_len=seq_len, name=name + "global_pooling_input_left")
        RO_0 = self._global_pooling(x=enc_seq_right, seq_len=seq_len, name=name + "global_pooling_input_right")

        LI_1, LO_1, RI_1, RO_1 = self._cnn_layer(x1=enc_seq_left, x2=enc_seq_right, seq_len=seq_len,
                                                 d=self.params["embedding_dim"], name=name + "cnn_layer_1")
        sims = [self._cosine_similarity(LO_0, RO_0), self._cosine_similarity(LO_1, RO_1)]

        _, LO_2, _, RO_2 = self._cnn_layer(x1=LI_1, x2=RI_1, seq_len=seq_len,
                                           d=self.params["bcnn_num_filters"],
                                           name=name + "cnn_layer_2")
        sims.append(self._cosine_similarity(LO_2, RO_2))
        return tf.concat(sims, axis=-1)


    def _build_model(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.params["random_seed"])

            with tf.name_scope("word_network"):
                _, enc_seq_word_left = self._semantic_feature_layer(self.seq_word_left, granularity="word", reuse=False,
                                                                    return_enc=True)
                _, enc_seq_word_right = self._semantic_feature_layer(self.seq_word_right, granularity="word", reuse=True,
                                                                     return_enc=True)
                sim_word = self._bcnn_interaction_feature_layer(enc_seq_word_left, enc_seq_word_right, granularity="word")

            with tf.name_scope("char_network"):
                _, enc_seq_char_left = self._semantic_feature_layer(self.seq_char_left, granularity="char", reuse=False,
                                                                    return_enc=True)
                _, enc_seq_char_right = self._semantic_feature_layer(self.seq_char_right, granularity="char",
                                                                     reuse=True,
                                                                     return_enc=True)
                sim_char = self._bcnn_interaction_feature_layer(enc_seq_char_left, enc_seq_char_right, granularity="char")

            with tf.name_scope("prediction"):
                out_0 = tf.concat([sim_word, sim_char], axis=-1)
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


class BCNN(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(BCNN, self).__init__(params, logger, init_embedding_matrix)


    def _cnn_layer(self, x1, x2, seq_len, d, name):
        # x1, x2 = [batch, s, d, 1]
        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        left_attention, right_attention = None, None

        left_wp = self._attention_pooling(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._global_pooling(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._attention_pooling(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._global_pooling(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        return left_wp, left_ap, right_wp, right_ap


class ABCNN1(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(ABCNN1, self).__init__(params, logger, init_embedding_matrix)


    def _cnn_layer(self, x1, x2, seq_len, d, name):
        # x1, x2 = [batch, s, d, 1]
        x1, x2 = self._expand_input(x1, x2, seq_len, d, name=name+"expand_input")

        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        left_attention, right_attention = None, None

        left_wp = self._attention_pooling(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._global_pooling(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._attention_pooling(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._global_pooling(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        return left_wp, left_ap, right_wp, right_ap


class ABCNN2(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(ABCNN2, self).__init__(params, logger, init_embedding_matrix)


    def _cnn_layer(self, x1, x2, seq_len, d, name):
        # x1, x2 = [batch, s, d, 1]
        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        # [batch, s+w-1, s+w-1]
        att_mat = self._make_attention_mat(left_conv, right_conv)
        # [batch, s+w-1], [batch, s+w-1]
        left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

        left_wp = self._attention_pooling(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._global_pooling(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._attention_pooling(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._global_pooling(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        return left_wp, left_ap, right_wp, right_ap


class ABCNN3(BCNNBaseModel):
    def __init__(self, params, logger, init_embedding_matrix):
        super(ABCNN3, self).__init__(params, logger, init_embedding_matrix)


    def _cnn_layer(self, x1, x2, seq_len, d, name):
        # x1, x2 = [batch, s, d, 1]
        x1, x2 = self._expand_input(x1, x2, seq_len, d, name=name + "expand_input")

        left_conv = self._convolution(x=self._padding(x1, name=name+"padding_left"), d=d, name=name+"conv", reuse=False)
        right_conv = self._convolution(x=self._padding(x2, name=name+"padding_right"), d=d, name=name+"conv", reuse=True)

        # [batch, s+w-1, s+w-1]
        att_mat = self._make_attention_mat(left_conv, right_conv)
        # [batch, s+w-1], [batch, s+w-1]
        left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

        left_wp = self._attention_pooling(x=left_conv, attention=left_attention, name=name+"attention_pooling_left")
        left_ap = self._global_pooling(x=left_conv, seq_len=seq_len, name=name+"global_pooling_left")
        right_wp = self._attention_pooling(x=right_conv, attention=right_attention, name=name+"attention_pooling_right")
        right_ap = self._global_pooling(x=right_conv, seq_len=seq_len, name=name+"global_pooling_right")

        return left_wp, left_ap, right_wp, right_ap
