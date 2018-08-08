
import numpy as np
import tensorflow as tf

"""
https://explosion.ai/blog/deep-learning-formula-nlp
embed -> encode -> attend -> predict
"""
def batch_normalization(x, training, name):
    # with tf.variable_scope(name, reuse=)
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(training, lambda: bn_train, lambda: bn_inference)
    return z


#### Step 1
def embed(x, size, dim, seed=0, flatten=False, reduce_sum=False):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * embed_dim
    out = tf.nn.embedding_lookup(emb, x)
    if flatten:
        out = tf.layers.flatten(out)
    if reduce_sum:
        out = tf.reduce_sum(out, axis=1)
    return out


def embed_subword(x, size, dim, sequence_length, seed=0, mask_zero=False, maxlen=None):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * max_word_len * embed_dim
    out = tf.nn.embedding_lookup(emb, x)
    if mask_zero:
        # word_len: None * max_seq_len
        # mask: shape=None * max_seq_len * max_word_len
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(mask, tf.float32)
        out = out * mask
    # None * max_seq_len * embed_dim
    # according to facebook subword paper, it's sum
    out = tf.reduce_sum(out, axis=2)
    return out


def word_dropout(x, training, dropout=0, seed=0):
    # word dropout (dropout the entire embedding for some words)
    """
    tf.layers.Dropout doesn't work as it can't switch training or inference
    """
    if dropout > 0:
        input_shape = tf.shape(x)
        noise_shape = [input_shape[0], input_shape[1], 1]
        x = tf.layers.Dropout(rate=dropout, noise_shape=noise_shape, seed=seed)(x, training=training)
    return x


#### Step 2
def fasttext(x):
    return x


# Language Modeling with Gated Convolutional Networks
# https://github.com/anantzoid/Language-Modeling-GatedCNN
def gated_conv1d_op(inputs, filters=8, kernel_size=3, padding="same", activation=None, strides=1, reuse=False, name=""):
    conv_linear = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        strides=strides,
        reuse=reuse,
        name=name+"_linear")
    conv_gated = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
        strides=strides,
        reuse=reuse,
        name=name+"_gated")
    conv = conv_linear * conv_gated
    return conv


def residual_gated_conv1d_op(inputs, filters=8, kernel_size=3, padding="same", activation=None, strides=1, reuse=False, name=""):
    conv_linear = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        strides=strides,
        reuse=reuse,
        name=name+"_linear")
    conv_gated = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
        strides=strides,
        reuse=reuse,
        name=name+"_gated")
    conv = inputs * (1. - conv_gated) + conv_linear * conv_gated
    return conv


def _textcnn(x, conv_op, num_filters=8, filter_sizes=[2, 3], bn=False, training=False,
            timedistributed=False, scope_name="textcnn", reuse=False, activation=tf.nn.relu):
    # x: None * step_dim * embed_dim
    conv_blocks = []
    for i, filter_size in enumerate(filter_sizes):
        scope_name_i = "%s_textcnn_%s"%(str(scope_name), str(filter_size))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if timedistributed:
                input_shape = tf.shape(x)
                step_dim = input_shape[1]
                embed_dim = input_shape[2]
                x = tf.transpose(x, [0, 2, 1])
                # None * embed_dim * step_dim
                x = tf.reshape(x, [input_shape[0] * embed_dim, step_dim, 1])
                conv = conv_op(
                    inputs=x,
                    filters=1,
                    kernel_size=filter_size,
                    padding="same",
                    activation=activation,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
                conv = tf.reshape(conv, [input_shape[0], embed_dim, step_dim])
                conv = tf.transpose(conv, [0, 2, 1])
            else:
                conv = conv_op(
                    inputs=x,
                    filters=num_filters,
                    kernel_size=filter_size,
                    padding="same",
                    activation=activation,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
            if bn:
                conv = tf.layers.BatchNormalization()(conv, training)
            # conv = activation(conv)
            conv_blocks.append(conv)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z


def textcnn(x, num_layers=2, num_filters=8, filter_sizes=[2, 3], bn=False, training=False,
            timedistributed=False, scope_name="textcnn", reuse=False, activation=tf.nn.relu,
            gated_conv=False, residual=False):
    if gated_conv:
        if residual:
            conv_op = residual_gated_conv1d_op
        else:
            conv_op = gated_conv1d_op
    else:
        conv_op = tf.layers.conv1d
    conv_blocks = []
    for i in range(num_layers):
        scope_name_i = "%s_textcnn_layer_%s" % (str(scope_name), str(i))
        x = _textcnn(x, conv_op, num_filters, filter_sizes, bn, training, timedistributed, scope_name_i, reuse, activation)
        conv_blocks.append(x)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z


def textrnn(x, num_units, cell_type, sequence_length, num_layers=1, mask_zero=False, scope_name="textrnn", reuse=False):
    for i in range(num_layers):
        scope_name_i = "%s_textrnn_%s_%s_%s" % (str(scope_name), cell_type, str(i), str(num_units))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if cell_type == "gru":
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
            elif cell_type == "lstm":
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
            if mask_zero:
                x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope_name_i)
            else:
                x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=None, scope=scope_name_i)
    return x


def textbirnn(x, num_units, cell_type, sequence_length, num_layers=1, mask_zero=False, scope_name="textbirnn", reuse=False):
    for i in range(num_layers):
        scope_name_i = "%s_textbirnn_%s_%s_%s" % (str(scope_name), cell_type, str(i), str(num_units))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if cell_type == "gru":
                cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
                cell_bw = tf.nn.rnn_cell.GRUCell(num_units)
            elif cell_type == "lstm":
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            if mask_zero:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope_name_i)
            else:
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=None, scope=scope_name_i)
            x = tf.concat([output_fw, output_bw], axis=-1)
    return x



def encode(x, method, params, input_dim,
           sequence_length=None, mask_zero=False,
           scope_name="encode", reuse=False,
           training=False, seed=0):
    """
    :param x: shape=(None,seqlen,dim)
    :param params:
    :return: shape=(None,seqlen,dim)
    """
    out_list = []
    params["encode_dim"] = 0
    for m in method.split("+"):
        if m == "fasttext":
            dim_f = input_dim  # params["embedding_dim"]
            z = fasttext(x)
            out_list.append(z)
            params["encode_dim"] += dim_f
        elif m == "project":
            dim_p = params["project_hidden_units"][-1]
            step_dim = tf.shape(x)[1]
            z = tf.reshape(x, [-1, input_dim])
            z = mlp_layer(z, fc_type=params["project_type"],
                          hidden_units=params["project_hidden_units"],
                          dropouts=params["project_dropouts"],
                          scope_name=scope_name,
                          reuse=reuse,
                          training=training,
                          seed=params["random_seed"])
            z = tf.reshape(z, [-1, step_dim, params["project_hidden_units"][-1]])
            out_list.append(z)
            params["encode_dim"] += dim_p
        elif m == "textcnn":
            dim_c = params["cnn_num_layers"] * len(params["cnn_filter_sizes"]) * params["cnn_num_filters"]
            z = textcnn(x, num_layers=params["cnn_num_layers"], num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                        timedistributed=params["cnn_timedistributed"], scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_c
        elif m == "textrnn":
            dim_r = params["rnn_num_units"]
            z = textrnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"], num_layers=params["rnn_num_layers"],
                        sequence_length=sequence_length, mask_zero=mask_zero, scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_r
        elif method == "textbirnn":
            dim_b = params["rnn_num_units"] * 2
            z = textbirnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"], num_layers=params["rnn_num_layers"],
                          sequence_length=sequence_length, mask_zero=mask_zero, scope_name=scope_name, reuse=reuse)
            out_list.append(z)
            params["encode_dim"] += dim_b
    z = tf.concat(out_list, axis=-1)
    return z


def scalar_attention(x, encode_dim, feature_dim, attention_dim, sequence_length=None,
                     mask_zero=False, maxlen=None, epsilon=1e-8, seed=0, scope_name="attention", reuse=False):
    """
    :param x: [batchsize, s, feature_dim]
    :param encode_dim: dim of encoder output
    :param feature_dim: dim of x (for self-attention, x is the encoder output;
                        for context-attention, x is the concat of encoder output and contextual info)
    :param sequence_length:
    :param mask_zero:
    :param maxlen:
    :param epsilon:
    :param seed:
    :param scope_name:
    :param reuse:
    :return: [batchsize, s, 1]
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        # W1: [feature_dim]
        W1 = tf.get_variable("W1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[feature_dim])
        # b1: [1]
        b1 = tf.get_variable("b1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[1])
    e = tf.einsum("bsf,f->bs", x, W1) + \
        tf.expand_dims(b1, axis=1)
    a = tf.exp(e)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # None * s
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.cast(mask, tf.float32)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    s = tf.reduce_sum(a, axis=1, keep_dims=True)
    a /= tf.cast(s + epsilon, tf.float32)
    a = tf.expand_dims(a, axis=-1)

    return a


# vector-based attention proposed in the following paper
# Enhancing Sentence Embedding with Generalized Pooling
def vector_attention(x, encode_dim, feature_dim, attention_dim, sequence_length=None,
                     mask_zero=False, maxlen=None, epsilon=1e-8, seed=0,
                     scope_name="attention", reuse=False):
    """
    :param x: [batchsize, s, feature_dim]
    :param encode_dim: dim of encoder output
    :param feature_dim: dim of x (for self-attention, x is the encoder output;
                        for context-attention, x is the concat of encoder output and contextual info)
    :param sequence_length:
    :param mask_zero:
    :param maxlen:
    :param epsilon:
    :param seed:
    :param scope_name:
    :param reuse:
    :return: [batchsize, s, encode_dim]
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        # W1: [attention_dim, feature_dim]
        W1 = tf.get_variable("W1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[attention_dim, feature_dim])
        # b1: [attention_dim]
        b1 = tf.get_variable("b1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[attention_dim])
        # W2: [encode_dim, attention_dim]
        W2 = tf.get_variable("W2_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[encode_dim, attention_dim])
        # b2: [encode_dim]
        b2 = tf.get_variable("b2_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[encode_dim])
    # [batchsize, attention_dim, s]
    e = tf.nn.relu(
        tf.einsum("bsf,af->bas", x, W1) + \
        tf.expand_dims(tf.expand_dims(b1, axis=0), axis=-1))
    # [batchsize, s, encode_dim]
    e = tf.einsum("bas,ea->bse", e, W2) + \
        tf.expand_dims(tf.expand_dims(b2, axis=0), axis=0)
    a = tf.exp(e)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # [batchsize, s, 1]
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    s = tf.reduce_sum(a, axis=1, keep_dims=True)
    a /= tf.cast(s + epsilon, tf.float32)

    return a


def _attend(x, sequence_length=None, method="ave", context=None, encode_dim=None,
            feature_dim=None, attention_dim=None, mask_zero=False, maxlen=None,
           bn=False, training=False, seed=0, scope_name="attention", reuse=False,
            num_heads=1):
    if method == "ave":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
            l = tf.reduce_sum(mask, axis=1)
            # in some cases especially in the early stages of training the sum may be almost zero
            epsilon = 1e-8
            z /= tf.cast(l + epsilon, tf.float32)
        else:
            z = tf.reduce_mean(x, axis=1)
    elif method == "sum":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
        else:
            z = tf.reduce_sum(x, axis=1)
    elif method == "max":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, (1, 1, tf.shape(x)[2]))
            masked_data = tf.where(tf.equal(mask, tf.zeros_like(mask)),
                                   tf.ones_like(x) * -np.inf, x)  # if masked assume value is -inf
            z = tf.reduce_max(masked_data, axis=1)
        else:
            z = tf.reduce_max(x, axis=1)
    elif method == "min":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, (1, 1, tf.shape(x)[2]))
            masked_data = tf.where(tf.equal(mask, tf.zeros_like(mask)),
                                   tf.ones_like(x) * np.inf, x)  # if masked assume value is -inf
            z = tf.reduce_min(masked_data, axis=1)
        else:
            z = tf.reduce_min(x, axis=1)
    elif "attention" in method:
        if context is not None:
            y = tf.concat([x, context], axis=-1)
        else:
            y = x
        zs = []
        for i in range(num_heads):
            if "vector" in method:
                a = vector_attention(y, encode_dim, feature_dim, attention_dim, sequence_length, mask_zero, maxlen, seed=seed, scope_name=scope_name+str(i), reuse=reuse)
            else:
                a = scalar_attention(y, encode_dim, feature_dim, attention_dim, sequence_length, mask_zero, maxlen, seed=seed, scope_name=scope_name+str(i), reuse=reuse)
            zs.append(tf.reduce_sum(x * a, axis=1))
        z = tf.concat(zs, axis=-1)
    if bn:
        z = tf.layers.BatchNormalization()(z, training=training)
    return z


def attend(x, sequence_length=None, method="ave", context=None, encode_dim=None,
           feature_dim=None, attention_dim=None, mask_zero=False, maxlen=None,
           bn=False, training=False, seed=0, scope_name="attention", reuse=False,
           num_heads=1):
    if isinstance(method, list):
        outputs = [None]*len(method)
        for i,m in enumerate(method):
            outputs[i] = _attend(x, sequence_length, m, context, encode_dim, feature_dim, attention_dim, mask_zero, maxlen,
                                bn, training, seed, scope_name+m, reuse, num_heads)
        return tf.concat(outputs, axis=-1)
    else:
        return _attend(x, sequence_length, method, context, encode_dim, feature_dim, attention_dim, mask_zero, maxlen,
                                bn, training, seed, scope_name+method, reuse, num_heads)


#### Step 4
def _dense_block_mode1(x, hidden_units, dropouts, densenet=False, scope_name="dense_block", reuse=False, training=False, seed=0, bn=False):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        scope_name_i = "%s-dense_block_mode1-%s"%(str(scope_name), str(i))
        with tf.variable_scope(scope_name, reuse=reuse):
            z = tf.layers.dense(x, h, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * i),
                                  reuse=reuse,
                                  name=scope_name_i)
            if bn:
                z = batch_normalization(z, training=training, name=scope_name_i+"-bn")
            z = tf.nn.relu(z)
            z = tf.layers.Dropout(d, seed=seed * i)(z, training=training) if d > 0 else z
            if densenet:
                x = tf.concat([x, z], axis=-1)
            else:
                x = z
    return x


def _dense_block_mode2(x, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        if bn:
            z = batch_normalization(x, training=training, name=name + "-" + str(i))
        z = tf.nn.relu(z)
        z = tf.layers.Dropout(d, seed=seed * i)(z, training=training) if d > 0 else z
        z = tf.layers.Dense(h, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * i), dtype=tf.float32,
                            bias_initializer=tf.zeros_initializer())(z)
        if densenet:
            x = tf.concat([x, z], axis=-1)
        else:
            x = z
    return x


def dense_block(x, hidden_units, dropouts, densenet=False, scope_name="dense_block", reuse=False, training=False, seed=0, bn=False):
    return _dense_block_mode1(x, hidden_units, dropouts, densenet, scope_name, reuse, training, seed, bn)


def _resnet_branch_mode1(x, hidden_units, dropouts, training, seed=0):
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    name = "resnet_block"
    # branch 2
    x2 = tf.layers.Dense(h1, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 2), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x)
    x2 = tf.layers.BatchNormalization()(x2, training=training)
    # x2 = batch_normalization(x2, training=training, name=name + "-" + str(1))
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr1, seed=seed * 1)(x2, training=training) if dr1 > 0 else x2

    x2 = tf.layers.Dense(h2, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 3), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)
    x2 = tf.layers.BatchNormalization()(x2, training=training)
    # x2 = batch_normalization(x2, training=training, name=name + "-" + str(2))
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr2, seed=seed * 2)(x2, training=training) if dr2 > 0 else x2

    x2 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 4), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)
    x2 = tf.layers.BatchNormalization()(x2, training=training)
    # x2 = batch_normalization(x2, training=training, name=name + "-" + str(3))

    return x2


def _resnet_block_mode1(x, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    name = "resnet_block"
    xs = []
    # branch 0
    if dense_shortcut:
        x0 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 1), dtype=tf.float32,
                             bias_initializer=tf.zeros_initializer())(x)
        x0 = tf.layers.BatchNormalization()(x0, training=training)
        # x0 = batch_normalization(x0, training=training, name=name + "-" + str(0))
        xs.append(x0)
    else:
        xs.append(x)

    # branch 1 ~ cardinality
    for i in range(cardinality):
        xs.append(_resnet_branch_mode1(x, hidden_units, dropouts, training, seed))

    x = tf.add_n(xs)
    x = tf.nn.relu(x)
    x = tf.layers.Dropout(dr3, seed=seed * 4)(x, training=training) if dr3 > 0 else x
    return x


def _resnet_branch_mode2(x, hidden_units, dropouts, training=False, seed=0, scope_name="_resnet_branch_mode2", reuse=False):
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    # name = "resnet"
    with tf.variable_scope(scope_name, reuse=reuse):
        # branch 2: bn-relu->weight
        x2 = tf.layers.BatchNormalization()(x)
        # x2 = batch_normalization(x, training=training, name=scope_name + "-bn-" + str(1))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr1)(x2, training=training) if dr1 > 0 else x2
        x2 = tf.layers.dense(x2, h1, kernel_initializer=tf.glorot_uniform_initializer(seed * 1),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name+"-dense-"+str(1),
                             reuse=reuse)

        x2 = tf.layers.BatchNormalization()(x2)
        # x2 = batch_normalization(x2, training=training, name=scope_name + "-bn-" + str(2))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr2)(x2, training=training) if dr2 > 0 else x2
        x2 = tf.layers.dense(x2, h2, kernel_initializer=tf.glorot_uniform_initializer(seed * 2),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name + "-dense-" + str(2),
                             reuse=reuse)

        x2 = tf.layers.BatchNormalization()(x2)
        # x2 = batch_normalization(x2, training=training, name=scope_name + "-bn-" + str(3))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr3)(x2, training=training) if dr3 > 0 else x2
        x2 = tf.layers.dense(x2, h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 3),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name + "-dense-" + str(3),
                             reuse=reuse)

    return x2


def _resnet_block_mode2(x, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0,
                        scope_name="_resnet_block_mode2", reuse=False):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts

    xs = []
    # branch 0
    if dense_shortcut:
        with tf.variable_scope(scope_name, reuse=reuse):
            x0 = tf.layers.dense(x, h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 1),
                                 bias_initializer=tf.zeros_initializer(),
                                 reuse=reuse,
                                 name=scope_name+"-dense-"+str("0"))
        xs.append(x0)
    else:
        xs.append(x)

    # branch 1 ~ cardinality
    for i in range(cardinality):
        xs.append(_resnet_branch_mode2(x, hidden_units, dropouts, training, seed, scope_name, reuse))

    x = tf.add_n(xs)
    return x


def resnet_block(input_tensor, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0,
                 scope_name="resnet_block", reuse=False):
    return _resnet_block_mode2(input_tensor, hidden_units, dropouts, cardinality, dense_shortcut, training, seed,
                               scope_name, reuse)


def mlp_layer(input, fc_type, hidden_units, dropouts, scope_name, reuse=False, training=False, seed=0):
    if fc_type == "fc":
        output = dense_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                         densenet=False, scope_name=scope_name,
                                         reuse=reuse,
                                         training=training, seed=seed)
    elif fc_type == "densenet":
        output = dense_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                         densenet=True, scope_name=scope_name,
                                         reuse=reuse,
                                         training=training, seed=seed)
    elif fc_type == "resnet":
        output = resnet_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                          cardinality=1, dense_shortcut=True, training=training,
                                          reuse=reuse,
                                          seed=seed,
                                          scope_name=scope_name)
    return output
