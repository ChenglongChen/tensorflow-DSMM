
import sys
import numpy as np
import tensorflow as tf

import config

from inputs.data import load_question, load_train, load_test
from inputs.data import init_embedding_matrix

from utils import log_utils, os_utils, time_utils
from models.model_library import get_model


def get_model_data(dataset, params):
    X = {
        "q1": dataset.q1.values,
        "q2": dataset.q2.values,
        "label": dataset.label.values,
    }
    return X


params = {
    "model_name": "semantic_matching",
    "offline_model_dir": "./weights/semantic_matching",
    "construct_neg": False,
    "n_runs": 10,
    "batch_size": 256,
    "epoch": 20,
    "l2_lambda": 0.000,

    # embedding
    "embedding_dropout": 0.2,
    "embedding_dim_word": init_embedding_matrix["word"].shape[1],
    "embedding_dim_char": init_embedding_matrix["char"].shape[1],
    "embedding_dim": init_embedding_matrix["word"].shape[1],
    "embedding_trainable": True,

    "max_num_word": init_embedding_matrix["word"].shape[0],
    "max_num_char": init_embedding_matrix["char"].shape[0],

    "threshold": 0.217277,
    "calibration_factor": 1.0,

    "max_seq_len_word": 12,
    "max_seq_len_char": 20,
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",

    # optimization
    "optimizer_type": "nadam",
    "init_lr": 0.001,
    "beta1": 0.975,
    "beta2": 0.999,
    "decay_steps": 1000,
    "decay_rate": 0.95,
    "schedule_decay": 0.004,
    "random_seed": 2018,
    "eval_every_num_update": 100,

    # semantic feature layer
    "encode_method": "fasttext",
    "attend_method": "attention",

    "cnn_num_filters": 32,
    "cnn_filter_sizes": [1, 2, 3],
    "cnn_timedistributed": False,

    "rnn_num_units": 20,
    "rnn_cell_type": "gru",

    # fc block
    "fc_type": "fc",
    "fc_hidden_units": [64*4, 64*2, 64],
    "fc_dropouts": [0, 0, 0],

    # match pyramid
    "mp_num_filters": 8,
    "mp_filter_size": 3,
    "mp_activation": tf.nn.relu,
    "mp_dynamic_pooling": False,
    "mp_pool_size_word": 4,
    "mp_pool_size_char": 4,

    # bcnn
    "bcnn_num_filters": 8,
    "bcnn_filter_size": 3,
    "bcnn_activation": tf.nn.tanh,
    "bcnn_match_score_type": "cosine", # euclidean/euclidean_exp produce nan
}


def main():
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1]

    os_utils._makedirs("../logs")
    os_utils._makedirs("../output")
    logger = log_utils._get_logger("../logs", "tf-%s.log" % time_utils._timestamp())

    Q = load_question(params)
    dfTrain = load_train()
    dfTest = load_test()
    X_test = get_model_data(dfTest, params)

    # shuffle training data
    dfTrain = dfTrain.sample(frac=1.0)


    # validation
    train_ratio = 0.7
    N = dfTrain.shape[0]
    train_num = int(N * train_ratio)
    X_train = get_model_data(dfTrain[:train_num], params)
    X_valid = get_model_data(dfTrain[train_num:], params)

    model = get_model(model_type)(params, logger, init_embedding_matrix=init_embedding_matrix)
    model.fit(X_train, Q, validation_data=X_valid, shuffle=True)


    # submit
    X_train = get_model_data(dfTrain, params)
    y_proba = np.zeros((dfTest.shape[0], params["n_runs"]), dtype=np.float32)
    for run in range(params["n_runs"]):
        params["random_seed"] = run
        params["model_name"] = "semantic_model_%s"%str(run+1)
        model = get_model(model_type)(params, logger, init_embedding_matrix=init_embedding_matrix)
        model.fit(X_train, Q, validation_data=None, shuffle=True)
        y_proba[:,run] = model.predict_proba(X_test, Q).flatten()
        dfTest["y_pre"] = np.mean(y_proba[:,:(run+1)], axis=1)
        dfTest[["y_pre"]].to_csv(config.SUB_FILE_PATTERN%str(run+1), header=True, index=False)


if __name__ == "__main__":
    main()
