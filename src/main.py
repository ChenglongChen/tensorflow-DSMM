
import sys
import pickle as pkl
import numpy as np
import tensorflow as tf

import config

from inputs.data import load_question, load_train, load_test
from inputs.data import init_embedding_matrix

from utils import log_utils, os_utils, time_utils
from models.model_library import get_model


def get_model_data(df, features, params):
    X = {
        "q1": df.q1.values,
        "q2": df.q2.values,
        "label": df.label.values,
    }
    if params["use_features"]:
        X.update({
            "features": features,
        })
        params["num_features"] = X["features"].shape[1]
    return X


params = {
    "model_name": "semantic_matching",
    "offline_model_dir": "./weights/semantic_matching",
    "summary_dir": "../summary",
    "construct_neg": False,

    "augmentation_init_permutation": 0.5,
    "augmentation_min_permutation": 0.01,
    "augmentation_permutation": False,

    "augmentation_init_dropout": 0.5,
    "augmentation_min_dropout": 0.01,
    "augmentation_decay_steps": 1000,
    "augmentation_decay_rate": 0.95,

    "use_features": False,
    "num_features": 1,

    "n_runs": 10,
    "batch_size": 128,
    "epoch": 25,
    "max_batch": 1,
    "l2_lambda": 0.000,

    # embedding
    "embedding_dropout": 0.2,
    "embedding_dim_word": init_embedding_matrix["word"].shape[1],
    "embedding_dim_char": init_embedding_matrix["char"].shape[1],
    "embedding_dim": init_embedding_matrix["word"].shape[1],
    "embedding_dim_compressed": 32,
    "embedding_trainable": True,
    "embedding_mask_zero": False,

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
    "eval_every_num_update": 1000,

    # semantic feature layer
    "encode_method": "textcnn",
    "attend_method": ["ave", "max", "min", "self-vector-attention"],
    "attention_dim": 64,
    "attention_num_heads": 5,

    # cnn
    "cnn_num_layers": 1,
    "cnn_num_filters": 32,
    "cnn_filter_sizes": [1, 2, 3],
    "cnn_timedistributed": False,
    "cnn_activation": tf.nn.relu,
    "cnn_gated_conv": True,
    "cnn_residual": True,

    "rnn_num_units": 32,
    "rnn_cell_type": "gru",
    "rnn_num_layers": 1,

    # fc block
    "fc_type": "fc",
    "fc_hidden_units": [64*4, 64*2, 64],
    "fc_dropouts": [0, 0, 0],

    # True: cosine(l1, l2), sum(abs(l1 - l2))
    # False: l1 * l2, abs(l1 - l2)
    "similarity_aggregation": False,

    # match pyramid
    "mp_num_filters": [8, 16],
    "mp_filter_sizes": [5, 3],
    "mp_activation": tf.nn.relu,
    "mp_dynamic_pooling": False,
    "mp_pool_sizes_word": [6, 3],
    "mp_pool_sizes_char": [10, 5],

    # bcnn
    "bcnn_num_layers": 2,
    "bcnn_num_filters": 16,
    "bcnn_filter_size": 3,
    "bcnn_activation": tf.nn.tanh, # tf.nn.relu with euclidean/euclidean_exp produce nan
    "bcnn_match_score_type": "euclidean_exp",

    "bcnn_mp_att_pooling": False,
    "bcnn_mp_num_filters": [8, 16],
    "bcnn_mp_filter_sizes": [5, 3],
    "bcnn_mp_activation": tf.nn.relu,
    "bcnn_mp_dynamic_pooling": False,
    "bcnn_mp_pool_sizes_word": [6, 3],
    "bcnn_mp_pool_sizes_char": [10, 5],

    # final layer
    "final_dropout": 0.2,

}


def main():
    model_type = None
    if len(sys.argv) > 1:
        model_type = sys.argv[1]

    os_utils._makedirs("../logs")
    os_utils._makedirs("../output")
    logger = log_utils._get_logger("../logs", "tf-%s.log" % time_utils._timestamp())


    # load data
    Q = load_question(params)
    dfTrain = load_train()
    dfTest = load_test()
    train_features = np.load(config.TRAIN_FEATURES_FILE)
    test_features = np.load(config.TEST_FEATURES_FILE)
    params["num_features"] = train_features.shape[1]


    # load split
    with open(config.SPLIT_FILE, "rb") as f:
        train_idx, valid_idx = pkl.load(f)


    # validation
    X_train = get_model_data(dfTrain.loc[train_idx], train_features[train_idx], params)
    X_valid = get_model_data(dfTrain.loc[valid_idx], train_features[valid_idx], params)

    model = get_model(model_type)(params, logger, init_embedding_matrix=init_embedding_matrix)
    model.fit(X_train, Q, validation_data=X_valid, shuffle=True)


    # submit
    X_train = get_model_data(dfTrain, train_features, params)
    X_test = get_model_data(dfTest, test_features, params)
    y_proba = np.zeros((dfTest.shape[0], params["n_runs"]), dtype=np.float32)
    for run in range(params["n_runs"]):
        params["random_seed"] = run
        params["model_name"] = "semantic_model_%s"%str(run+1)
        model = get_model(model_type)(params, logger, init_embedding_matrix=init_embedding_matrix)
        model.fit(X_train, Q, validation_data=None, shuffle=True)
        y_proba[:,run] = model.predict_proba(X_test, Q).flatten()
        dfTest["y_pre"] = np.mean(y_proba[:,:(run+1)], axis=1)
        dfTest[["y_pre"]].to_csv(config.SINGLE_SUB_FILE_PATTERN%(model_type, str(run+1)), header=True, index=False)


if __name__ == "__main__":
    main()
