
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

from optparse import OptionParser

import config

from inputs.data import load_question, load_train, load_test
from inputs.data import init_embedding_matrix
from models.model_library import get_model
from utils import log_utils, os_utils, time_utils


params = {
    "model_name": "semantic_matching",
    "offline_model_dir": "./weights/semantic_matching",
    "summary_dir": "../summary",
    "construct_neg": False,

    "augmentation_init_permutation": 0.5,
    "augmentation_min_permutation": 0.01,
    "augmentation_permutation_decay_steps": 2000,
    "augmentation_permutation_decay_rate": 0.975,

    "augmentation_init_dropout": 0.5,
    "augmentation_min_dropout": 0.01,
    "augmentation_dropout_decay_steps": 2000,
    "augmentation_dropout_decay_rate": 0.975,

    "use_features": False,
    "num_features": 1,

    "n_runs": 10,
    "batch_size": 128,
    "epoch": 50,
    "max_batch": -1,
    "l2_lambda": 0.000,

    # embedding
    "embedding_dropout": 0.3,
    "embedding_dim_word": init_embedding_matrix["word"].shape[1],
    "embedding_dim_char": init_embedding_matrix["char"].shape[1],
    "embedding_dim": init_embedding_matrix["word"].shape[1],
    "embedding_dim_compressed": 32,
    "embedding_trainable": True,
    "embedding_mask_zero": True,

    "max_num_word": init_embedding_matrix["word"].shape[0],
    "max_num_char": init_embedding_matrix["char"].shape[0],

    "threshold": 0.217277,
    "calibration": False,

    "max_seq_len_word": 12,
    "max_seq_len_char": 20,
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",

    # optimization
    "optimizer_type": "lazyadam",
    "init_lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "decay_steps": 2000,
    "decay_rate": 0.95,
    "schedule_decay": 0.004,
    "random_seed": 2018,
    "eval_every_num_update": 5000,

    # semantic feature layer
    "encode_method": "textcnn",
    "attend_method": ["ave", "max", "min", "self-scalar-attention"],
    "attention_dim": 64,
    "attention_num_heads": 1,

    # cnn
    "cnn_num_layers": 1,
    "cnn_num_filters": 32,
    "cnn_filter_sizes": [1, 2, 3],
    "cnn_timedistributed": False,
    "cnn_activation": tf.nn.relu,
    "cnn_gated_conv": False,
    "cnn_residual": False,

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
    "bcnn_match_score_type": "cosine",

    "bcnn_mp_att_pooling": False,
    "bcnn_mp_num_filters": [8, 16],
    "bcnn_mp_filter_sizes": [5, 3],
    "bcnn_mp_activation": tf.nn.relu,
    "bcnn_mp_dynamic_pooling": False,
    "bcnn_mp_pool_sizes_word": [6, 3],
    "bcnn_mp_pool_sizes_char": [10, 5],

    # final layer
    "final_dropout": 0.3,

}


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


def downsample(df):
    # downsample negative
    num_pos = np.sum(df.label)
    num_neg = int((1. / config.POS_RATIO_OFFLINE - 1.) * num_pos)
    idx_pos = np.where(df.label == 1)[0]
    idx_neg = np.where(df.label == 0)[0]
    np.random.shuffle(idx_neg)
    idx = np.hstack([idx_pos, idx_neg[:num_neg]])
    return df.loc[idx]


def get_train_valid_test_data(augmentation=False):
    # load data
    Q = load_question(params)
    dfTrain = load_train()
    dfTest = load_test()
    # train_features = load_feat("train")
    # test_features = load_feat("test")
    # params["num_features"] = train_features.shape[1]

    # load split
    with open(config.SPLIT_FILE, "rb") as f:
        train_idx, valid_idx = pkl.load(f)

    # validation
    if augmentation:
        dfDev = pd.read_csv(config.DATA_DIR + "/" + "dev_aug.csv")
        dfDev = downsample(dfDev)
        params["use_features"] = False
        params["augmentation_decay_steps"] = 50000
        params["decay_steps"] = 50000
        X_dev = get_model_data(dfDev, None, params)
    else:
        X_dev = get_model_data(dfTrain.loc[train_idx], None, params)
    X_valid = get_model_data(dfTrain.loc[valid_idx], None, params)

    # submit
    if augmentation:
        dfTrain = pd.read_csv(config.DATA_DIR + "/" + "train_aug.csv")
        dfTrain = downsample(dfTrain)
        params["use_features"] = False
        params["augmentation_decay_steps"] = 50000
        params["decay_steps"] = 50000
        X_train = get_model_data(dfTrain, None, params)
    else:
        X_train = get_model_data(dfTrain, None, params)
    X_test = get_model_data(dfTest, None, params)

    return X_dev, X_valid, X_train, X_test, Q


def parse_args(parser):
    parser.add_option("-m", "--model", type="string", dest="model",
                      help="model type", default="cdssm")
    parser.add_option("-a", "--augmentation", action="store_true", dest="augmentation",
                      help="augmentation", default=False)
    parser.add_option("-g", "--granularity", type="string", dest="granularity",
                      help="granularity, e.g., word or char", default="word")

    (options, args) = parser.parse_args()
    return options, args


def main(options):

    os_utils._makedirs("../logs")
    os_utils._makedirs("../output")
    logger = log_utils._get_logger("../logs", "tf-%s.log" % time_utils._timestamp())

    params["granularity"] = options.granularity

    # save path
    model_name = "augmentation_%s_%s_%s"%(str(options.augmentation), options.granularity, options.model)
    path = config.SUB_DIR + "/" + model_name
    os_utils._makedirs(path)

    # load data
    X_dev, X_valid, X_train, X_test, Q = get_train_valid_test_data(options.augmentation)

    # validation
    model = get_model(options.model)(params, logger, init_embedding_matrix=init_embedding_matrix)
    model.fit(X_dev, Q, validation_data=X_valid, shuffle=True)
    y_pred_valid = model.predict_proba(X_valid, Q).flatten()
    # save for stacking
    df = pd.DataFrame({"y_pred": y_pred_valid, "y_true": X_valid["label"]})
    df.to_csv(path + "/valid.csv", index=False, header=True)

    # submission
    y_proba = np.zeros((len(X_test["label"]), params["n_runs"]), dtype=np.float32)
    for run in range(params["n_runs"]):
        params["random_seed"] = run
        params["model_name"] = "semantic_model_%s"%str(run+1)
        model = get_model(options.model)(params, logger, init_embedding_matrix=init_embedding_matrix)
        model.fit(X_train, Q, validation_data=None, shuffle=True)
        y_proba[:,run] = model.predict_proba(X_test, Q).flatten()
        df = pd.DataFrame(y_proba[:,:(run+1)], columns=["y_proba_%d"%(i+1) for i in range(run+1)])
        df.to_csv(path + "/test.csv", index=False, header=True)


if __name__ == "__main__":

    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
