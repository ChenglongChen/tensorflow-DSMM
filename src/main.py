
import config

from inputs.data import load_question, load_train, load_test
from inputs.data import word_embedding_matrix, char_embedding_matrix

from utils import log_utils, os_utils, time_utils
from models.dssm import DSSM, CDSSM
from models.dsmm import DSMM
from models.match_pyramid import MatchPyramid


def get_model_data(dataset, params):
    X = {
        "q1": dataset.q1.values,
        "q2": dataset.q2.values,
        "label": dataset.label.values,
    }
    return X


params = {
    "offline_model_dir": "./weights/semantic_matching",
    "construct_neg": False,
    "batch_size": 256,
    "epoch": 20,
    "l2_lambda": 0.000,

    "embedding_dropout": 0.2,
    "embedding_dim_word": word_embedding_matrix.shape[1],
    "embedding_dim_char": char_embedding_matrix.shape[1],
    "embedding_dim": word_embedding_matrix.shape[1],
    "embedding_trainable": True,

    "max_num_word": word_embedding_matrix.shape[0],
    "max_num_char": char_embedding_matrix.shape[0],

    "threshold": 0.217277,
    "calibration_factor": 1.0,

    "max_sequence_length_word": 12,
    "max_sequence_length_char": 20,
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",

    "data1_psize": 3,
    "data2_psize": 3,

    "optimizer_type": "nadam",
    "init_lr": 0.001,
    "beta1": 0.975,
    "beta2": 0.999,
    "decay_steps": 1000,
    "decay_rate": 0.95,
    "schedule_decay": 0.004,
    "random_seed": 2018,
    "eval_every_num_update": 100,

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
    "mp_filter_sizes": 3,
    "mp_pool_size_word": 4,
    "mp_pool_size_char": 4,
}

model_name = "semantic_matching"


def main():
    os_utils._makedirs("../logs")
    os_utils._makedirs("../output")
    logger = log_utils._get_logger("../logs", "tf-%s.log" % time_utils._timestamp())

    Q = load_question(params)
    dfTrain = load_train()

    # shuffle training data
    dfTrain = dfTrain.sample(frac=1.0)

    # validation
    train_ratio = 0.7
    N = dfTrain.shape[0]
    train_num = int(N * train_ratio)
    X_train = get_model_data(dfTrain[:train_num], params)
    X_valid = get_model_data(dfTrain[train_num:], params)

    model = DSMM(model_name, params, logger, params["threshold"],
                                  params["calibration_factor"],
                                  training=True,
                                  word_embedding_matrix=word_embedding_matrix,
                                  char_embedding_matrix=char_embedding_matrix)
    model.fit(X_train, Q, validation_data=X_valid, shuffle=True)

    dfTest = load_test()
    X_test = get_model_data(dfTest, params)

    dfTest["y_pre"] = model.predict_proba(X_test, Q)
    dfTest[["y_pre"]].to_csv(config.SUB_FILE, header=True, index=False)


if __name__ == "__main__":
    main()
