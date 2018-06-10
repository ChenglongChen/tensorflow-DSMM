
import config
import sys
import numpy as np
import pandas as pd
import pickle as pkl

from keras.preprocessing.sequence import pad_sequences

import utils
from preprocessor import DataProcessor
from model import SemanticMatchingModel


def get_model_data(dataset, params):

    X = {}
    X['id'] = dataset['id'].values
    X["label"] = dataset['label'].values

    # word level
    X['seq_word_left'] = pad_sequences(dataset.seq_word_left, maxlen=params["max_sequence_length_word"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    X["sequence_length_word"] = params["max_sequence_length_word"] * np.ones(dataset.shape[0])

    X['seq_word_right'] = pad_sequences(dataset.seq_word_right, maxlen=params["max_sequence_length_word"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    X["sequence_length_word"] = params["max_sequence_length_word"] * np.ones(dataset.shape[0])

    # char level
    X['seq_char_left'] = pad_sequences(dataset.seq_char_left, maxlen=params["max_sequence_length_char"],
                                       padding=params["pad_sequences_padding"],
                                       truncating=params["pad_sequences_truncating"])
    X["sequence_length_char"] = params["max_sequence_length_char"] * np.ones(dataset.shape[0])

    X['seq_char_right'] = pad_sequences(dataset.seq_char_right, maxlen=params["max_sequence_length_char"],
                                        padding=params["pad_sequences_padding"],
                                        truncating=params["pad_sequences_truncating"])
    X["sequence_length_char"] = params["max_sequence_length_char"] * np.ones(dataset.shape[0])

    return X

params = {
    "offline_model_dir": "./weights/semantic_matching",
    "batch_size": 32,
    "epoch": 5,
    "l2_lambda": 0.0001,

    "embedding_dropout": 0.2,
    "embedding_word_dim": 128,
    "embedding_char_dim": 128,
    "embedding_dim": 128,

    "max_num_words": 10000,
    "max_num_chars": 10000,

    "threshold": 0.217277,

    "max_sequence_length_word": 20,
    "max_sequence_length_char": 30,
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",

    "optimizer_type": "nadam",
    "init_lr": 0.001,
    "beta1": 0.975,
    "beta2": 0.999,
    "decay_steps": 500,
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
    "fc_dim": 64,
    "fc_dropout": 0,
}

model_name = "semantic_matching"

def train():

    utils._makedirs("../logs")
    utils._makedirs("../output")
    logger = utils._get_logger("../logs", "tf-%s.log" % utils._timestamp())


    dfTrain = pd.read_csv(config.TRAIN_FILE, header=None, sep="\t")
    dfTrain.columns = ["id", "left", "right", "label"]

    dfTrain.dropna(inplace=True)

    # shuffle training data
    dfTrain = dfTrain.sample(frac=1.0)

    dp = DataProcessor(max_num_words=params["max_num_words"], max_num_chars=params["max_num_chars"])
    dfTrain = dp.fit_transform(dfTrain)

    N = dfTrain.shape[0]
    train_ratio = 0.6
    train_num = int(N*train_ratio)
    X_train = get_model_data(dfTrain[:train_num], params)
    X_valid = get_model_data(dfTrain[train_num:], params)

    model = SemanticMatchingModel(model_name, params, logger=logger, threshold=0.2)
    model.fit(X_train, validation_data=X_valid, shuffle=False)

    # save model
    model.save_session()
    with open("dp.pkl", "wb") as f:
        pkl.dump((dp, model.threshold), f, protocol=2)


def submit(input_file, output_file):

    print("read %s"%input_file)
    print("write %s"%output_file)

    # load model
    with open("dp.pkl", "rb") as f:
        dp, threshold = pkl.load(f)
    model = SemanticMatchingModel(model_name, params, logger=None, threshold=threshold, training=False)
    model.restore_session()

    dfTest = pd.read_csv(input_file, header=None, sep="\t")
    dfTest.columns = ["id", "left", "right"]
    dfTest["label"] = np.zeros(dfTest.shape[0])

    dfTest = dp.transform(dfTest)
    X_test = get_model_data(dfTest, params)

    dfTest["label"] = model.predict(X_test)

    dfTest[["id", "label"]].to_csv(output_file, header=False, index=False, sep="\t")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        submit(sys.argv[1], sys.argv[2])
    else:
        train()
