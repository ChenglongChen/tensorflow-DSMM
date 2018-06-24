
import config
import numpy as np
import pandas as pd
import scipy as sp
from keras.preprocessing.sequence import pad_sequences


def _to_ind(qid):
    return int(qid[1:])


def load_question(params):
    df = pd.read_csv(config.QUESTION_FILE)
    df["words"] = df.words.str.split(" ").apply(lambda x: [_to_ind(z) for z in x])
    df["chars"] = df.chars.str.split(" ").apply(lambda x: [_to_ind(z) for z in x])
    Q = {}
    Q["seq_len_word"] = sp.minimum(df["words"].apply(len).values, params["max_seq_len_word"])
    Q["seq_len_char"] = sp.minimum(df["chars"].apply(len).values, params["max_seq_len_char"])
    Q["words"] = pad_sequences(df["words"], maxlen=params["max_seq_len_word"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    Q["chars"] = pad_sequences(df["chars"], maxlen=params["max_seq_len_char"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    return Q


def load_train():
    df = pd.read_csv(config.TRAIN_FILE)
    df["q1"] = df.q1.apply(_to_ind)
    df["q2"] = df.q2.apply(_to_ind)
    return df


def load_test():
    df = pd.read_csv(config.TEST_FILE)
    df["q1"] = df.q1.apply(_to_ind)
    df["q2"] = df.q2.apply(_to_ind)
    df["label"] = np.zeros(df.shape[0])
    return df


def load_embedding_matrix(embedding_file):
    print("read embedding from: %s " %embedding_file)
    d = {}
    n = 0
    with open(embedding_file, "r") as f:
        line = f.readline()
        while line:
            n += 1
            w, v = line.strip().split(" ", 1)
            d[int(w[1:])] = v
            line = f.readline()
    dim = len(v.split(" "))

    emb_matrix = np.zeros((n, dim), dtype=float)
    for key ,val in d.items():
        v = np.asarray(val.split(" "), dtype=float)
        emb_matrix[key] = v
    emb_matrix = np.array(emb_matrix, dtype=np.float32)
    return emb_matrix


word_embedding_matrix = load_embedding_matrix(config.WORD_EMBEDDING_FILE)
char_embedding_matrix = load_embedding_matrix(config.CHAR_EMBEDDING_FILE)
init_embedding_matrix = {
    "word": word_embedding_matrix,
    "char": char_embedding_matrix,
}