
import numpy as np

import config
from inputs.data import load_raw_question, load_train, load_test
from utils import dist_utils


def forloop_single_func(df, Q, q, funcs):
    feats = []
    for i in range(df.shape[0]):
        s = Q[int(df.iloc[i][q])]
        feat = [func(s) for func in funcs]
        feats.append(np.hstack(feat))
    f = np.array(feats).reshape((df.shape[0],-1))
    return f


def forloop_pairwise_func(df, Q, funcs):
    feats = []
    for i in range(df.shape[0]):
        s1 = Q[int(df.iloc[i]["q1"])]
        s2 = Q[int(df.iloc[i]["q2"])]
        feat = [func(s1, s2) for func in funcs]
        feats.append(np.hstack(feat))
    f = np.array(feats).reshape((df.shape[0],-1))
    return f


def compute_features(df, Q_raw, pairwise_funcs):
    features = [
        forloop_pairwise_func(df, Q_raw["words"], pairwise_funcs),
        forloop_pairwise_func(df, Q_raw["chars"], pairwise_funcs),
    ]
    return np.hstack(features)


if __name__ == "__main__":
    Q_raw = load_raw_question()
    dfTrain = load_train()
    dfTest = load_test()

    pairwise_funcs = [dist_utils._count_stats, dist_utils._edit_dist,
                      dist_utils._longest_match_size, dist_utils._longest_match_size,
                      dist_utils._get_bleu_feat, dist_utils._get_rouge_feat]
    X_train = compute_features(dfTrain, Q_raw, pairwise_funcs)
    X_test = compute_features(dfTest, Q_raw, pairwise_funcs)

    print(X_train.shape)
    print(X_test.shape)
    np.save(config.TRAIN_FEATURES_FILE, X_train)
    np.save(config.TEST_FEATURES_FILE, X_test)
