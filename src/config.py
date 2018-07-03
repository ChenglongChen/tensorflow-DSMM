

DATA_DIR = "../data"

TRAIN_FILE = DATA_DIR + "/train.csv"
TEST_FILE = DATA_DIR + "/test.csv"

TRAIN_FEATURES_FILE = DATA_DIR + "/train_features.npy"
TEST_FEATURES_FILE = DATA_DIR + "/test_features.npy"

QUESTION_FILE = DATA_DIR + "/question.csv"

WORD_EMBEDDING_FILE = DATA_DIR + "/word_embed.txt"
CHAR_EMBEDDING_FILE = DATA_DIR + "/char_embed.txt"

SUB_FILE = "submission.csv"
SINGLE_SUB_FILE_PATTERN = "submission_%s_%s.csv"
STACKING_SUB_FILE_PATTERN = "submission_%s.csv"


# missing
MISSING_INDEX_WORD = 20891
PADDING_INDEX_WORD = 20892

MISSING_INDEX_CHAR = 3048
PADDING_INDEX_CHAR = 3049

# ratio
POS_RATIO_OFFLINE = 0.5191087559849992
POS_RATIO_ONLINE = 0.50296075348400959

"""
1/(p0 + p1) * (P0 * (0*log(0+eps) + (1-0)*log(1-0-eps)) + P1 * (1*log(0+eps) + (1-1)*log(1-0-eps))) = 17.371649
1/(p0 + p1) * (p0 * log(1-eps) + p1 * log(0+eps)) = 17.371649
p1/(p0 + p1) ~= 17.371649/log(eps)
              = 17.371649/log(1e-15)
              = 0.50296075348400959
"""