
import sys
import string
import jieba
import numpy as np
import pandas as pd
#from pypinyin import pinyin,lazy_pinyin
from multiprocessing import Pool
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

import config
from topk import top_k_selector

#
user_words = [
'花呗',
'借呗',
'淘宝',
'支付宝'
]
for w in user_words:
    jieba.add_word(w)


def clean(text):
    if sys.version_info < (3,):
        text = text.decode("utf-8")
    text = text.replace("***","*")
    filters = ''
    KERAS_SPLIT = " "
    maketrans = str.maketrans
    TRANSLATE_MAP = maketrans(filters, KERAS_SPLIT * len(filters))
    res = text.translate(TRANSLATE_MAP).strip()
    return res


def get_valid_tokens(sentence):
    return [w.strip() for w in sentence if w.strip()]


def seg_to_list(text, mode="word"):
    text = clean(text)
    token_list = []
    if mode == "char":
        token_list = [text[i] for i in range(len(text))]
    elif mode == "word":
        token_list = jieba.lcut(text, cut_all=False, HMM=False)
    elif mode == "pinyin":
        text = " ".join(jieba.lcut(text))
        token_list = ''.join(lazy_pinyin(text)).split()
    return get_valid_tokens(token_list)


def seg_to_char_list(text):
    return seg_to_list(text, "char")

def df_seg_to_char_list(df):
    return df.apply(seg_to_char_list)


def seg_to_word_list(text):
    return seg_to_list(text, "word")

def df_seg_to_word_list(df):
    return df.apply(seg_to_word_list)


def seg_to_pinyin_list(text):
    return seg_to_list(text, "pinyin")

def df_seg_to_pinyin_list(df):
    return df.apply(seg_to_pinyin_list)


N_JOBS = 4
NUM_PARTITIONS = 32
def parallelize_df_func(df, func, num_partitions=NUM_PARTITIONS, n_jobs=N_JOBS):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


class DataProcessor(object):
    def __init__(self, max_num_words, max_num_chars):
        self.word_index = {}
        self.char_index = {}
        self.max_num_words = max_num_words
        self.max_num_chars = max_num_chars

    def get_word_index(self, words, max_num):
        sorted_voc = top_k_selector.topKFrequent(words, max_num - 1)
        word_index = dict(zip(sorted_voc, range(2, max_num)))
        return word_index

    def word2ind(self, word_lst, vocab):
        vect = []
        for w in word_lst:
            if w in vocab:
                vect.append(vocab[w])
        return vect

    def word_list_to_sequences(self, word_lst):
        return self.word2ind(word_lst, self.word_index)

    def df_word_lst_to_sequences(self, df):
        return df.apply(self.word_list_to_sequences)

    def char_list_to_sequences(self, char_lst):
        return self.word2ind(char_lst, self.char_index)

    def df_char_lst_to_sequences(self, df):
        return df.apply(self.char_list_to_sequences)

    def fit_transform(self, df):

        #### tokenize
        df["seq_word_left"] = df_seg_to_word_list(df["left"])
        df["seq_word_right"] = df_seg_to_word_list(df["right"])
        df["seq_char_left"] = df_seg_to_char_list(df["left"])
        df["seq_char_right"] = df_seg_to_char_list(df["right"])

        ##### word_index
        words = df.seq_word_left.tolist() + df.seq_word_right.tolist()
        self.word_index = self.get_word_index(words, self.max_num_words)

        #### char index
        chars = df.seq_char_left.tolist() + df.seq_char_right.tolist()
        self.char_index = self.get_word_index(chars, self.max_num_chars)

        df["seq_word_left"] = self.df_word_lst_to_sequences(df["seq_word_left"])
        df["seq_word_right"] = self.df_word_lst_to_sequences(df["seq_word_right"])
        df["seq_char_left"] = self.df_char_lst_to_sequences(df["seq_char_left"])
        df["seq_char_right"] = self.df_char_lst_to_sequences(df["seq_char_right"])

        print('Average content word sequence length: {}'.format(df["seq_word_left"].apply(len).mean()))
        print('Average content char sequence length: {}'.format(df["seq_char_left"].apply(len).mean()))

        return df

    def transform(self, df):
        #### tokenize
        df["seq_word_left"] = df_seg_to_word_list(df["left"])
        df["seq_word_right"] = df_seg_to_word_list(df["right"])
        df["seq_char_left"] = df_seg_to_char_list(df["left"])
        df["seq_char_right"] = df_seg_to_char_list(df["right"])
        #### token to index
        df["seq_word_left"] = self.df_word_lst_to_sequences(df["seq_word_left"])
        df["seq_word_right"] = self.df_word_lst_to_sequences(df["seq_word_right"])
        df["seq_char_left"] = self.df_char_lst_to_sequences(df["seq_char_left"])
        df["seq_char_right"] = self.df_char_lst_to_sequences(df["seq_char_right"])

        return df


if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE, header=None, sep="\t")
    df.columns = ["id", "left", "right", "label"]
    dp = DataProcessor(max_num_words=10000, max_num_chars=10000)
    df = dp.fit_transform(df)
    print(df.columns)
    print(df.head())
    print(df.label.value_counts()/df.shape[0])
