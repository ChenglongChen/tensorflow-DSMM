# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for distance computation
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
try:
    import lzma
    import Levenshtein
except:
    pass
from difflib import SequenceMatcher
from rouge import Rouge
from utils import ngram_utils, np_utils


def _edit_dist(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        # d = Levenshtein.ratio(str1, str2)
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d


def _longest_match_size(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def _longest_match_ratio(str1, str2):
    sq = SequenceMatcher(lambda x: x==" ", str1, str2)
    match = sq.find_longest_match(0, len(str1), 0, len(str2))
    return np_utils._try_divide(match.size, min(len(str1), len(str2)))


def _common_num(s1, s2):
    c = 0
    for s1_ in s1:
        for s2_ in s2:
            if s1_ == s2_:
                c += 1
    return c


def _count_stats(s1, s2):
    # length
    l1 = len(s1)
    l2 = len(s2)
    len_diff = np_utils._try_divide(np.abs(l1-l2), (l1+l2)/2.)

    # set
    s1_set = set(s1)
    s2_set = set(s2)

    # unique length
    l1_unique = len(s1_set)
    l2_unique = len(s2_set)
    len_diff_unique = np_utils._try_divide(np.abs(l1_unique-l2_unique), (l1_unique+l2_unique)/2.)

    # unique ratio
    r1_unique = np_utils._try_divide(l1_unique, l1)
    r2_unique = np_utils._try_divide(l2_unique, l2)

    # jaccard coef
    li = len(s1_set.intersection(s2_set))
    lu = len(s1_set.union(s2_set))
    jaccard_coef = np_utils._try_divide(li, lu)

    # dice coef
    dice_coef = np_utils._try_divide(li, l1_unique + l2_unique)

    # common number
    common_ = _common_num(s1, s2)
    common_ratio_avg = np_utils._try_divide(common_, (l1 + l2) / 2.)
    common_ratio_max = np_utils._try_divide(common_, min(l1, l2))
    common_ratio_min = np_utils._try_divide(common_, max(l1, l2))

    # over all features
    f = [l1, l2, len_diff,
         l1_unique, l2_unique, len_diff_unique,
         r1_unique, r2_unique,
         li, lu, jaccard_coef, dice_coef,
         common_, common_ratio_avg, common_ratio_max, common_ratio_min
    ]
    return np.array(f, dtype=np.float32)


rouge = Rouge()
def _get_rouge_feat(s1, s2):
    if isinstance(s1, list):
        s1 = " ".join(s1)
    if isinstance(s2, list):
        s2 = " ".join(s2)
    scores = rouge.get_scores(s1, s2)
    feat = []
    for k,v in scores[0].items():
        feat.extend(v.values())
    return np.array(feat, dtype=np.float32)


def _get_bleu(s1, s2):
    count_dict={}
    count_dict_clip={}
    #1. count for each token at predict sentence side.
    for token in s1:
        if token not in count_dict:
            count_dict[token]=1
        else:
            count_dict[token]=count_dict[token]+1
    count=np.sum([value for key,value in count_dict.items()])

    #2.count for tokens existing in predict sentence for target sentence side.
    for token in s2:
        if token in count_dict:
            if token not in count_dict_clip:
                count_dict_clip[token]=1
            else:
                count_dict_clip[token]=count_dict_clip[token]+1

    #3. clip value to ceiling value for that token
    count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
    count_clip=np.sum([value for key,value in count_dict_clip.items()])
    result=float(count_clip)/(float(count)+0.00000001)
    return result


def _get_bleu_feat(s1, s2, ngrams=3):
    if isinstance(s1, str):
        s1 = s1.split(" ")
    if isinstance(s2, str):
        s2 = s2.split(" ")
    feat = []
    for ngram in range(ngrams+1):
        s1_ngram = ngram_utils._ngrams(s1, ngram+1, "_")
        s2_ngram = ngram_utils._ngrams(s2, ngram+1, "_")
        feat.append(_get_bleu(s1_ngram, s2_ngram))
    return np.array(feat, dtype=np.float32)



if __name__ == "__main__":
    s1 = ["W1", "W2", "W3", "W4", "W10"]
    s2 = ["W1", "W2", "W4", "W6", "W8"]
    print(_count_stats(s1, s2))
    print(_edit_dist(s1, s2))
    print(_longest_match_size(s1, s2))
    print(_longest_match_ratio(s1, s2))
    print(_get_rouge_feat(s1, s2))
    print(_get_bleu_feat(s1, s2))