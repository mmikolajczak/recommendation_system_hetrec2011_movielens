"""
Performs processing on .ffm file (or files in case of cv) generated earlier from dataset and splits it into all possible
another files, containing all possible feature combinations (avoiding re-generation of ffm format from original data).
"""
import os
import os.path as osp
from itertools import chain, combinations
from collections import namedtuple

# SCRIPT CONFIG
FFM_DATA_PATH = '../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_cv_5'
NB_DATA_FEATURES = 8
MIN_FEATURES = 2
CV = True


def load_ffm(path):
    with open(path, 'r', newline='\n') as f:
        data = f.readlines()
    return data


def save_ffm(data, path):
    with open(path, 'w', newline='\n') as f:
        f.writelines(data)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def load_ffm_cv_data(src):  # not used - too much memory required, although it's longer version that perform reading
                            # every time will be used
    splits_data = []
    for split_dir in os.listdir(src):
        split_train_path = osp.join(src, split_dir, 'train.ffm')
        split_test_path = osp.join(src, split_dir, 'test.ffm')
        split_train_rows = load_ffm(split_train_path)
        split_test_rows = load_ffm(split_test_path)
        splits_data.append((split_train_rows, split_test_rows))
    return splits_data


if __name__ == '__main__':
    possible_combinations = [comb for comb in powerset(range(NB_DATA_FEATURES)) if len(comb) >= MIN_FEATURES]

