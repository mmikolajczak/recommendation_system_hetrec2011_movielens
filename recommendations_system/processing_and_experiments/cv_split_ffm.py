import os
import os.path as osp
import numpy as np
from sklearn.model_selection import KFold
# Note: currently ffm split only

# SCRIPT CONFIG
NB_SPLITS = 5
SEED = 42
INPUT_DATA_PATH = '../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_gen_mm.ffm'
OUTPUT_DATA_PATH = f'../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_cv_{NB_SPLITS}'


def load_ffm(path):
    with open(path, 'r', newline='\n') as f:
        data = f.readlines()
    return data


def save_ffm(data, path):
    with open(path, 'w', newline='\n') as f:
        f.writelines(data)


if __name__ == '__main__':
    data = np.array(load_ffm(INPUT_DATA_PATH))
    kf = KFold(NB_SPLITS, shuffle=True, random_state=SEED)
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    for i, (train_idxs, test_idxs) in enumerate(kf.split(data)):
        split_train, split_test = data[train_idxs], data[test_idxs]
        split_subdir_path = osp.join(OUTPUT_DATA_PATH, f'split_{i}')
        os.makedirs(split_subdir_path, exist_ok=True)
        split_train_path = osp.join(split_subdir_path, 'train.ffm')
        split_test_path = osp.join(split_subdir_path, 'test.ffm')
        save_ffm(split_train, split_train_path)
        save_ffm(split_test, split_test_path)
