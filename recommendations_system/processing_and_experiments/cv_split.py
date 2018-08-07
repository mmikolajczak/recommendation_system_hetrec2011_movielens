import os
import os.path as osp
import numpy as np
from recommendations_system.io_ import load_hetrec_to_df, load_pickle, save_pickle, load_ffm, save_ffm
from sklearn.model_selection import KFold
# Note: currently ffm split only

# SCRIPT CONFIG
NB_SPLITS = 5
SEED = 42
# INPUT_DATA_PATH = '../../data/ffm_converted/heatrec2011_full_data_full_columns_gen_mm.ffm'
# OUTPUT_DATA_PATH = f'../../data/ffm_converted/heatrec2011_full_data_full_columns_cv_{NB_SPLITS}'
INPUT_DATA_PATH = '../../data/hetrec2011-movielens-2k-v2'
OUTPUT_DATA_PATH = f'../../data/numpy_converted/heatrec_2011_full_data_full_columns_cv_{NB_SPLITS}'


def _split_and_save(kf, data, output_path, save_fnc, extension):
    os.makedirs(output_path, exist_ok=True)
    for i, (train_idxs, test_idxs) in enumerate(kf.split(data)):
        split_train, split_test = data[train_idxs], data[test_idxs]
        split_subdir_path = osp.join(output_path, f'split_{i}')
        os.makedirs(split_subdir_path, exist_ok=True)
        split_train_path = osp.join(split_subdir_path, f'train.{extension}')
        split_test_path = osp.join(split_subdir_path, f'test.{extension}')
        save_fnc(split_train, split_train_path)
        save_fnc(split_test, split_test_path)


def ffm_cv_split(input_path, output_path, nb_splits, seed):
    data = np.array(load_ffm(input_path))
    kf = KFold(nb_splits, shuffle=True, random_state=seed)
    _split_and_save(kf, data, output_path, save_fnc=save_ffm, extension='ffm')


def df_cv_split(input_path, output_path, nb_splits, seed, use_cols=None):
    df = load_hetrec_to_df(input_path)
    # Note: we simplify task here, by educing it to binary classification problem. Binarization rule is as follows:
    # 1 (user liked movie) if user rating is above average rating in whole database, 0 (user didn't like the movie)
    # otherwise.
    df['rating'] = (df['rating'] >= df['rating'].mean()).astype(np.uint8)
    if use_cols is not None:
        df = df[use_cols]
    data = df.values
    kf = KFold(nb_splits, shuffle=True, random_state=seed)
    _split_and_save(kf, data, output_path, save_fnc=save_pickle, extension='pkl')


if __name__ == '__main__':
    # ffm_cv_split(INPUT_DATA_PATH, OUTPUT_DATA_PATH, NB_SPLITS, SEED)
    df_cv_split(INPUT_DATA_PATH, OUTPUT_DATA_PATH, NB_SPLITS, SEED)
