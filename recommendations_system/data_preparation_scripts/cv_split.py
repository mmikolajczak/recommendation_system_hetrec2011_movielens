import os
import os.path as osp
import numpy as np
import pandas as pd
from recommendations_system.io_ import load_hetrec_to_df, save_pickle, load_ffm, save_ffm
from sklearn.model_selection import KFold

# SCRIPT CONFIG
NB_SPLITS = 5
SEED = 42
INPUT_DATA_PATH = '../../data/hetrec2011-movielens-2k-v2'
OUTPUT_DATA_PATH = f'../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_{NB_SPLITS}'
LIST_NAN_IN_MULTIVAL_COLS = True


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


def _add_list_nan_in_multival_cols(df, inplace=False):
    if not inplace:
        df = df.copy()
    hetrec_multival_columns = ('actorID', 'genre', 'location')
    for col in hetrec_multival_columns:
        nan_rows = df[col] == 'NaN'
        df.loc[nan_rows, col] = pd.Series([[] for _ in range(len(nan_rows))])
    return df


def ffm_cv_split(input_path, output_path, nb_splits, seed):
    data = np.array(load_ffm(input_path))
    kf = KFold(nb_splits, shuffle=True, random_state=seed)
    _split_and_save(kf, data, output_path, save_fnc=save_ffm, extension='ffm')


def load_hetrec_to_np(input_path, use_cols=None, list_nan_in_multival_cols=False):
    df = load_hetrec_to_df(input_path)
    if list_nan_in_multival_cols:
        df = _add_list_nan_in_multival_cols(df)
    # Note: we simplify task here, by educing it to binary classification problem. Binarization rule is as follows:
    # 1 (user liked movie) if user rating is above average rating in whole database, 0 (user didn't like the movie)
    # otherwise.
    df['rating'] = (df['rating'] >= df['rating'].mean()).astype(np.uint8)
    if use_cols is not None:
        df = df[use_cols]
    data = df.values
    return data


def np_cv_split(input_path, output_path, nb_splits, seed, use_cols=None, list_nan_in_multival_cols=False):
    data = load_hetrec_to_np(input_path, use_cols, list_nan_in_multival_cols)
    kf = KFold(nb_splits, shuffle=True, random_state=seed)
    _split_and_save(kf, data, output_path, save_fnc=save_pickle, extension='pkl')


def csv_cv_split(input_path, output_path, nb_splits, seed, use_cols=None, list_nan_in_multival_cols=False):
    data = load_hetrec_to_np(input_path, use_cols, list_nan_in_multival_cols)
    kf = KFold(nb_splits, shuffle=True, random_state=seed)
    _split_and_save(kf, data, output_path, save_fnc=lambda split, path: pd.DataFrame(split).to_csv(path, index=False),
                    extension='csv')


if __name__ == '__main__':
    # ffm_cv_split(INPUT_DATA_PATH, OUTPUT_DATA_PATH, NB_SPLITS, SEED)
    np_cv_split(INPUT_DATA_PATH, OUTPUT_DATA_PATH, NB_SPLITS, SEED, None, LIST_NAN_IN_MULTIVAL_COLS)
