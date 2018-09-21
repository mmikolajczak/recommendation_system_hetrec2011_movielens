import time
import numpy as np
import pickle
from recommendations_system.io_ import load_hetrec_to_df, DF2FFMConverter


# SCRIPT CONFIG
SIMPLIFY_TASK = True
SAVE_CONVERTER_PATH = '../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_converter.pkl'
DATA_PATH = '../../data/hetrec2011-movielens-2k-v2/'
FFM_FILE_PATH = '../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_gen_mm.ffm'
USE_COLS = ['userID', 'movieID', 'rating']


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    df = load_hetrec_to_df(DATA_PATH)
    if USE_COLS != 'all':
        df = df[USE_COLS]
    if SIMPLIFY_TASK:
        # Note: we simplify task here, by educing it to binary classification problem. Binarization rule is as follows:
        # 1 (user liked movie) if user rating is above average rating in whole database, 0 (user didn't like the movie)
        # otherwise.
        df['rating'] = (df['rating'] >= df['rating'].mean()).astype(np.uint8)

    converter = DF2FFMConverter()
    converter.fit(df, pred_type='binary', pred_field='rating', nan_const='NaN')

    start = time.perf_counter()
    transformed = converter.transform(df, FFM_FILE_PATH, n_cpus=-1)
    print(f'Data generation time: {time.perf_counter() - start}s')

    if SAVE_CONVERTER_PATH is not None:
        save_pickle(converter, SAVE_CONVERTER_PATH)
