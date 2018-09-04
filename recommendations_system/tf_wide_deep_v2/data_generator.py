import os
import os.path as osp
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from recommendations_system.io_ import load_pickle
from typing import Dict, Tuple


def encoder_from_vocab_file(vocab_path: str) -> MultiLabelBinarizer:
    with open(vocab_path, encoding='utf-8') as f:
        vocab_vector = np.loadtxt(f, dtype='object', delimiter='&|&', encoding='utf-8')
        # delimiter not present in data itself as the shape is (n rows x 1), but values can contain spaces
        ml_binarizer = MultiLabelBinarizer()
        ml_binarizer.fit(vocab_vector.reshape(-1, 1))
    return ml_binarizer


def encoders_from_vocabs_dir(vocabs_dir_path: str) -> Dict[str, MultiLabelBinarizer]:
    encoders = {}
    for vocab_f in os.listdir(vocabs_dir_path):
        # if 'cross' in vocab_f:  # cross columns are not encoded using Label Binarizer but processed with hashing
        #     continue
        vocab_path = osp.join(vocabs_dir_path, vocab_f)
        feature_name = vocab_f.split('_')[0]
        feature_encoder = encoder_from_vocab_file(vocab_path)
        encoders[feature_name] = feature_encoder
    return encoders


_COL_NAMES_TO_PICKLE_MAPPING = {'users': 0, 'movies': 1, 'rating': 2, 'actors': 3, 'countries': 4,
                                'directors': 5, 'genres': 6, 'locations': 7}


# TODO: comment it a bit better
# TODO: reshapes
def to_binarizer_input(X):
    if type(X[0]) != list:
        if type(X[0]) == int:
            X = X.astype(str)
        X = X.reshape(-1, 1)
    return X


def apply_encoding(X, encoders):
    # assumption: all data in X is to be encoded
    X_encoded = np.empty((len(X), len(encoders)), dtype=np.object)
    for i, encoder in enumerate(encoders):
        encoder_out = encoder.transform(to_binarizer_input(X[:, i]))
        X_encoded[:, i] = to_ndarray_of_ndarrays(encoder_out)
    return X_encoded


def to_ndarray_of_ndarrays(X):
    transformed = np.empty(len(X), dtype=np.object)
    transformed[:] = [np.array(row) for row in X.tolist()]
    return transformed


def to_nn_inputs(X):
    X = [np.array([row for row in X[:, i]]) for i in range(X.shape[1])]
    return X


def cross_product_with_hashing(*columns, nb_buckets):
    assert len(columns) >= 2
    crossed = columns[0].astype(str)
    for col in columns[1:]:
        crossed = np.char.add(crossed, np.char.add('_', col.astype(str)))
    hashed = np.array([hash(el) for el in crossed])
    bucketized = np.abs(hashed) % nb_buckets
    bucketized = bucketized.reshape(-1, 1)
    one_hot = np.zeros((len(bucketized), nb_buckets))
    one_hot[:, bucketized] = 1

    #TODO: fix tmp sol
    # tmp = one_hot.tolist()
    # tmp = [np.array(el, dtype='object') for el in tmp]
    # one_hot = np.array(tmp)
    return one_hot


def get_nn_hetrec_data_gen(data_path: str, vocabs_dir_path: str, nb_epochs: int, batch_size: int, shuffle: bool=True,
                           seed: int=42, input_columns_order: Tuple[str]=('actors', 'countries', 'directors', 'genres',
                                                                          'locations', 'movies', 'users')):
    # if nb_epochs == -1, generator will run indefinitely
    X = load_pickle(data_path)
    y = X[:, _COL_NAMES_TO_PICKLE_MAPPING['rating']]
    X = X[:, [_COL_NAMES_TO_PICKLE_MAPPING[col_name]
              for col_name in input_columns_order]]
    encoders = encoders_from_vocabs_dir(vocabs_dir_path)
    encoders = [encoders[col] for col in input_columns_order]
    rand_state = np.random.RandomState(seed)

    for epoch in range(nb_epochs):
        if shuffle:
            shuffled_idxs = np.arange(len(X))
            rand_state.shuffle(shuffled_idxs)
            X = X[shuffled_idxs]
            y = y[shuffled_idxs]
        for i in range(0, len(X), batch_size):
            if i + batch_size > len(X):
                break  # last elements, not enough for batch - start new epoch
            X_batch = X[i: i + batch_size]
            X_batch_cross_movies_users = cross_product_with_hashing(X_batch[:, input_columns_order.index('movies')],
                                                                  X_batch[:, input_columns_order.index('users')],
                                                                  nb_buckets=10000)
            X_batch = apply_encoding(X_batch, encoders)
            X_batch_cross_movies_users = to_ndarray_of_ndarrays(X_batch_cross_movies_users).reshape(-1, 1)
            X_batch = np.hstack((X_batch, X_batch_cross_movies_users))
            y_batch = y[i: i + batch_size]
            X_batch = to_nn_inputs(X_batch)
            yield X_batch, y_batch


# test zone
# gen = get_nn_hetrec_data_gen('../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/train.pkl',
#                              '../../data/generated_categories_vocabs', nb_epochs=5, batch_size=128)
# import time
# start = time.time()
# for i, (X, y) in enumerate(gen):
#     end = time.time()
#     print(i)
#     print(end - start)
#     start = end
