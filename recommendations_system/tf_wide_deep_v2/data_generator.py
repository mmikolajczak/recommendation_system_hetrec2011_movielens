import os
import os.path as osp
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from recommendations_system.io_ import load_pickle
from typing import Dict, Tuple


def encoder_from_vocab_file(vocab_path: str) -> MultiLabelBinarizer:
    with open(vocab_path, encoding='utf-8') as f:
        vocab_vector = np.loadtxt(f, dtype='object', delimiter='&|&', encoding='utf-8', comments=None)
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
    #X_encoded = np.empty((len(X), len(encoders)), dtype=np.object)
    # for i, encoder in enumerate(encoders):
    #     encoder_out = encoder.transform(to_binarizer_input(X[:, i]))
    #    X_encoded[:, i] = to_ndarray_of_ndarrays(encoder_out)
    encoders_out = [encoder.transform(to_binarizer_input(X[:, i]))
                    for i, encoder in enumerate(encoders)]
    X_encoded = np.hstack(encoders_out)
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
    return one_hot


def split_to_multiple_inputs(X, input_columns_order, nb_cols_categories):
    start_idx = 0
    inputs = []
    for col in input_columns_order:
        inputs.append(X[:, start_idx: start_idx + nb_cols_categories[col]])
        start_idx += nb_cols_categories[col]
    return inputs


def get_nn_hetrec_data_gen(data_path: str, vocabs_dir_path: str, nb_epochs: int, batch_size: int, shuffle: bool=True,
                           seed: int=42, input_columns_order: Tuple[str]=('actors', 'countries', 'directors', 'genres',
                                                                          'locations', 'movies', 'users'),
                           nb_cols_categories: Dict[str, int]=None):
    # if nb_epochs == -1, generator will run indefinitely
    assert nb_cols_categories is not None, 'nb_cols_categories must be passed as a parameter'
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
            #X_batch_cross_movies_users = to_ndarray_of_ndarrays(X_batch_cross_movies_users).reshape(-1, 1)
            # X_batch = np.hstack((X_batch, X_batch_cross_movies_users))
            y_batch = y[i: i + batch_size]
            X_batch = split_to_multiple_inputs(X_batch, input_columns_order, nb_cols_categories) + [X_batch_cross_movies_users]
            yield X_batch, y_batch.astype(np.float32)


# test zone
COLS_NB_CATEGORIES = {'actors': 94903, 'countries': 72, 'directors': 4032, 'genres': 20,
                      'locations': 1391, 'movies': 10109, 'users': 2113,
                      'movies_users_cross': 10000}  # cross - after hashing

# gen = get_nn_hetrec_data_gen('../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/train.pkl',
#                              '../../data/generated_categories_vocabs', nb_epochs=5, batch_size=128, nb_cols_categories=COLS_NB_CATEGORIES)
# import time
# start = time.time()
# for i, (X, y) in enumerate(gen):
#     end = time.time()
#     print(i)
#     print(end - start)
#     start = end


# Alternative generator (can leverage multiprocessing)
from tensorflow.python.keras.utils import Sequence


'''
data_path: str, vocabs_dir_path: str, nb_epochs: int, batch_size: int, shuffle: bool=True,
                           seed: int=42, input_columns_order: Tuple[str]=('actors', 'countries', 'directors', 'genres',
                                                                          'locations', 'movies', 'users'),
                           nb_cols_categories: Dict[str, int]=None):
'''


class HetrecSequence(Sequence):

    def __init__(self, data_path: str, vocabs_dir_path: str, batch_size: int, shuffle: bool=True,
                           seed: int=42, input_columns_order: Tuple[str]=('actors', 'countries', 'directors', 'genres',
                                                                          'locations', 'movies', 'users'),
                           nb_cols_categories: Dict[str, int]=None):
        assert nb_cols_categories is not None, 'nb_cols_categories must be passed as a parameter'
        X = load_pickle(data_path)
        y = X[:, _COL_NAMES_TO_PICKLE_MAPPING['rating']]
        X = X[:, [_COL_NAMES_TO_PICKLE_MAPPING[col_name]
                  for col_name in input_columns_order]]
        self.X = X
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rand_state = np.random.RandomState(seed)
        self._input_columns_order = input_columns_order
        encoders = encoders_from_vocabs_dir(vocabs_dir_path)
        self._encoders = [encoders[col] for col in input_columns_order]
        self._nb_cols_categories = nb_cols_categories

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))
        # shouldn't it be just self.X // batch_size, to avoid counting incomplete batch? No, the incomplete batch will
        # be just used instead.

    def __getitem__(self, idx):
        X_batch = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        X_batch_cross_movies_users = cross_product_with_hashing(X_batch[:, self._input_columns_order.index('movies')],
                                                                X_batch[:, self._input_columns_order.index('users')],
                                                                nb_buckets=10000)
        X_batch = apply_encoding(X_batch, self._encoders)
        X_batch = split_to_multiple_inputs(X_batch, self._input_columns_order,
                                           self._nb_cols_categories) + [X_batch_cross_movies_users]
        return X_batch, y_batch

    def on_epoch_end(self):  # Note: won't be called in case of validation
        if self.shuffle:
            idxs = np.arange(len(self.X))
            self._rand_state.shuffle(idxs)
            self.X = self.X[idxs]
            self.y = self.y[idxs]
