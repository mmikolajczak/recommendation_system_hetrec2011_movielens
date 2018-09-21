import tensorflow as tf
from recommendations_system.io_ import load_pickle
from enum import IntEnum
from typing import Sequence
import random
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)


class HetrecFeatureColumns(IntEnum):
    UserId = 0,
    MovieId = 1,
    ActorId = 2,
    Country = 3,
    DirectorId = 4,
    Genre = 5,
    Location = 6


COLUMNS = ['user_id', 'movie_id', 'rating', 'actors', 'country', 'director_id', 'genres', 'locations']
FIELD_DEFAULTS = [[''] for _ in range(len(COLUMNS))]


def initialize_randomness(seed: int):
    """
    Initialises libraries random states with passed seed.
    Note that it's still doesn't guarantee deterministic results (due to parallelism of processing).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def prepare_hetrec_estimator_columns(vocabs_paths: dict, embeddings_dims: dict) -> Sequence[Sequence]:
    """Prepares feature columns based on vocabulary files containing possible feature categories."""
    user_id = tf.feature_column.categorical_column_with_vocabulary_file('user_id', vocabs_paths['users'])
    movie_id = tf.feature_column.categorical_column_with_vocabulary_file('movie_id', vocabs_paths['movies'])
    user_movie_cross = tf.feature_column.crossed_column(['user_id', 'movie_id'], hash_bucket_size=2000)

    # Note: when choosing embeddings dimensions, the rule for determining size is as follows dim = ceil(nb_cat ** 1/4).
    # Note: in case of some columns we actually have multiple values at once and by that not one-hot but rather
    # multi-hot encoding - it is handled by intermediary indicator columns.
    # well, on the other hand in genre, etc. combinations it will be too much so is set emperically
    user_id_emb = tf.feature_column.embedding_column(user_id, dimension=embeddings_dims['users'])
    movie_id_emb = tf.feature_column.embedding_column(movie_id, dimension=embeddings_dims['movies'])
    director_id_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('director_id', vocabs_paths['directors']),
        dimension=embeddings_dims['directors'])
    country_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('country', vocabs_paths['countries']),
        dimension=embeddings_dims['countries'])
    actors_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('actors', vocabs_paths['actors']),
        dimension=embeddings_dims['actors'])
    locations_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('locations', vocabs_paths['locations']),
        dimension=embeddings_dims['locations'])
    genres_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('genres', vocabs_paths['genres']),
        dimension=embeddings_dims['genres'])

    base_columns = []#[user_id, movie_id]
    crossed_columns = [user_movie_cross]
    deep_columns = [user_id_emb, movie_id_emb, director_id_emb, country_emb, actors_emb, locations_emb, genres_emb]
    return base_columns, crossed_columns, deep_columns


def load_train_test_numpy(train_data_path: str, test_data_path: str) -> Sequence[np.ndarray]:
    """Loads previously split train and test data from pickle and divides them into features/labels."""
    rating_idx = 2
    train = load_pickle(train_data_path)
    test = load_pickle(test_data_path)
    X_train = np.delete(train, rating_idx, axis=1)
    y_train = train[:, rating_idx]
    X_test = np.delete(test, rating_idx, axis=1)
    y_test = test[:, rating_idx]
    return X_train, X_test, y_train, y_test


def _csv_list_repr_to_string_list_tensor(x):
    return tf.regex_replace([tf.string_split([tf.regex_replace(x, '[\[\]]', '')], ', ').values], '^\'|\'$', '')


def _parse_line(line):
    """Parsing single line of prepared hetrec csv into features/label tuple."""
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    # pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))
    # transforms
    for feature_name in ('actors', 'genres', 'locations'):
        features[feature_name] = _csv_list_repr_to_string_list_tensor(features[feature_name])
        feature_sh = tf.shape(features[feature_name])
        paddings = [[0, 0], [0, 220 - feature_sh[1]]]
        features[feature_name] = tf.pad(features[feature_name], paddings, constant_values='')
        features[feature_name] = tf.squeeze(features[feature_name])
    label = features.pop('rating')
    label = tf.string_to_number(label)
    return features, label


def input_fn(data_csv_path: str, batch_size: int, nb_epochs: int, shuffle: bool=True) -> tf.data.Dataset:
    """Estimator input function, that sets up generation of data from passed in parameter csv file."""
    ds = tf.data.TextLineDataset(data_csv_path).skip(1)
    ds = ds.map(_parse_line)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.repeat(nb_epochs).batch(batch_size)
    return ds
