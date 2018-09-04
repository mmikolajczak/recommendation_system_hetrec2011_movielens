import tensorflow as tf
from recommendations_system.io_ import load_hetrec_to_df, load_pickle
from enum import IntEnum
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import os.path as osp
tf.logging.set_verbosity(tf.logging.INFO)
# Initial goal: run it on movie/users id only, should be able achieve at least 0.75 auc if everything works well
# Future problem: how to represent features with multiple values? (solution (untested): cat_column -> indicator column -> embedding)


class HetrecFeatureColumns(IntEnum):
    UserId = 0,
    MovieId = 1,
    ActorId = 2,
    Country = 3,
    DirectorId = 4,
    Genre = 5,
    Location = 6


HETREC_VOCABS_MAIN_DIR_PATH = '../../data/generated_categories_vocabs'
HETREC_VOCABS_PATHS = {feature_name: osp.join(HETREC_VOCABS_MAIN_DIR_PATH, f'{feature_name}_vocab.txt')
                       for feature_name in ('users', 'movies', 'actors', 'directors', 'genres',
                                            'countries', 'locations')}
EMBEDDINGS_DIMENSIONS = {'users': 16,#np.ceil(2113 ** 0.25),
                         'movies': 16,#np.ceil(10109 ** 0.25),
                         'actors': 16,#-1, # embedding here might different as we are dealing with combinations actually
                         'directors': 16,#np.ceil(4032 ** 0.25),
                         'genres': 16,#-1,
                         'countries': 16,#np.ceil(72 ** 0.25),
                         'locations': 16}#-1}
print(EMBEDDINGS_DIMENSIONS)
MODEL_HIPERPARAMS = {}


def prepare_hetrec_estimator_columns(vocabs_paths: dict):
    user_id = tf.feature_column.categorical_column_with_vocabulary_file('user_id', vocabs_paths['users'])
    movie_id = tf.feature_column.categorical_column_with_vocabulary_file('movie_id', vocabs_paths['movies'])
    user_movie_cross = tf.feature_column.crossed_column(['user_id', 'movie_id'], hash_bucket_size=2000)

    # Note: when choosing embeddings dimensions, the rule for determining size is as follows dim = ceil(nb_cat ** 1/4).
    # Note: in case of some columns we actually have multiple values at once and by that not one-hot but rather
    # multi-hot encoding - it is handled by intermediary indicator columns.
    # wll, on the other hand in genre, etc. combinations it will be too mych so is set emperically
    user_id_emb = tf.feature_column.embedding_column(user_id, dimension=EMBEDDINGS_DIMENSIONS['users'])
    movie_id_emb = tf.feature_column.embedding_column(movie_id, dimension=EMBEDDINGS_DIMENSIONS['movies'])
    director_id_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('director_id', vocabs_paths['directors']),
        dimension=EMBEDDINGS_DIMENSIONS['directors'])
    country_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('country', vocabs_paths['countries']),
        dimension=EMBEDDINGS_DIMENSIONS['countries'])
    actors_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('actors', vocabs_paths['actors']),
        dimension=EMBEDDINGS_DIMENSIONS['actors'])
    locations_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('locations', vocabs_paths['locations']),
        dimension=EMBEDDINGS_DIMENSIONS['locations'])
    genres_emb = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file('genres', vocabs_paths['genres']),
        dimension=EMBEDDINGS_DIMENSIONS['genres'])
    # actors_emb = tf.feature_column.embedding_column(tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_file('actors', vocabs_paths['actors'])),
    #     dimension=EMBEDDINGS_DIMENSIONS['actors'])
    # locations_emb = tf.feature_column.embedding_column(tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_file('locations', vocabs_paths['locations'])),
    #     dimension=EMBEDDINGS_DIMENSIONS['locations'])
    # genres_emb = tf.feature_column.embedding_column(tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_file('genres', vocabs_paths['genres'])),
    #     dimension=EMBEDDINGS_DIMENSIONS['genres'])
    # mało tych embedingów, niecałe 30 ; p
    # w paperze googla maja 32 na kazda kategoryczna

    # actors = None  # cat + indicator + embedding
    # countries = None  # cat + indicator + embedding
    # genres = None  # cat + indicator + embedding
    # locations = None  # cat + indicator + embedding

    base_columns = []#[user_id, movie_id]
    crossed_columns = [user_movie_cross]
    deep_columns = [user_id_emb, movie_id_emb, director_id_emb, country_emb, actors_emb, locations_emb, genres_emb]
    return base_columns, crossed_columns, deep_columns


def load_train_test_numpy(train_data_path, test_data_path):
    rating_idx = 2
    train = load_pickle(train_data_path)
    test = load_pickle(test_data_path)
    X_train = np.delete(train, rating_idx, axis=1)
    y_train = train[:, rating_idx]
    X_test = np.delete(test, rating_idx, axis=1)
    y_test = test[:, rating_idx]
    return X_train, X_test, y_train, y_test


def input_fn_train(x, y, batch_size, nb_epochs, shuffle=True):
    x = {'user_id': x[:, HetrecFeatureColumns.UserId].astype('<U'),
         'movie_id': x[:, HetrecFeatureColumns.MovieId].astype('<U'),
         'director_id': x[:, HetrecFeatureColumns.DirectorId].astype('<U'),
         'country': x[:, HetrecFeatureColumns.Country].astype('<U')}
    return tf.estimator.inputs.numpy_input_fn(x=x, y=y.astype(int).ravel(), batch_size=batch_size, num_epochs=nb_epochs, shuffle=shuffle)


def input_fn_eval(x, y, batch_size, nb_epochs, shuffle=True):  # duplicate of train, just the naming is different (future change candidate)
    x = {'user_id': x[:, HetrecFeatureColumns.UserId].astype('<U'),
         'movie_id': x[:, HetrecFeatureColumns.MovieId].astype('<U'),
         'director_id': x[:, HetrecFeatureColumns.DirectorId].astype('<U'),
         'country': x[:, HetrecFeatureColumns.Country].astype('<U')}
    return tf.estimator.inputs.numpy_input_fn(x=x, y=y.astype(int).ravel(), batch_size=batch_size, num_epochs=nb_epochs, shuffle=shuffle)


def input_fn_predict(x):
    x = {'user_id': x[:, HetrecFeatureColumns.UserId].astype('<U'),
         'movie_id': x[:, HetrecFeatureColumns.MovieId].astype('<U'),
         'director_id': x[:, HetrecFeatureColumns.DirectorId].astype('<U'),
         'country': x[:, HetrecFeatureColumns.Country].astype('<U')}
    return tf.estimator.inputs.numpy_input_fn(x=x, y=None, shuffle=False)


#DBG - START

COLUMNS = ['user_id', 'movie_id', 'rating', 'actors', 'country', 'director_id', 'genres', 'locations']
FIELD_DEFAULTS = [[''] for _ in range(len(COLUMNS))]


def _csv_list_repr_to_string_list_tensor(x):
    return tf.regex_replace([tf.string_split([tf.regex_replace(x, '[\[\]]', '')], ', ').values], '^\'|\'$', '')


features_padding_dict = {'actors': 220, 'genres': 8, 'locations': 39}


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))
    # Separate the label from the features
    # transform
    for feature_name in ('actors', 'genres', 'locations'):
        features[feature_name] = _csv_list_repr_to_string_list_tensor(features[feature_name])
        paddings = [[0, 0], [0, features_padding_dict[feature_name] - tf.shape(features[feature_name])[0]]]
        features[feature_name] = tf.pad(features[feature_name], paddings, constant_values='')
    label = features.pop('rating')
    label = tf.string_to_number(label)
    return features, label


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))
    # Separate the label from the features
    # transform
    for feature_name in ('actors', 'genres', 'locations'):
        features[feature_name] = _csv_list_repr_to_string_list_tensor(features[feature_name])
        feature_sh = tf.shape(features[feature_name])
        paddings = [[0, 0], [0, 220 - feature_sh[1]]]
        features[feature_name] = tf.pad(features[feature_name], paddings, constant_values='')
        features[feature_name] = tf.squeeze(features[feature_name])
    label = features.pop('rating')
    label = tf.string_to_number(label)
    return features, label

def input_fn():
    ds = tf.data.TextLineDataset(TRAIN_DATA_PATH).skip(1)
    ds = ds.map(_parse_line)
    ds = ds.shuffle(1000).repeat().batch(1)
    return ds


def input_fn_train(x, y, batch_size, nb_epochs, shuffle=True):
    ds = tf.data.TextLineDataset(TRAIN_DATA_PATH).skip(1)
    ds = ds.map(_parse_line)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.repeat(nb_epochs).batch(batch_size)
    return ds


def input_fn_eval(x, y, batch_size, nb_epochs, shuffle=True):
    ds = tf.data.TextLineDataset(TEST_DATA_PATH).skip(1)
    ds = ds.map(_parse_line)  # test - unnecessary
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.repeat(nb_epochs).batch(batch_size)
    return ds

# DBG - END


# Jebnąć to wszystko i zrobić własny indicator column? ; p

if __name__ == '__main__':
    #TRAIN_DATA_PATH = '../../data/numpy_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/train.pkl'
    #TEST_DATA_PATH = '../../data/numpy_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/test.pkl'
    TRAIN_DATA_PATH = '../../data/csv_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/train.csv'
    TEST_DATA_PATH = '../../data/csv_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/test.csv'
    MODEL_DIR_PATH = os.path.abspath('./model_user_id_move_id_rating_test')
    SEED = 42
    NB_EPOCHS = 25
    BATCH_SIZE = 128

    tf.set_random_seed(SEED)

    #X_train, X_test, y_train, y_test = load_train_test_numpy(TRAIN_DATA_PATH, TEST_DATA_PATH)
    base_columns, crossed_columns, deep_columns = prepare_hetrec_estimator_columns(HETREC_VOCABS_PATHS)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

    # config - fighting memory allocation problems.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(session_config=session_config)
    #my_estimator = tf.estimator.Estimator(..., config=estimator_config)

    #deep model
    model = tf.estimator.DNNLinearCombinedClassifier(  # may not work as we don't have dnn columns currently
        # wide settings
        linear_feature_columns=base_columns + crossed_columns,
        linear_optimizer=tf.train.FtrlOptimizer(1e-2,
                                                l1_regularization_strength=0.01,
                                                l2_regularization_strength=0.01),
        # deep settings
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[64, 32, 16],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(1e-2,
                                                        l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        # warm-start settings
        warm_start_from=None,
        model_dir='models/test20',
        config=estimator_config)
    # # To apply L1 and L2 regularization, you can set optimizers as follows:
    # tf.train.ProximalAdagradOptimizer(
    #     learning_rate=0.1,
    #     l1_regularization_strength=0.001,
    #     l2_regularization_strength=0.001)
    # # It is same for FtrlOptimizer.

    X_train, y_train, X_test, y_test = [None] * 4
    for n in range(NB_EPOCHS):
        model.train(input_fn=lambda: input_fn_train(X_train, y_train, BATCH_SIZE, 1, shuffle=True))
        # Note: on each call of train/eval/predict methods model is recreated so this current form is rather bad.
        eval_results = model.evaluate(input_fn=lambda: input_fn_eval(X_test, y_test, BATCH_SIZE, 1, shuffle=True))
        #pred_results = model.predict(input_fn=input_fn_predict(X_test))
        #y_pred = np.array([row['probabilities'][1] for row in list(pred_results)])
        #auc_ = roc_auc_score(y_test.astype(np.float32), y_pred)  # auc is included by evaluate

        # Display evaluation metrics
        tf.logging.info(f'Results at epoch {(n + 1)} / {NB_EPOCHS}')
        tf.logging.info('-' * 60)
        for key in sorted(eval_results):
            tf.logging.info(f'{key}: {eval_results[key]}')
        tf.logging.info('-' * 60)

        # benchmark_logger.log_evaluation_result(results)
        # if early_stop and model_helpers.past_stop_threshold(
        #         flags_obj.stop_threshold, results['accuracy']):
        #     break

    # Notes - models:
    # model2 and 3 (not sure though, one of them is certain) - wide only model. base + crossed features
    # model5 - added categorical columns + embeddings, excluding multihot features
    # model6 - model 5 but lr changed from 3e-4 -> 1e-3
    # model7 - deep net reduce 2x, lr reduced 3x, epochs set to 10 to see potential results quicker
    # model8 - regularization added
    # model9 -> next lr reduction? + trained for additional 10 epochs
    # model10 - fixed embedding, dim 16
    # modl 11 -> neurons amount lowered again, also nb epochs +5 (+10 later)
    # model 12 -> neurons reduced again
    # model 13 - model12 copy with additional epochs (20) (first ten maybe ok, later look like it starts to overfit.
    # model 14 - further neurons reduction (/2)  (caps at 80.4)
    # model 15 - embeddings i original, lower size
    # model 16 - test, adding rest of features
    # 17 - 19 various tests
    # model 20 - looking-legit model with all columns
    # model 21 - base columns dropped (they were dropped also at 20, right?)

    # to test - reduce number of neurons in deep part
    # make embedding dimensions bigger
    # add regularization
    # make lr even bigger than now
    # to multihot it is highly possible that the padding will be needed
