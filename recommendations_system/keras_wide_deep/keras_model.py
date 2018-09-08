"""
Keras implementation of wide and deep model along with code for running experiments for Hetrec data.
"""
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
from recommendations_system.keras_wide_deep.data_generator import HetrecSequence, KerasSequence
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Generator, Union, Sequence


# Note: due to time/resources constraints CV is not used. Also, evaluation is done on validation set, without test set.
# More appropriate tests (maybe full cv) will be done once some promising architecture parameters are found.

# Experiments parameters - data
NB_TRAIN_SAMPLES = 684478
NB_VALIDATION_SAMPLES = 171120
NB_COLS_CATEGORIES = {'actors': 94903, 'countries': 72, 'directors': 4032, 'genres': 20,
                      'locations': 1391, 'movies': 10109, 'users': 2113,
                      'movies_users_cross': 10000}  # cross - after hashing
TRAIN_DATA_PATH = '../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/train.pkl'
TEST_DATA_PATH = '../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/test.pkl'
CATEGORY_VOCABS_PATH = '../../data/generated_categories_vocabs'


# Experiments parameters - network
SEED = 42
EARLY_STOPPING_PATIENCE = 2
LR = 3e-4
BATCH_SIZE = 128
NB_EPOCHS = 20
EMBEDDINGS_DIMENSIONS = {'users': 32,#np.ceil(2113 ** 0.25),
                         'movies': 32,#np.ceil(10109 ** 0.25),
                         'actors': 32,#np.ceil(94903 ** 0.25),
                         'directors': 32,#np.ceil(4032 ** 0.25),
                         'genres': 32,#np.ceil(20 ** 0.25),
                         'countries': 32,#np.ceil(72 ** 0.25),
                         'locations': 32}#np.ceil(1391 ** 0.25)}
DEEP_PART_DIMS = (128, 64, 32, 16)
MODEL_SAVE_PATH = '../../sandbox/keras_models/keras_w_n_d_experiment_7.h5'  # experiments parameters are described in separate csv


def initialize_randomness(seed: int):
    """
    Initialises libraries random states with passed seed.
    Note that it's still doesn't guarantee deterministic results (due to parallelism of processing).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def CategoricalEmbedding(out_dim: int):
    """
    Custom Categorical Embedding layer. Keras build in Embedding layer is not used as it is designed for sequences.
    """
    x = Dense(out_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform')
    return x


def get_wide_n_deep_model(cols_nb_categories: Dict[str, int], embeddings_dimensions: Dict[str, int],
                          deep_part_dims: Sequence[int]):
    """
    Creates and returns wide and deep model. Embeddings and deep part of network are parametrized.
    """
    assert len(deep_part_dims) > 1
    actors_in = Input(shape=(cols_nb_categories['actors'], ))
    actors_emb = CategoricalEmbedding(embeddings_dimensions['actors'])(actors_in)
    countries_in = Input(shape=(cols_nb_categories['countries'], ))
    countries_emb = CategoricalEmbedding(embeddings_dimensions['countries'])(countries_in)
    directors_in = Input(shape=(cols_nb_categories['directors'], ))
    directors_emb = CategoricalEmbedding(embeddings_dimensions['directors'])(directors_in)
    genres_in = Input(shape=(cols_nb_categories['genres'], ))
    genres_emb = CategoricalEmbedding(embeddings_dimensions['genres'])(genres_in)
    locations_in = Input(shape=(cols_nb_categories['locations'], ))
    locations_emb = CategoricalEmbedding(embeddings_dimensions['locations'])(locations_in)
    movies_in = Input(shape=(cols_nb_categories['movies'], ))
    movies_emb = CategoricalEmbedding(embeddings_dimensions['movies'])(movies_in)
    users_in = Input(shape=(cols_nb_categories['users'], ))
    users_emb = CategoricalEmbedding(embeddings_dimensions['users'])(users_in)
    embedding_concat = Concatenate(axis=-1)([actors_emb, countries_emb, directors_emb, genres_emb, locations_emb,
                                             movies_emb, users_emb])

    deep_part = Dense(deep_part_dims[0], activation='relu')(embedding_concat)
    for dim in deep_part_dims[1:]:
        deep_part = Dense(dim, activation='relu')(deep_part)
        deep_part = Dropout(0.5)(deep_part)

    # crossing done in generator, with hashing.
    # Note: brute way by default is unacceptable, it would result with 8,5GB batch when using batch_size=100.
    # (when not using sparse matrices)
    movies_users_cross = Input(shape=(cols_nb_categories['movies_users_cross'], ))
    wide_part = Concatenate(axis=-1)([movies_in, users_in, movies_users_cross])

    wide_deep_concat = Concatenate(axis=-1)([deep_part, wide_part])
    wide_deep = Dense(1, activation='sigmoid')(wide_deep_concat)

    model = Model(inputs=[actors_in, countries_in, directors_in, genres_in, locations_in, movies_in, users_in,
                          movies_users_cross],
                  outputs=[wide_deep])
    return model


def auc_eval(model: Model, data_gen: Union[Generator, KerasSequence]):
    """
    Evaluates model on data from passed generator. Returns achieved AUC.
    """
    y, y_pred = [], []
    for X_batch, y_batch in data_gen:
        y_pred_batch = model.predict(X_batch).ravel()
        y_pred.append(y_pred_batch)
        y.append(y_batch)
    y_pred = np.hstack(y_pred)
    y = np.hstack(y)
    auc = roc_auc_score(y, y_pred)
    return auc


class AUCScoreCallback(Callback):
    """
    Callback reporting achieved AUC on passed generator at the end of the epoch.
    """
    def __init__(self, data_gen: Union[Generator, KerasSequence]):
        super().__init__()
        self.data_gen = data_gen
        self.reports = []

    def on_epoch_end(self, epoch: int, logs: dict={}):
        auc = auc_eval(self.model, self.data_gen)
        self.reports.append(auc)
        print(f'\n\nEpoch {epoch + 1}, validation auc: {auc}\n\n')


def keras_wide_and_deep_experiment(train_data_path: str, test_data_path: str, category_vocabs_path: str,
                                   nb_cols_categories: Dict[str, int], embeddings_dimensions: Dict[str, int],
                                   deep_part_dims: Sequence[int], nb_epochs: int, batch_size: int, lr: float,
                                   seed: int = 42, early_stopping_patience: int = 5, model_save_path: str = None):
    """
    Performs experiment on wide and deep keras model, using passed parameters.
    Trained model is saved afterwards (early stopping included), if model_save_path is provided.
    """
    initialize_randomness(seed)
    train_seq = HetrecSequence(train_data_path, category_vocabs_path, batch_size=batch_size, shuffle=True, seed=seed,
                               nb_cols_categories=nb_cols_categories)
    validation_seq = HetrecSequence(test_data_path, category_vocabs_path, batch_size=batch_size, shuffle=True, seed=seed,
                                    nb_cols_categories=nb_cols_categories)

    model = get_wide_n_deep_model(nb_cols_categories, embeddings_dimensions, deep_part_dims)
    model.compile(Adam(lr=lr), loss='binary_crossentropy')
    model.summary()

    # auc_report_callback = AUCScoreCallback(validation_seq) turned off due to lower training time
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    callbacks = [early_stop_callback] #, auc_report_callback]
    model.fit_generator(train_seq, epochs=nb_epochs, validation_data=validation_seq)#, callbacks=callbacks)
    if model_save_path is not None:
        model.save(model_save_path)

    print(auc_eval(model, validation_seq))


if __name__ == '__main__':
    keras_wide_and_deep_experiment(TRAIN_DATA_PATH, TEST_DATA_PATH, CATEGORY_VOCABS_PATH, NB_COLS_CATEGORIES,
                                   EMBEDDINGS_DIMENSIONS, DEEP_PART_DIMS, NB_EPOCHS, BATCH_SIZE, LR, SEED,
                                   EARLY_STOPPING_PATIENCE, MODEL_SAVE_PATH)
