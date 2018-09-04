import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Embedding, Input, Concatenate
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation
from recommendations_system.tf_wide_deep_v2.data_generator import get_nn_hetrec_data_gen
import numpy as np
from sklearn.metrics import roc_auc_score
from abc import abstractmethod, ABCMeta
from typing import List, Tuple


class InputColumn(metaclass=ABCMeta):

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class WideNDeep:

    def __init__(self, wide_columns: List[InputColumn], deep_columns: List[InputColumn], deep_net_shape: Tuple):
        wide_branch = WideNDeep._build_wide_branch(wide_columns)
        deep_branch = WideNDeep._build_deep_branch(deep_columns, deep_net_shape)
        branches_concat = Concatenate([wide_branch, deep_branch])
        output = Dense(1, activation='sigmoid')(branches_concat)
        self._model = Model(inputs=wide_columns + deep_columns, outputs=output)

    @property
    def model(self):  # dunno
        return self._model

    @staticmethod
    def _build_wide_branch(wide_columns: List[InputColumn]):
        return wide_columns

    @staticmethod
    def _build_deep_branch(deep_columns: List[InputColumn], deep_net_shape):
        deep_net = Concatenate(deep_columns)
        for layer_units in deep_net_shape:
            layer = Dense(layer_units, activation='relu')(deep_net)
            deep_net = layer
        return deep_net

    def fit(self):
        pass

    def predict(self):
        pass


COLS_NB_CATEGORIES = {'actors': 94903, 'countries': 72, 'directors': 4032, 'genres': 20,
                      'locations': 1391, 'movies': 10109, 'users': 2113,
                      'movies_users_cross': 10000}  # cross - after hashing


EMBEDDINGS_DIMENSIONS = {'users': 16,#np.ceil(2113 ** 0.25),
                         'movies': 16,#np.ceil(10109 ** 0.25),
                         'actors': 16,#-1, # embedding here might different as we are dealing with combinations actually
                         'directors': 16,#np.ceil(4032 ** 0.25),
                         'genres': 16,#-1,
                         'countries': 16,#np.ceil(72 ** 0.25),
                         'locations': 16}#-1}


def CategoricalEmbedding(out_dim):
    x = Dense(out_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform')
    return x


def get_model(cols_nb_categories, embeddings_dimensions):
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
    deep_part = Dense(64, activation='relu')(embedding_concat)
    deep_part = Dense(32, activation='relu')(deep_part)
    deep_part = Dense(16, activation='relu')(deep_part)

    # crossing done in generator, with hashing.
    # Note: brute way by default is unacceptable, it would result with 8,5GB batch when using batch_size=100. (when not using sparse matrices)
    # movies_users_cross = Input(shape=(cols_nb_categories['movies_users_cross'], ))
    wide_part = Concatenate(axis=-1)([movies_in, users_in])#, movies_users_cross])

    wide_deep_concat = Concatenate(axis=-1)([deep_part, wide_part])
    wide_deep = Dense(1, activation='sigmoid')(wide_deep_concat)

    model = Model(inputs=[actors_in, countries_in, directors_in, genres_in, locations_in, movies_in, users_in],
                          #movies_users_cross],
                  outputs=[wide_deep])
    return model


# utility
class roc_callback(Callback):

    def __init__(self, val_gen, val_samples, val_batch_size):
        super().__init__()
        self.val_gen = val_gen
        self.val_reports = []
        self.val_samples = val_samples
        self.val_batch_size = val_batch_size

    def on_epoch_end(self, epoch, logs={}):
        X, y = self._val_data_from_gen()
        y_pred = self.model.predict(X)
        y_true = y
        val_roc = roc_auc_score(y_true, y_pred)
        self.val_reports.append(val_roc)
        print(f'Epoch {epoch}, validation auc: {val_roc}')

    def _val_data_from_gen(self):
        X, y = None, None
        for _ in range(self.val_samples // self.val_batch_size):  # might result in problems for values under batch size ; p
            X_batch, y_batch = next(self.val_gen)
            X = np.vstack((X, X_batch)) if X is not None else X_batch
            y = np.vstack((y, y_batch)) if y is not None else y_batch
        return X, y


if __name__ == '__main__':
    #model = WideNDeep.build_model()
    model = get_model(COLS_NB_CATEGORIES, EMBEDDINGS_DIMENSIONS)
    model.summary()
    model.compile(Adam(lr=3e-4), loss='binary_crossentropy')#, metrics=['auc'])
    nb_train_samples = 684478
    nb_validation_samples = 171120
    train_data_gen = get_nn_hetrec_data_gen('../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/train.pkl',
                                      '../../data/generated_categories_vocabs', nb_epochs=20, batch_size=128)
    validation_data_gen = get_nn_hetrec_data_gen(
        '../../data/numpy_converted/heatrec_2011_full_data_full_columns_list_nans_cv_5/split_0/test.pkl',
        '../../data/generated_categories_vocabs', nb_epochs=20, batch_size=128)
    callbacks = [roc_callback(validation_data_gen, nb_validation_samples, 128)]
    model.fit_generator(train_data_gen, epochs=20, steps_per_epoch=nb_train_samples // 128, callbacks=callbacks)
                       # validation_data=validation_data_gen, validation_steps=nb_validation_samples // 128)
    model.save('test.h5')

