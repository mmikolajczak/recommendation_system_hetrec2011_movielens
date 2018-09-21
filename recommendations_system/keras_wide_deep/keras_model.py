"""
Keras implementation of wide and deep model along with code for running experiments for Hetrec data.
"""
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.python.keras.callbacks import Callback, EarlyStopping
from recommendations_system.keras_wide_deep.data_generator import KerasSequence
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Generator, Union, Sequence


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
                          deep_part_dims: Sequence[int], use_dropout: bool=False):
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
        if use_dropout:
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
