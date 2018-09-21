"""Experiment using wide and deep keras implementation (with replaced optimizers)"""
from typing import Dict, Sequence
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from recommendations_system.keras_wide_deep.data_generator import HetrecSequence
from recommendations_system.keras_wide_deep.keras_model import get_wide_n_deep_model, initialize_randomness, auc_eval
# Note: due to time/resources constraints CV is not used. Also, evaluation is done on validation set, without test set.
# More appropriate tests (maybe full cv) will be done once some promising architecture parameters are found.
# Architecture/used features can be modified in keras_model.py

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
# one of potential rules (recommended) for nb_embeddings - #np.ceil(nb_feature_categories ** 0.25)
EMBEDDINGS_DIMENSIONS = {'users': 32,
                         'movies': 32,
                         'actors': 32,
                         'directors': 32,
                         'genres': 32,
                         'countries': 32,
                         'locations': 32}
DEEP_PART_DIMS = (128, 64, 32, 16)
USE_DROPOUT = True
MODEL_SAVE_PATH = '../../sandbox/keras_models/keras_w_n_d_experiment_7.h5'


def keras_wide_and_deep_experiment(train_data_path: str, test_data_path: str, category_vocabs_path: str,
                                   nb_cols_categories: Dict[str, int], embeddings_dimensions: Dict[str, int],
                                   deep_part_dims: Sequence[int], use_dropout: False, nb_epochs: int,
                                   batch_size: int, lr: float, seed: int = 42, early_stopping_patience: int = 5,
                                   model_save_path: str = None):
    """
    Performs experiment on wide and deep keras model, using passed parameters.
    Trained model is saved afterwards (early stopping included), if model_save_path is provided.
    """
    initialize_randomness(seed)
    train_seq = HetrecSequence(train_data_path, category_vocabs_path, batch_size=batch_size, shuffle=True, seed=seed,
                               nb_cols_categories=nb_cols_categories)
    validation_seq = HetrecSequence(test_data_path, category_vocabs_path, batch_size=batch_size, shuffle=True, seed=seed,
                                    nb_cols_categories=nb_cols_categories)

    model = get_wide_n_deep_model(nb_cols_categories, embeddings_dimensions, deep_part_dims, use_dropout)
    model.compile(Adam(lr=lr), loss='binary_crossentropy')
    model.summary()

    # auc_report_callback = AUCScoreCallback(validation_seq) turned off due to lower training time
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    callbacks = [early_stop_callback] #, auc_report_callback]
    model.fit_generator(train_seq, epochs=nb_epochs, validation_data=validation_seq, callbacks=callbacks)
    if model_save_path is not None:
        model.save(model_save_path)

    print(auc_eval(model, validation_seq))


if __name__ == '__main__':
    keras_wide_and_deep_experiment(TRAIN_DATA_PATH, TEST_DATA_PATH, CATEGORY_VOCABS_PATH, NB_COLS_CATEGORIES,
                                   EMBEDDINGS_DIMENSIONS, DEEP_PART_DIMS, USE_DROPOUT, NB_EPOCHS, BATCH_SIZE, LR, SEED,
                                   EARLY_STOPPING_PATIENCE, MODEL_SAVE_PATH)
