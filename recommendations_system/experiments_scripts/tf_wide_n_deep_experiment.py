import os.path as osp
from typing import Dict, Any
import tensorflow as tf
from recommendations_system.tf_wide_deep import initialize_randomness, prepare_hetrec_estimator_columns, input_fn
# Note: due to time/resources constraints CV is not used. Also, evaluation is done on validation set, without test set.
# More appropriate tests (maybe full cv) will be done once some promising architecture parameters are found.
# Architecture/used features can be modified in tf_wide_deep.py


# Configuration zone - start
TRAIN_DATA_PATH = '../../data/csv_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/train.csv'
TEST_DATA_PATH = '../../data/csv_converted/hetrec_2011_full_data_full_columns_cv_5/split_0/test.csv'
HETREC_VOCABS_MAIN_DIR_PATH = '../../data/generated_categories_vocabs'
HETREC_VOCABS_PATHS = {feature_name: osp.join(HETREC_VOCABS_MAIN_DIR_PATH, f'{feature_name}_vocab.txt')
                       for feature_name in ('users', 'movies', 'actors', 'directors', 'genres',
                                            'countries', 'locations')}
MODEL_DIR_PATH = osp.abspath('../../sandbox/tf_models/test23')  # abspath for windows issue with not working tf.saver

SEED = 42
NB_EPOCHS = 25
BATCH_SIZE = 128
# one of potential rules (recommended) for nb_embeddings - #np.ceil(nb_feature_categories ** 0.25)
EMBEDDINGS_DIMENSIONS = {'users': 16,
                         'movies': 16,
                         'actors': 16,
                         'directors': 16,
                         'genres': 16,
                         'countries': 16,
                         'locations': 16}
MODEL_HIPERPARAMS = {'LIN_OPTIM_LR': 1e-2,
                     'LIN_OPTIM_L1': 0.01,
                     'LIN_OPTIM_l2': 0.01,
                     'DNN_OPTIM_LR': 1e-2,
                     'DNN_OPTIM_L1': 0.001,
                     'DNN_OPTIM_L2': 0.001,
                     'DNN_UNITS': [64, 32, 16]}
# Configuration zone - end


def tf_wide_and_deep_experiment(train_data_path: str, test_data_path: str, hetrec_vocabs_paths: Dict[str, str],
                                embeddings_dims: Dict[str, int], model_hiperparams: Dict[str, Any],
                                nb_epochs: int=20, batch_size: int=32, seed: int=42):
    initialize_randomness(seed)
    base_columns, crossed_columns, deep_columns = prepare_hetrec_estimator_columns(hetrec_vocabs_paths, embeddings_dims)

    # config - preventing memory allocation problems.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    estimator_config = tf.estimator.RunConfig(session_config=session_config)

    model = tf.estimator.DNNLinearCombinedClassifier(  # may not work as we don't have dnn columns currently
        # wide settings
        linear_feature_columns=base_columns + crossed_columns,
        linear_optimizer=tf.train.FtrlOptimizer(model_hiperparams['LIN_OPTIM_LR'],
                                                l1_regularization_strength=model_hiperparams['LIN_OPTIM_L1'],
                                                l2_regularization_strength=model_hiperparams['LIN_OPTIM_l2']),
        # deep settings
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=model_hiperparams['DNN_UNITS'],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(model_hiperparams['DNN_OPTIM_LR'],
                                                        l1_regularization_strength=model_hiperparams['DNN_OPTIM_L1'],
                                                        l2_regularization_strength=model_hiperparams['DNN_OPTIM_L2']),
        # warm-start settings
        warm_start_from=None,
        model_dir=MODEL_DIR_PATH,
        config=estimator_config)

    for n in range(nb_epochs):
        model.train(input_fn=lambda: input_fn(train_data_path, batch_size, 1, shuffle=True))
        # Note: on each call of train/eval/predict methods model is recreated so this current form is a bit wasteful.
        eval_results = model.evaluate(input_fn=lambda: input_fn(test_data_path, batch_size, 1, shuffle=True))

        # Display evaluation metrics
        tf.logging.info(f'Results at epoch {(n + 1)} / {nb_epochs}')
        tf.logging.info('-' * 60)
        for key in sorted(eval_results):
            tf.logging.info(f'{key}: {eval_results[key]}')
        tf.logging.info('-' * 60)


if __name__ == '__main__':
    tf_wide_and_deep_experiment(TRAIN_DATA_PATH, TEST_DATA_PATH, HETREC_VOCABS_PATHS, EMBEDDINGS_DIMENSIONS,
                                MODEL_HIPERPARAMS, NB_EPOCHS, BATCH_SIZE, SEED)
