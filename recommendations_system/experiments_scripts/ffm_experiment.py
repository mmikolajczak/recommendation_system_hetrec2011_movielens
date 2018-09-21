import natsort
import os
import os.path as osp
from recommendations_system.ffm.ffm import FFM
from sklearn.metrics import roc_auc_score
from recommendations_system.experiments_scripts.plotting import plot_roc_auc


# SCRIPT CONFIG
TRAIN_BINARY_PATH = '../../sandbox/windows_10_64_binaries/ffm-train.exe'
PREDICT_BINARY_PATH = '../../sandbox/windows_10_64_binaries/ffm-predict.exe'
DATA_PATH = '../../data/ffm_converted/heatrec2011_full_data_all_columns_cv_5'
EXPERIMENT_TYPE = 'cv_split'  # possible: cv_split, single_split
OUTPUT_MODEL_PATH = 'model'

# FITTING PARAMS
REGULARIZATION_PARAM = 0.0004
LATENT_FACTORS = 10
EPOCHS = 1
LR = 0.17
NB_THREADS = 8
# Note: in this (https://arxiv.org/pdf/1701.04099.pdf) ffm paper authors use low value of latent factors
# (k =4 or even k=2), to reduce interference time, noting that the trade-off between performance in used metric (NLL)
# is neglible for significant speed boost (especially that they are discussing a real-time big, production system).


def ffm_single_split_experiment(train_binary_path, predict_binary_path, data_path, output_model_path,
                                regularization_param, latent_factors, epochs, lr, nb_threads):
    ffm = FFM(train_binary_path, predict_binary_path)
    train_ffm_path = osp.join(data_path, 'train.ffm')
    test_ffm_path = osp.join(data_path, 'test.ffm')
    ffm.fit(train_ffm_path, output_model_path, l=regularization_param, k=latent_factors, t=epochs, r=lr, s=nb_threads)
    ffm.predict(test_ffm_path, 'tmp.txt')
    y_pred = FFM.pred_file_to_numpy('tmp.txt')
    y_true = FFM.ground_truth_from_ffm_file(test_ffm_path)
    os.remove('tmp.txt')
    auc_ = roc_auc_score(y_true, y_pred)
    print(f'Test auc: {auc_}')
    plot_roc_auc(y_true, y_pred, savepath='roc_auc.png')


def ffm_cv_split_experiment(train_binary_path, predict_binary_path, data_path, output_model_path,
                                regularization_param, latent_factors, epochs, lr, nb_threads):
    ffm = FFM(train_binary_path, predict_binary_path)
    splits_y_true = []
    splits_y_pred = []
    splits_auc = []
    for split_subdir in natsort.natsorted(os.listdir(data_path)):
        train_ffm_path = osp.join(data_path, split_subdir, 'train.ffm')
        test_ffm_path = osp.join(data_path, split_subdir, 'test.ffm')
        ffm.fit(train_ffm_path, output_model_path, l=regularization_param, k=latent_factors, t=epochs,
                r=lr, s=nb_threads)
        ffm.predict(test_ffm_path, 'tmp.txt')
        y_pred = FFM.pred_file_to_numpy('tmp.txt')
        y_true = FFM.ground_truth_from_ffm_file(test_ffm_path)
        os.remove('tmp.txt')
        auc_ = roc_auc_score(y_true, y_pred)
        splits_y_true.append(y_true)
        splits_y_pred.append(y_pred)
        splits_auc.append(auc_)
    plot_roc_auc(splits_y_true, splits_y_pred, 'roc_auc.png')


if __name__ == '__main__':
    if EXPERIMENT_TYPE == 'single_split':
        ffm_single_split_experiment(TRAIN_BINARY_PATH, PREDICT_BINARY_PATH, DATA_PATH, OUTPUT_MODEL_PATH,
                                    REGULARIZATION_PARAM, LATENT_FACTORS, EPOCHS, LR, NB_THREADS)
    elif EXPERIMENT_TYPE == 'cv_split':
        ffm_cv_split_experiment(TRAIN_BINARY_PATH, PREDICT_BINARY_PATH, DATA_PATH, OUTPUT_MODEL_PATH,
                                    REGULARIZATION_PARAM, LATENT_FACTORS, EPOCHS, LR, NB_THREADS)
    else:
        raise ValueError('Incorrect value in EXPERIMENT_TYPE.')
