import natsort
import os
import os.path as osp
from recommendations_system.ffm.ffm import FFM
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt


# SCRIPT CONFIG
TRAIN_BINARY_PATH = '../../sandbox/windows_10_64_binaries/ffm-train.exe'
PREDICT_BINARY_PATH = '../../sandbox/windows_10_64_binaries/ffm-predict.exe'
DATA_PATH = '../../data/ffm_converted/heatrec2011_full_data_movie_id_user_id_rating_columns_cv_5'
EXPERIMENT_TYPE = 'cv_split'  # possible: cv_split, single_split
OUTPUT_MODEL_PATH = 'model'

# FITTING PARAMS
REGULARIZATION_PARAM = 0.00004
LATENT_FACTORS = 5
EPOCHS = 50
LR = 0.16
NB_THREADS = 1


def plot_roc_auc(y_true, y_pred, savepath=None):
    if type(y_true) in (list, tuple) and type(y_pred) in (list, tuple):
        assert len(y_true) == len(y_pred)
        if len(y_true) > 5:
            raise ValueError('Up to 5 lines supported.')
        colors = ('b', 'g', 'c', 'm', 'y')
        for i, (split_y_true, split_y_pred) in enumerate(zip(y_true, y_pred)):
            fpr, tpr, threshold = roc_curve(split_y_true, split_y_pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Split {i}, AUC = %0.4f' % roc_auc, color=colors[i])
    else:
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


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
    plot_roc_auc(splits_y_true, splits_y_pred, 'auc2.png')


if __name__ == '__main__':
    if EXPERIMENT_TYPE == 'single_split':
        ffm_single_split_experiment(TRAIN_BINARY_PATH, PREDICT_BINARY_PATH, DATA_PATH, OUTPUT_MODEL_PATH,
                                    REGULARIZATION_PARAM, LATENT_FACTORS, EPOCHS, LR, NB_THREADS)
    elif EXPERIMENT_TYPE == 'cv_split':
        ffm_cv_split_experiment(TRAIN_BINARY_PATH, PREDICT_BINARY_PATH, DATA_PATH, OUTPUT_MODEL_PATH,
                                    REGULARIZATION_PARAM, LATENT_FACTORS, EPOCHS, LR, NB_THREADS)
    else:
        raise ValueError('Incorrect value in EXPERIMENT_TYPE.')
