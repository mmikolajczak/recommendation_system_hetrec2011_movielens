import subprocess
import warnings
import os.path as osp
import os
import numpy as np


# Note: libffm doesn't handle relative paths very well, hence abspath used.
class FFM:

    def __init__(self, train_binary_path, predict_binary_path, model_path=None):
        self.train_binary_path = osp.abspath(train_binary_path)
        self.predict_binary_path = osp.abspath(predict_binary_path)
        self.model_path = osp.abspath(model_path) if model_path is not None else None

    def fit(self, X, model_path='model', l=0.00002, k=4, t=15, r=0.2, s=1):
        """
        -l <lambda>: regularization parameter (default 0.00002)
        -k <factor>: number of latent factors (default 4)
        -t <iteration>: number of iterations (default 15)
        -r <eta>: learning rate (default 0.2)
        -s <nr_threads>: number of threads (default 1)
        """
        # validation support?
        warnings.warn('Please note that unix newline format (LF) is required for libffm binaries to work correctly.' +
                      ' Windows (CR LF) will cause the issues.')

        if type(X) != str:
            raise ValueError(f'Improper input type {type(X)}.X must be a path to ffm file.')
        self.model_path = osp.abspath(model_path)
        train_data_abspath = osp.abspath(X)
        cmd = f'{self.train_binary_path} -l {l} -k {k} -t {t} -r {r} -s {s} {train_data_abspath} {self.model_path}'
        proc = subprocess.Popen(cmd)
        proc.wait()
        os.remove(f'{train_data_abspath}.bin')

    def predict(self, X, output_file):
        warnings.warn('Please note that unix newline format (LF) is required for libffm binaries to work correctly.' +
                      ' Windows (CR LF) will cause the issues.')
        if self.model_path is None:
            raise RuntimeError('Model must be fitted first!')
        if type(X) != str:
            raise ValueError(f'Improper input type {type(X)}.X must be a path to ffm file.')

        predicted_data_abspath = osp.abspath(X)
        output_file_abspath = osp.abspath(output_file)

        cmd = f'{self.predict_binary_path} {predicted_data_abspath} {self.model_path} {output_file_abspath}'
        proc = subprocess.Popen(cmd)
        proc.wait()

    @classmethod
    def pred_file_to_numpy(cls, preds_file):
        return np.loadtxt(preds_file)

    @classmethod
    def ground_truth_from_ffm_file(cls, ffm_file):
        with open(ffm_file, 'r') as f:
            labels = [line.split(' ')[0] for line in f]
        return np.array(labels).astype(float)
