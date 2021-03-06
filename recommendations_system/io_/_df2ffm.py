import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import math
import functools
from multiprocessing import Pool
from . import _utils as utils


class DF2FFMConverter:
    """
    Class to converting data between pandas DataFrame and ffm format with basic, scikit-like API.
    """
    PROCESSES_PER_CPU = 2

    def __init__(self):
        pass

    def fit(self, X, pred_type='binary', pred_field='', nan_const=np.nan):
        self._fields_map = {col: str(i) for i, col in enumerate(X.columns)}
        self._fields_map_rev = {v: k for k, v in self._fields_map.items()}
        self._fields_types = {}
        self._fields_labelers = {}
        self._pred_type = pred_type
        self._pred_field = pred_field
        self._nan_const = nan_const
        if type(X) != pd.DataFrame:
            raise ValueError('Input must be pandas DF')

        for field in X.columns:
            if field == self._pred_field and self._pred_type == 'binary':
                continue
            field_type = type(X.loc[0, field])
            if utils.is_iterable(field_type()) and field_type != str:
                self._fields_types[field] = 'multi_value'
                self._fields_labelers[field] = LabelEncoder()
                # Note: warning about scalar can can be simply ignored, it does retur bool vector instead of scalar as written
                all_features_vector = X[~(X[field] == self._nan_const)][field]
                all_features_vector = utils.flatten(all_features_vector)
                self._fields_labelers[field].fit(all_features_vector)
            else:
                self._fields_types[field] = 'single_value'
                self._fields_labelers[field] = LabelEncoder()
                try:
                    all_features_vector = X[~(X[field] == self._nan_const)][field]
                except TypeError:  # Note: this means that there is not a single nan value in this column as it has other type than const
                    all_features_vector = X[field]
                self._fields_labelers[field].fit(all_features_vector)
            # TODO: extend it to handle properly other iterable types?

    def transform(self, X, save_to=None, n_cpus=1):
        # TODO: further upgrades of conversion, vectorization for columns with single values?
        if type(X) != pd.DataFrame:
            raise ValueError('Input must be pandas DF')

        nb_processes = self.PROCESSES_PER_CPU * (os.cpu_count() if n_cpus == -1 else n_cpus)
        X_transformed = DF2FFMConverter._parallel_apply(X, self._row_transform, nb_processes=nb_processes)

        if save_to is not None:
            with open(save_to, 'w', newline='\n') as f:
                f.write('\n'.join(X_transformed))
        else:
            return X_transformed

    def _row_transform(self, row):
        str_repr = f'{row[self._pred_field]} '
        for k, v in self._fields_map.items():
            if (k == self._pred_field and self._pred_type == 'binary') or row[k] == self._nan_const:
                continue
            encoded_cell = self._fields_labelers[k].transform([row[k]] if self._fields_types[k] == 'single_value' else row[k])
            encoded_cell = [f'{v}:{val}:1' for val in encoded_cell]
            str_repr += ' '.join(encoded_cell) + ' '
        return str_repr

    @staticmethod
    def _apply_wrapper(part, func):
        return part.apply(func, axis=1)

    @staticmethod
    def _parallel_apply(df, func, nb_processes):
        chunk_size = int(math.ceil(len(df) / nb_processes))
        partitions = [df.iloc[i: i + chunk_size] for i in range(0, len(df), chunk_size)]
        pool = Pool(nb_processes)
        fnc = functools.partial(DF2FFMConverter._apply_wrapper, func=func)
        res = pool.map(fnc, partitions)
        res = pd.concat(res)
        pool.close()
        pool.join()
        return res

    def inverse_transform(self, X_transformed, restore_from=None):
        pass
        # TODO: implementation of this method

