import numpy as np
from sklearn.preprocessing import LabelEncoder


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class DF2FFMConverter:
    """
    Class to converting data between pandas DataFrame and ffm format with basic, scikit-like API.
    """

    def __init__(self):
        pass

    def fit(self, X, pred_type='binary', pred_field='', nan_const=np.nan):
        self._fields_map = {col: str(i) for i, col in enumerate(X.columns)}  # both fields and inner features ordered from 0
        self._fields_map_rev = {v: k for k, v in self._fields_map.items()}
        self._fields_types = {}
        self._fields_labelers = {}
        self._pred_type = pred_type
        self._pred_field = pred_field
        self._nan_const = nan_const

        for field in X.columns:
            if field == self._pred_field and self._pred_type == 'binary':
                continue
            field_type = type(X.loc[0, field])
            if is_iterable(field_type()) and field_type != str:
                self._fields_types[field] = 'multi_value'
                self._fields_labelers[field] = LabelEncoder()
                # Note: warning about scalar can can be simply ignored, it does retur bool vector instead of scalar as written
                all_features_vector = X[~(X[field] == self._nan_const)][field]
                all_features_vector = flatten(all_features_vector)
                self._fields_labelers[field].fit(all_features_vector)
            else:
                self._fields_types[field] = 'single_value'
                self._fields_labelers[field] = LabelEncoder()
                try:
                    all_features_vector = X[~(X[field] == self._nan_const)][field]
                except TypeError:  # this means that there is not a single nan value in this column as it has other type than const
                    all_features_vector = X[field]
                self._fields_labelers[field].fit(all_features_vector)
            # TODO: extend it to handle properly other iterable types?
        #handling pred column

    def _row_transform(self, row):
        str_repr = f'{row[self._pred_field]} '
        for k, v in self._fields_map.items():
            if (k == self._pred_field and self._pred_type == 'binary') or row[k] == self._nan_const:  # (first term) delete it from items instead of checking?
                continue
            encoded_cell = self._fields_labelers[k].transform([row[k]] if self._fields_types[k] == 'single_value' else row[k])
            encoded_cell = [f'{v}:{val}:1' for val in encoded_cell]
            str_repr += ' '.join(encoded_cell) + ' '
        return str_repr

    def transform(self, X, save_to=None):
        # impossible to encode it column by column (well, maybe it can be hacked, but rather really hard way), let's see how many times row by row will take
        X_trasformed = X.apply(self._row_transform, axis=1)
        if save_to is not None:
            with open(save_to, 'w') as f:
                f.write('\n'.join(X_trasformed))
        else:
            return X_trasformed

    def inverse_transform(self, X_transformed, restore_from=None):
        pass
