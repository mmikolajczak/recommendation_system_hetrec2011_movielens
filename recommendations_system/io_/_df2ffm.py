import collections
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


class EmptyValueHandlingLabelEncoder(LabelEncoder):

    def __init__(self, empty_value=None):
        self._empty_value = empty_value
        super().__init__()

    def fit(self, y):
        # assume numpy array input? how about lists? can be handled outside, and called list by list eventually
        y = np.array(y)
        y = y[y != self._empty_value]
        return super().fit(y)

    def transform(self, y):
        pass

    def inverse_transform(self, y):
        pass

# looks like handling it outside the actual label encoder makes more sense


class DF2FFMConverter:
    """
    Class to converting data between pandas DataFrame and ffm format with basic, scikit-like API.
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._fields_map = {col: str(i) for i, col in enumerate(X.columns)}  # both fields and inner features ordered from 0
        self._fields_map_rev = {v: k for k, v in self._fields_map.items()}
        self._fields_types = {}
        self._fields_labelers = {}
        self._empty_values = {}

        for field in X.columns:
            field_type = type(X.loc[0, field])

            if is_iterable(field_type()) and field_type != str:
                self._fields_types[field] = 'multi_value'
                self._empty_values[field] = field_type()
                self._fields_labelers[field] = LabelEncoder()
                print(len(X))
                all_features_vector = X[X[field].isin([self._empty_values[field]])][field]
                print(len(all_features_vector))
                all_features_vector = flatten(all_features_vector)
                self._fields_labelers[field].fit(all_features_vector)
            else:
                self._fields_types[field] = 'single_value'
                self._empty_values[field] = field_type()
                self._fields_labelers[field] = LabelEncoder()
                all_features_vector = X[X[field].isin([self._empty_values[field]])][field]
                self._fields_labelers[field].fit(all_features_vector)
            print(field)
            # problem 1 - types
            # problem 2 - handling nans and other empty values
            # TODO: extend it to handle properly other iterable types?

    def transform(self, X, save_to=None):
        X_trasformed = []

        # actual processing stuff

        if save_to is not None:
            with open(save_to, 'w') as f:
                f.write('\n'.join(X_trasformed))
        else:
            return X_trasformed

    def inverse_transform(self, X_transformed, restore_from=None):
        pass