


class DF2FFMConverter:
    """
    Class to converting data between pandas DataFrame and ffm format with basic, scikit-like API.
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._fields_map = {col: str(i) for i, col in enumerate(X.columns, 1)}

    def transform(self, X, save_to=None):
        pass

    def inverse_transform(self, X_transformed, restore_from=None):
        pass