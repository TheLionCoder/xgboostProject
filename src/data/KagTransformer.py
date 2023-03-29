from sklearn import base
from src.data.Utils import Utils


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    A trasformer for tweaking Kaggle survey data.

    This trasformer takes a Pandas DataFrame containing
    Kaggle survey data as input and returns a new version of
    the DataFrame. The modifications include extracting and
    trasforming certain columns, renaming columns, and
    selecting a subset of columns.

    Args:
        ycol: str, optional
            The name of the column to be used as the target variables.
            If not specified, the target variable will not be set.

    Attributes:
        ycol: str
            The name of the column to be used as the target variable.
    """

    def __init__(self, ycol=None):
        self.ycol = ycol

    def transform(self, x):
        return Utils.tweak_kag(x, Utils.top_n)

    def fit(self, x, y=None):
        return self
