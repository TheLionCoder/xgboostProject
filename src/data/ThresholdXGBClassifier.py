import xgboost as xgb


class ThresholdXGBClassifier(xgb.XGBClassifier):
    def __init__(self, threshold=0.5,  **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def predict(self, X, *args, **kwargs):
        """"
        Predict with threshold applied to predicted class probabilities.
        """
        proba = self.predict_proba(X, *args, **kwargs)
        return (proba[:, 1] > self.threshold).astype(int)

